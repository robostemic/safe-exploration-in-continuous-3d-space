import os
import time
import numpy as np
import gymnasium as gym
import pybullet as p

from PyFlyt.gym_envs.quadx_envs.quadx_hover_env import QuadXHoverEnv

from agents import HeuristicAgent, PPOAgent, DQNAgent, DDPGAgent
from discretize_wrapper import DiscretizeActionWrapper


class QuadXObstacleCourseEnv(QuadXHoverEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Phase One goal: Center of the gate
        self.gate_center_pos = np.array([1.5, 0.0, 1.5], dtype=np.float32)
        # Phase Two goal: Final goal past the gate
        self.target_pos = np.array([3.0, 0.0, 1.5], dtype=np.float32)
        
        self.gate_x = 1.5
        self.max_episode_steps = 350

        self.door_object_ids = []
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

    def _build_door_geometry(self, client_id):
        if len(self.door_object_ids) > 0:
            try:
                p.getBodyInfo(self.door_object_ids[0], physicsClientId=client_id)
            except Exception:
                self.door_object_ids.clear()

        if len(self.door_object_ids) == 0:
            wall_thickness = 0.05
            wall_color = [0.9, 0.1, 0.1, 1.0]
            corridor_color = [0.4, 0.4, 0.4, 1.0]

            # Left side wall
            col_left = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, 1.0, 1.5], physicsClientId=client_id)
            vis_left = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, 1.0, 1.5], rgbaColor=wall_color, physicsClientId=client_id)
            id_left = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_left, baseVisualShapeIndex=vis_left, basePosition=[1.5, 2.0, 1.5], physicsClientId=client_id)
            self.door_object_ids.append(id_left)
            
            # Right side wall
            col_right = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, 1.0, 1.5], physicsClientId=client_id)
            vis_right = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, 1.0, 1.5], rgbaColor=wall_color, physicsClientId=client_id)
            id_right = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_right, baseVisualShapeIndex=vis_right, basePosition=[1.5, -2.0, 1.5], physicsClientId=client_id)
            self.door_object_ids.append(id_right)
            
            # Ceiling to prevent the drones from flying out (they kept exploiting a loophole and I was tired of it)
            col_top = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, 1.0, 0.75], physicsClientId=client_id)
            vis_top = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, 1.0, 0.75], rgbaColor=wall_color, physicsClientId=client_id)
            id_top = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_top, baseVisualShapeIndex=vis_top, basePosition=[1.5, 0.0, 3.25], physicsClientId=client_id)
            self.door_object_ids.append(id_top)

            # Floor
            col_bot = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness, 1.0, 0.25], physicsClientId=client_id)
            vis_bot = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness, 1.0, 0.25], rgbaColor=wall_color, physicsClientId=client_id)
            id_bot = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_bot, baseVisualShapeIndex=vis_bot, basePosition=[1.5, 0.0, 0.25], physicsClientId=client_id)
            self.door_object_ids.append(id_bot)

            # Long left corridor wall
            col_side_l = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2.5, wall_thickness, 1.5], physicsClientId=client_id)
            vis_side_l = p.createVisualShape(p.GEOM_BOX, halfExtents=[2.5, wall_thickness, 1.5], rgbaColor=corridor_color, physicsClientId=client_id)
            id_side_l = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_side_l, baseVisualShapeIndex=vis_side_l, basePosition=[1.5, 3.0, 1.5], physicsClientId=client_id)
            self.door_object_ids.append(id_side_l)

            # Long right corridor wall
            col_side_r = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2.5, wall_thickness, 1.5], physicsClientId=client_id)
            vis_side_r = p.createVisualShape(p.GEOM_BOX, halfExtents=[2.5, wall_thickness, 1.5], rgbaColor=corridor_color, physicsClientId=client_id)
            id_side_r = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col_side_r, baseVisualShapeIndex=vis_side_r, basePosition=[1.5, -3.0, 1.5], physicsClientId=client_id)
            self.door_object_ids.append(id_side_r)

    def reset(self, seed=None, options=None):
        #self.door_object_ids.clear()

        obs_flat, info = super().reset(seed=seed, options=options)

        self.passed_gate = False
        self.current_step_count = 0
        
        try: 
            client_id = self.env._client
        except Exception: 
            client_id = getattr(self, '_client', None)
        
        if client_id is not None:
            self._build_door_geometry(client_id)
            
        obs_flat = self._get_custom_obs()
        self.prev_dist_to_target = np.linalg.norm(np.array([1.5, 0.0, 1.5]) - obs_flat[0:3])

        return obs_flat, {}

    def _get_custom_obs(self):
        attitude = np.array(self.state, dtype=np.float32).flatten()
        drone_pos = attitude[0:3]
        
        if not self.passed_gate:
            current_waypoint = np.array([1.5, 0.0, 1.5], dtype=np.float32)
        else:
            current_waypoint = self.target_pos
            
        waypoint_delta = current_waypoint - drone_pos
        final_destination_delta = self.target_pos - drone_pos

        lidar_distances = np.ones(5, dtype=np.float32) * 4

        try:
            client_id = self.env._client
        except Exception:
            client_id = getattr(self, "_client", None)

        if client_id is not None and len(self.env.drones) > 0:
            pos, orn = p.getBasePositionAndOrientation(self.env.drones[0].Id, physicsClientId=client_id)
            rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

            angles = [-0.52, -0.26, 0.0, 0.26, 0.52]
            ray_from = []
            ray_to = []

            for angle in angles:
                local_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
                global_dir = rot_matrix.dot(local_dir)
                
                ray_from.append(pos)
                ray_to.append(pos + global_dir * 4.0)

            results = p.rayTestBatch(ray_from, ray_to, physicsClientId=client_id)

            for i, res in enumerate(results):
                # 
                if isinstance(res, (list, tuple)) and len(res) > 2:
                    hit_fraction = res[2]
                else:
                    hit_fraction = res if isinstance(res, (float, int)) else 1.0
                lidar_distances[i] = hit_fraction * 4.0
    
        flat_obs = np.zeros(24, dtype=np.float32)
        # Telemetry
        flat_obs[0:12] = attitude[0:12]
        # Relative Distance to Relative Goal (Phase 1, Phase 2)
        flat_obs[12:15] = waypoint_delta[0:3]
        # Final Target Radar
        flat_obs[15:18] = final_destination_delta
        # Phase flag - Did it pass gate yet or no?
        flat_obs[18] = 1.0 if self.passed_gate else 0.0
        # LiDAR
        flat_obs[19:24] = lidar_distances

        return flat_obs

    def step(self, action):
        processed_action = np.array(action, dtype=np.float32).flatten()
        processed_action = np.clip(processed_action + 0.56, 0.0, 1.0)
        processed_action[:3] = np.clip(processed_action[:3], -1.0, 1.0)

        try:
            raw_obs, _, terminated, truncated, info = super().step(processed_action)
        except IndexError:
            terminated = True
            truncated = False
            info = {}
            print("Internal physics matrix mismatch caught.")
            
        self.current_step_count += 1
        
        obs_flat = self._get_custom_obs()
        drone_pos = obs_flat[0:3]
        
        # STOP FLYING UP OUT OF THE SIMULATION YOU GUYS.
        ceiling_penalty = 0.0
        if drone_pos[2] > 2.3:
            terminated = True
            ceiling_penalty = -300.0
            print("Terminated: Drone tried to fly over the wall")

        # Gate rewards
        gate_reward = 0.0
        if not self.passed_gate and drone_pos[0] >= self.gate_x:
            self.passed_gate = True 
            centering_accuracy = 1.0 - abs(drone_pos[1])
            gate_reward = 800.0 + (centering_accuracy * 200.0)
            print(f"Gate passed; phase 2 entered; reward of {gate_reward:.2f}")

        # Collision penalties
        wall_penalty = 0.0
        try: 
            client_id = self.env._client
        except Exception: 
            client_id = getattr(self, '_client', None)
        
        if client_id is not None and len(self.env.drones) > 0:
            for wall_segment_id in self.door_object_ids:
                contacts = p.getContactPoints(
                    bodyA=self.env.drones[0].Id, 
                    bodyB=wall_segment_id, 
                    physicsClientId=client_id
                    )
                if len(contacts) > 0:
                    wall_penalty = -25.0 
                    print("Drone smacked into something")
                    
                    pos, orn = p.getBasePositionAndOrientation(
                        self.env.drones[0].Id, 
                        physicsClientId=client_id
                        )
                    new_pos = [pos[0] - 0.2, pos[1], pos[2]]
                    p.resetBasePositionAndOrientation(
                        self.env.drones[0].Id, 
                        new_pos, 
                        orn, 
                        physicsClientId=client_id)
                    break

        # Goal Reached Reward
        completion_reward = 0.0
        dist_to_final = np.linalg.norm(self.target_pos - drone_pos)

        if dist_to_final < 0.5:
            terminated = True
            completion_reward = 1000.0
            print("DRONE COMPLETED COURSE SUCCESSFULLY (thank goodness!)")

        # Distance Rewards
        if not self.passed_gate:
            active_target_coords = np.array([1.5, 0.0, 1.5], dtype=np.float32)
        else:
            active_target_coords = self.target_pos
            
        dist_to_active_target = np.linalg.norm(active_target_coords - drone_pos)
        progress_reward = (self.prev_dist_to_target - dist_to_active_target) * 200.0
        self.prev_dist_to_target = dist_to_active_target

        # Ongoing time pentalty
        survival_penalty = -0.1 
        if self.current_step_count >= self.max_episode_steps:
            truncated = True

        if hasattr(self, 'terminated'): 
            self.terminated = terminated

        if hasattr(self, 'truncated'): 
            self.truncated = truncated

        if hasattr(self, 'env') and hasattr(self.env, 'terminated'):
            self.env.terminated = terminated

        if hasattr(self, 'env') and hasattr(self.env, 'truncated'):
            self.env.truncated = truncated

        total_reward = progress_reward + gate_reward + completion_reward + ceiling_penalty + survival_penalty + wall_penalty
        return obs_flat, total_reward, terminated, truncated, {}


def run_evaluation(name, agent_type, agent, discrete_wrapper=False, episodes=3):
    print(f"[Evaluating strategy {name}]")

    # DQN needs its discrete wrapper
    if name == "DQN" or discrete_wrapper:
        raw_env = QuadXObstacleCourseEnv(render_mode="human")
        env = DiscretizeActionWrapper(raw_env)
    else:
        env = QuadXObstacleCourseEnv(render_mode="human")
        raw_env = env

    client_id = None

    client_id = None
    if hasattr(raw_env, 'client_id'):
        client_id = raw_env.client_id
    elif hasattr(raw_env, 'env') and hasattr(raw_env.env, 'aviary'):
        client_id = raw_env.env.aviary.client
    elif hasattr(raw_env, 'env') and hasattr(raw_env.env, '_client'):
        client_id = raw_env.env._client
    else:
        client_id = getattr(raw_env, '_client', getattr(getattr(raw_env, 'env', None), '_client', None))
        
    print("="*20)
    print("-"*20)
    print(f"[DEBUG ENGINE]: Connected to PyBullet Client ID: {client_id}")
    print("="*20)
    print("-"*20)

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward, steps = 0.0, 0
        text_id, log_id = None, None

        client_id = None
        if hasattr(raw_env, 'client_id'):
            client_id = raw_env.client_id
        elif hasattr(raw_env, 'env') and hasattr(raw_env.env, 'aviary'):
            client_id = raw_env.env.aviary.client
        elif hasattr(raw_env, 'env') and hasattr(raw_env.env, '_client'):
            client_id = raw_env.env._client
        else:
            client_id = getattr(raw_env, '_client', getattr(getattr(raw_env, 'env', None), '_client', None))
            
        print(f"[DEBUG ENGINE - Episode {ep+1}]: Active PyBullet Client ID: {client_id}")

        # Recording the episodes automatically because they go by so fast
        target_dir = os.path.abspath(f"vids/{name}")
        os.makedirs(target_dir, exist_ok=True)
        absolute_video_path = os.path.join(target_dir, f"video_{name}_ep{ep+1}.mp4")
        
        if client_id is not None:
            text_id = p.addUserDebugText(
                text=f"Algorithm: {name} (Ep {ep+1})",
                textPosition=[0.0, 0.0, 3.5], 
                textColorRGB=[1.0, 1.0, 1.0], 
                textSize=1.5, 
                physicsClientId=client_id
            )
            log_id = p.startStateLogging(
                loggingType=p.STATE_LOGGING_VIDEO_MP4, 
                fileName=absolute_video_path, 
                physicsClientId=client_id
            )
            
        while True:
            if client_id is not None:
                p.resetDebugVisualizerCamera(
                    cameraDistance=5.5, cameraYaw=-50, cameraPitch=-20,
                    cameraTargetPosition=[2.5, 0.0, 1.2], physicsClientId=client_id
                )
                
            if name == "Random": 
                action = env.action_space.sample()
            elif name == "Heuristic": 
                action, _ = agent.predict(obs)
            else: 
                clean_obs_eval = np.array(obs, dtype=np.float32).flatten()[:24]
                action, _ = agent.predict(clean_obs_eval, deterministic=True)
                
            if isinstance(action, np.ndarray): 
                action = action.flatten()
                
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if name == "Random": 
                time.sleep(0.02)
                
            if terminated or truncated: 
                break
                
        print(f" Ep {ep+1} | Steps: {steps} | Reward: {total_reward:.2f}")
        
        if client_id is not None:
            if log_id is not None:
                try: p.stopStateLogging(log_id, physicsClientId=client_id)
                except Exception: pass
            if text_id is not None:
                try: p.removeUserDebugItem(text_id, physicsClientId=client_id)
                except Exception: pass
                
        time.sleep(0.8) 
            
    env.close()

    try:
        p.disconnect()
        time.sleep(0.2)
    except Exception:
        pass


if __name__ == "__main__":
    dqn_path = "custom_dqn_weights.pth"
    ppo_path = "custom_ppo_weights.pth"
    ddpg_path = "custom_ddpg_weights.pth"
    
    init_env = QuadXObstacleCourseEnv(render_mode=None)
    state_dim = (24,)
    action_dim_cont = init_env.action_space.shape
    init_env.close()
    
    dqn_agent = DQNAgent(state_dim, 7)
    ppo_agent = PPOAgent(state_dim, action_dim_cont)
    ddpg_agent = DDPGAgent(state_dim, action_dim_cont)
    
    TRAIN_STEPS = 150000
    
    if not os.path.exists(dqn_path):
        print("Training DQN (150k steps)")
        train_env_disc = DiscretizeActionWrapper(QuadXObstacleCourseEnv(render_mode=None))
        obs, info = train_env_disc.reset()
        
        for step in range(TRAIN_STEPS):
            clean_obs = np.array(obs, dtype=np.float32).flatten()[:24]
            action, _ = dqn_agent.predict(clean_obs, deterministic=False)
            
            next_obs, reward, terminated, truncated, info = train_env_disc.step(action)
            done = terminated or truncated
            
            clean_next_obs = np.array(next_obs, dtype=np.float32).flatten()[:24]
            
            dqn_agent.store_transition(clean_obs, action, reward, clean_next_obs, float(done))
            dqn_agent.train_step(batch_size=128)
            if step % 1000 == 0: 
                dqn_agent.update_target()
                
            cycle_step = step % 40000
            if cycle_step < 20000:
                start_eps = 1.0
                dqn_agent.epsilon = max(0.05, start_eps - (cycle_step / 20000) * (start_eps - 0.05))
            else: 
                dqn_agent.epsilon = 0.05
                
            if done: 
                obs, info = train_env_disc.reset()
            else: 
                obs = next_obs
                
        train_env_disc.close()
        dqn_agent.save(dqn_path)
        print("DQN weights exported successfully.")

    if not os.path.exists(ppo_path):
        print("Training PPO (150k steps)")
        train_env_cont = QuadXObstacleCourseEnv(render_mode=None)
        obs, info = train_env_cont.reset()
        states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
        ep_states, ep_actions, ep_log_probs, ep_rewards, ep_next_states, ep_dones = [], [], [], [], [], []
        
        for step in range(TRAIN_STEPS):
            clean_obs = np.array(obs, dtype=np.float32).flatten()[:24]
            action, log_prob = ppo_agent.predict(clean_obs, deterministic=False)
            action = action.flatten()
            
            next_obs, reward, terminated, truncated, info = train_env_cont.step(action)
            done = terminated or truncated
            
            clean_next_obs = np.array(next_obs, dtype=np.float32).flatten()[:24]
            
            ep_states.append(clean_obs)
            ep_actions.append(action)
            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
            ep_next_states.append(clean_next_obs)
            ep_dones.append(float(done))
            obs = next_obs
            
            if done:
                multiplier = 4 if any(r > 100.0 for r in ep_rewards) else 1
                for _ in range(multiplier):
                    states.extend(ep_states)
                    actions.extend(ep_actions)
                    log_probs.extend(ep_log_probs)
                    rewards.extend(ep_rewards)
                    
                    next_states.extend(ep_next_states) 
                    
                    dones.extend(ep_dones)
                    
                ep_states, ep_actions, ep_log_probs, ep_rewards, ep_next_states, ep_dones = [], [], [], [], [], []
                obs, info = train_env_cont.reset()

                
            if len(states) >= 2048:
                ppo_agent.train_step(states, actions, log_probs, rewards, next_states, dones)
                states, actions, log_probs, rewards, next_states, dones = [], [], [], [], [], []
                
        train_env_cont.close()
        ppo_agent.save(ppo_path)
        print("PPO weights exported successfully.")

    if not os.path.exists(ddpg_path):
        print("Training DDPG model (150k steps)")
        train_env_cont = QuadXObstacleCourseEnv(render_mode=None)
        obs, info = train_env_cont.reset()
        ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
        
        for step in range(TRAIN_STEPS):
            clean_obs = np.array(obs, dtype=np.float32).flatten()[:24]
            action, _ = ddpg_agent.predict(clean_obs, deterministic=False)
            action = action.flatten()
            
            next_obs, reward, terminated, truncated, info = train_env_cont.step(action)
            done = terminated or truncated
            
            clean_next_obs = np.array(next_obs, dtype=np.float32).flatten()[:24]
            
            ep_states.append(clean_obs)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_next_states.append(clean_next_obs)
            ep_dones.append(float(done))
            obs = next_obs
            
            if done:
                multiplier = 4 if any(r > 100.0 for r in ep_rewards) else 1
                for _ in range(multiplier):
                    for i in range(len(ep_states)):
                        clean_s = np.array(ep_states[i], dtype=np.float32).flatten()[:24]
                        clean_s_next = np.array(ep_next_states[i], dtype=np.float32).flatten()[:24]
                        ddpg_agent.store_transition(clean_s, ep_actions[i], ep_rewards[i], clean_s_next, ep_dones[i])
                ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
                obs, info = train_env_cont.reset()
                
            if step > 1000:
                ddpg_agent.train_step(batch_size=128)
            if step % 5000 == 0:
                ddpg_agent.noise_scale = max(0.05, ddpg_agent.noise_scale * 0.95)
                
        train_env_cont.close()
        ddpg_agent.save(ddpg_path)
        print("DDPG weights exported successfully.")

    print("="*20)
    print("FINALLY STARTING THE ASSESSMENT.")
    heuristic_agent = HeuristicAgent()
    
    run_evaluation("Random", None, None)
    print("="*20)

    run_evaluation("Heuristic", None, heuristic_agent)
    print("="*20)
    
    if os.path.exists(ppo_path):
        ppo_eval = PPOAgent((24,), action_dim_cont)
        ppo_eval.load(ppo_path)
        run_evaluation("PPO", None, ppo_eval)
        print("="*20)
        
    if os.path.exists(ddpg_path):
        ddpg_eval = DDPGAgent((24,), action_dim_cont)
        ddpg_eval.load(ddpg_path)
        run_evaluation("DDPG", None, ddpg_eval)
        print("="*20)
        
    if os.path.exists(dqn_path):
        dqn_eval = DQNAgent((24,), 7)
        dqn_eval.load(dqn_path)
        run_evaluation("DQN", None, dqn_eval, discrete_wrapper=True)
        print("="*20)
        
    print("Finished benchmarking! (Finally!!)")
