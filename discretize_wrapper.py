import gymnasium as gym
import numpy as np

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(7)
        
    def action(self, action):
        continuous_action = np.zeros(4, dtype=np.float32)
        
        if action == 0:   
            continuous_action[1] = 0.20
            continuous_action[3] = 0.15
        elif action == 1: 
            continuous_action[1] = -0.15
            continuous_action[3] = 0.10
        elif action == 2: 
            continuous_action[0] = -0.15
            continuous_action[3] = 0.05
        elif action == 3: 
            continuous_action[0] = 0.15
            continuous_action[3] = 0.05
        elif action == 4: 
            continuous_action[3] = 0.30
        elif action == 5: 
            continuous_action[3] = -0.20
        elif action == 6: 
            continuous_action[3] = 0.0
            
        return continuous_action
