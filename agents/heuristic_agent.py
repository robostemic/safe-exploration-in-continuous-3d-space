import numpy as np

class HeuristicAgent:
    def predict(self, state, deterministic=True):
        dx = state[12]
        dy = state[13]
        dz = state[14]
        
        pitch = np.clip(dx * 0.4, -0.2, 0.2)
        roll = np.clip(dy * 0.4, -0.2, 0.2)
        thrust = np.clip(dz * 0.5, -0.3, 0.3)
        yaw = 0.0
        
        action = np.array([roll, pitch, yaw, thrust], dtype=np.float32)
        return action, None
