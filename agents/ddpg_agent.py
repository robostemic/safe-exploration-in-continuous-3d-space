import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, tuple): 
            state_dim = state_dim[0]

        if isinstance(action_dim, tuple): 
            action_dim = action_dim[0]

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state): 
        return self.network(state)

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, tuple): 
            state_dim = state_dim[0]

        if isinstance(action_dim, tuple): 
            action_dim = action_dim[0]

        self.s_layer = nn.Linear(state_dim, 128)
        self.a_layer = nn.Linear(action_dim, 128)
        self.network = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        s_out = nn.ReLU()(self.s_layer(state))
        a_out = nn.ReLU()(self.a_layer(action))
        combined = torch.cat([s_out, a_out], dim=-1)
        return self.network(combined)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005):
        self.action_dim = action_dim[0] if isinstance(action_dim, tuple) else action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = DDPGActor(state_dim, action_dim).to(self.device)
        self.actor_target = DDPGActor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.memory = deque(maxlen=50000)
        self.noise_scale = 0.2
        
    def predict(self, state, deterministic=False):
        is_single_vector = (np.ndim(state) == 1)
        state_t = torch.FloatTensor(state).to(self.device)

        if is_single_vector:
            state_t = state_t.unsqueeze(0)            

        self.actor.eval()

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()

        self.actor.train()
        
        if not deterministic:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action += noise
            
        action = np.clip(action, -1.0, 1.0)
        
        return (action.flatten(), None) if is_single_vector else (action, None)

        
    def store_transition(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))
        
    def train_step(self, batch_size=128):
        if len(self.memory) < batch_size: 
            return

        batch = random.sample(self.memory, batch_size)
        s, a, r, s_, done = zip(*batch)
        
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.FloatTensor(np.array(a)).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(np.array(s_)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_a = self.actor_target(s_)
            target_q = r + (self.gamma * self.critic_target(s_, next_a) * (1 - done))
            
        curr_q = self.critic(s, a)
        critic_loss = nn.MSELoss()(curr_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def save(self, path): 
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
        else:
            self.actor.load_state_dict(checkpoint)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
