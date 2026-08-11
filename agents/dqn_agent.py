import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        if isinstance(state_dim, tuple):
            state_dim = state_dim[0]
            
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0

    def predict(self, state, deterministic=False):
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), None

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return torch.argmax(q_values, dim=1).item(), None

    def store_transition(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def train_step(self, batch_size=128):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        s, a, r, s_, done = zip(*batch)
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(np.array(s_)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        curr_q = self.q_net(s).gather(1, a)

        with torch.no_grad():
            next_q = self.target_net(s_).max(1)[0].unsqueeze(1)
            target_q = r + (self.gamma * next_q * (1 - done))

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path): 
        torch.save(self.q_net.state_dict(), path)

        
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
