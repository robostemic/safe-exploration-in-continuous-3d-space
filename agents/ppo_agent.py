import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        if isinstance(state_dim, tuple): 
            state_dim = state_dim[0]

        if isinstance(action_dim, tuple): 
            action_dim = action_dim[0]

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def forward(self, state):
        return self.actor(state), torch.exp(self.log_std), self.critic(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def predict(self, state, deterministic=False):
        is_single_vector = (np.ndim(state) == 1)
        state_t = torch.FloatTensor(state).to(self.device)
        
        if is_single_vector:
            state_t = state_t.unsqueeze(0)
            
        with torch.no_grad():
            mean, std, _ = self.ac(state_t)
            
        if deterministic:
            return (mean.cpu().numpy().flatten(), None) if is_single_vector else (mean.cpu().numpy(), None)
            
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1).item()
        
        clipped_action = np.clip(action.cpu().numpy(), -1.0, 1.0)
        return (clipped_action.flatten(), log_prob) if is_single_vector else (clipped_action, log_prob)

    def train_step(self, states, actions, old_log_probs, rewards, next_states, dones):
        s_t = torch.FloatTensor(np.array(states)).to(self.device)
        a_t = torch.FloatTensor(np.array(actions)).to(self.device)
        lp_t = torch.FloatTensor(old_log_probs).to(self.device)

        with torch.no_grad():
            _, _, vals = self.ac(s_t)
            _, _, next_vals = self.ac(torch.FloatTensor(np.array(next_states)).to(self.device))
            vals = vals.squeeze(-1).cpu().numpy()
            next_vals = next_vals.squeeze(-1).cpu().numpy()

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_vals[t] * next_non_terminal - vals[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + vals
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(10): 
            mean, std, values = self.ac(s_t)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(a_t).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - lp_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(-1), ret_t)
            entropy_loss = dist.entropy().sum(dim=-1).mean()
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.2)
            self.optimizer.step()

    def save(self, path): 
        torch.save(self.ac.state_dict(), path)

    def load(self, path): 
        self.ac.load_state_dict(torch.load(path, map_location=self.device))

        