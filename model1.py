import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(256, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  
        self.critic = nn.Linear(256, 1)

        # Critic head (value)
        self.critic = nn.Linear(256, 1)

    def forward(self, obs):
        x = self.base(obs)
        mean = self.actor_mean(x)
        std = torch.exp(self.log_std)  
        value = self.critic(x)
        return mean, std, value
    