import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(128, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head (value)
        self.critic = nn.Linear(128, 1)

    def forward(self, obs):
        # Forward pass through shared layers
        x = self.base(obs)

        # Policy mean and log std
        mean = self.actor_mean(x)
        log_std = self.actor_log_std.exp()

        # State-value
        value = self.critic(x)
        return mean, log_std, value
    