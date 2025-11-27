import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mlp(input_dim, hidden_sizes, output_dim, activation=nn.ReLU, output_activation=None):
    layers = []
    prev_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)

class TD3Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes, act_dim)
    
    def forward(self, obs):
        return torch.tanh(self.net(obs))  # actions in [-1, 1]

class TD3Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden_sizes, 1)
        self.q2 = mlp(obs_dim + act_dim, hidden_sizes, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2

    def q1_only(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x)

class TD3Agent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        lr=3e-4,
        hidden_sizes=(256, 256),
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

        self.actor = TD3Actor(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.actor_target = TD3Actor(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_target = TD3Critic(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, obs, noise_scale=0.0):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t)[0].cpu().numpy()
        if noise_scale > 0.0:
            action = action + noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)

    def update(self, replay_buffer, batch_size=256):
        self.total_it += 1

        batch = replay_buffer.sample_batch(batch_size)
        obs = torch.tensor(batch["obs"], dtype=torch.float32).to(self.device)
        acts = torch.tensor(batch["acts"], dtype=torch.float32).to(self.device)
        rews = torch.tensor(batch["rews"], dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32).to(self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32).unsqueeze(-1).to(self.device)

        # Select action according to policy and add clipped noise
        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rews + (1 - done) * self.gamma * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(obs, acts)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # Actor loss
            actor_actions = self.actor(obs)
            actor_loss = -self.critic.q1_only(obs, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update targets
            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.mul_(1 - self.tau)
                    target_param.data.add_(self.tau * param.data)
