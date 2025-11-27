import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2

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

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, hidden_sizes, 2 * act_dim)
        self.act_dim = act_dim

    def forward(self, obs):
        x = self.net(obs)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t  # already in [-1, 1]

        # log prob with tanh correction
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q_net = mlp(obs_dim + act_dim, hidden_sizes, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        q = self.q_net(x)
        return q

class SACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        hidden_sizes=(256, 256),
        automatic_entropy_tuning=True,
        target_entropy=None,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.act_dim = act_dim
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Policy
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes).to(self.device)
        # Critics
        self.q1 = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        # Target critics
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_sizes).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        if self.automatic_entropy_tuning:
            if target_entropy is None:
                # heuristic: -|A|
                target_entropy = -float(act_dim)
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

    @property
    def alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp().item()
        else:
            return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    def select_action(self, obs, evaluate=False):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, mean_action = self.policy.sample(obs_t)
                action = mean_action[0].cpu().numpy()
            else:
                action, _, _ = self.policy.sample(obs_t)
                action = action[0].cpu().numpy()
        return action

    def update(self, replay_buffer, batch_size=256):
        batch = replay_buffer.sample_batch(batch_size)
        obs = torch.tensor(batch["obs"], dtype=torch.float32).to(self.device)
        acts = torch.tensor(batch["acts"], dtype=torch.float32).to(self.device)
        rews = torch.tensor(batch["rews"], dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32).to(self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32).unsqueeze(-1).to(self.device)

        # Sample actions from policy for next state
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_action)
            q2_next = self.q2_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = rews + (1 - done) * self.gamma * q_next

        # Q1, Q2 losses
        q1_pred = self.q1(obs, acts)
        q2_pred = self.q2(obs, acts)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy loss
        action_pi, log_pi, _ = self.policy.sample(obs)
        q1_pi = self.q1(obs, action_pi)
        q2_pi = self.q2(obs, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * log_pi - q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Entropy temperature update
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update targets
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)

            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.mul_(1 - self.tau)
                target_param.data.add_(self.tau * param.data)
 