import torch
import torch.nn as nn
from torch.distributions import Normal

class PPO:
    def __init__(self, model, lr=3e-4, gamma=0.99, lam=0.95, clip_range=0.2, K_epochs=10): #k-epochs=10 for train.py
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.K_epochs = K_epochs

    def _get_logprob_and_entropy(self, mean, std, actions):
        dist = Normal(mean, std)
        logprob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logprob, entropy

    def update(self, buffer):
        states = torch.tensor(buffer.states, dtype=torch.float32)
        actions = torch.tensor(buffer.actions, dtype=torch.float32)
        old_logprobs = torch.tensor(buffer.logprobs, dtype=torch.float32)

        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=self.gamma, lam=self.lam
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            mean, std, values = self.model(states)
            logprobs, entropy = self._get_logprob_and_entropy(mean, std, actions)

            # PPO ratio
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogate objectives
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages

            # PPO loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)

            # Entropy bonus
            entropy_loss = -entropy.mean() * 0.001

            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()