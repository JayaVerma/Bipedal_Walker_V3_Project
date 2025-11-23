import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        rewards = self.rewards
        dones = self.dones
        values = self.values

        returns = []
        advantages = []

        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

            next_value = values[step]

        return (
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )