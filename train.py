import gymnasium as gym
import torch
import numpy as np
from model1 import ActorCritic
from ppo import PPO
from utils.memory import RolloutBuffer

best_reward = -9999

def train():
    env = gym.make("BipedalWalker-v3")
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim)
    ppo = PPO(model)

    buffer = RolloutBuffer()

    max_episodes = 10000

    rollout_steps = 2048
    save_interval = 50

    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0

        for step in range(rollout_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Actor produces mean, std, and critic value
            mean, std, value = model(obs_tensor)

            # Sample action from policy distribution
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

            action_clipped = action.numpy().clip(-1, 1)

            next_obs, reward, done, truncated, info = env.step(action_clipped)

            # Store transition
            buffer.states.append(obs)
            buffer.actions.append(action.numpy())
            buffer.logprobs.append(dist.log_prob(action).sum().item())
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.values.append(value.item())

            obs = next_obs
            episode_reward += reward

            if done or truncated:
                obs, info = env.reset()

        # PPO update
        ppo.update(buffer)
        buffer.clear()

        #print(f"Episode {episode} | Reward: {episode_reward:.2f}")

        with open("reward_log.txt", "a") as f:
            f.write(f"{episode},{episode_reward}\n")

        global best_reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(model.state_dict(), "best_model.pt")
            print(f"BEST MODEL UPDATED at episode {episode} with reward {episode_reward:.2f}")

        # Save model
        if episode % save_interval == 0:
            torch.save(model.state_dict(), f"ppo_bipedal_{episode}.pt")
            print(f"Model saved at episode {episode}.")

    env.close()

if __name__ == "__main__":
    train()