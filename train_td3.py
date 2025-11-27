import gymnasium as gym
import torch
import numpy as np
from td3 import TD3Agent
from utils.replay_buffer import ReplayBuffer

def train_td3():
    env = gym.make("BipedalWalker-v3")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = TD3Agent(obs_dim, act_dim)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, size=int(1e6))

    max_episodes = 4000
    max_steps = 1600
    start_steps = 10000  # pure random actions at the beginning
    batch_size = 256
    exploration_noise = 0.1

    total_steps = 0
    best_reward = -1e9

    for episode in range(1, max_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, noise_scale=exploration_noise)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.store(obs, action, reward, next_obs, float(done))

            obs = next_obs
            episode_reward += reward
            total_steps += 1

            if replay_buffer.size >= batch_size:
                agent.update(replay_buffer, batch_size=batch_size)

            if done:
                break

        with open("td3_reward_log.txt", "a") as f:
            f.write(f"{episode},{episode_reward}\n")

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.actor.state_dict(), "td3_actor_best.pt")
            print(f"[TD3] BEST ACTOR UPDATED at ep {episode} | reward {episode_reward:.2f}")

        print(f"[TD3] Episode {episode} | Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    train_td3()
