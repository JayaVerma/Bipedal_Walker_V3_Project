import torch
import gymnasium as gym
import numpy as np
from td3 import TD3Agent

def run_td3_agent(model_path="td3_actor_best.pt", episodes=1):
    env = gym.make("BipedalWalker-v3", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = TD3Agent(obs_dim, act_dim)
    agent.actor.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.actor.eval()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(obs, noise_scale=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        print(f"[TD3] Episode {ep+1} Reward: {ep_reward}")
    env.close()

if __name__ == "__main__":
    run_td3_agent()
