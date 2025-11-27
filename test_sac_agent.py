import torch
import gymnasium as gym
import numpy as np
from sac import SACAgent

def run_sac_agent(model_path="/Users/jayaverma/Deep_Reinforcement_Project/SAC/sac_policy_best.pt", episodes=1):
    env = gym.make("BipedalWalker-v3", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim)
    agent.policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.policy.eval()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        print(f"[SAC] Episode {ep+1} Reward: {ep_reward}")
    env.close()

if __name__ == "__main__":
    run_sac_agent()
