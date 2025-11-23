import torch
import gymnasium as gym
from model1 import ActorCritic

def run_trained_agent(model_path):
    env = gym.make("BipedalWalker-v3", render_mode="human")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    obs, info = env.reset()

    episode_reward = 0
    while True:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            mean, std, value = model(obs_tensor)
            action = mean.numpy().clip(-1, 1)

        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        if done or truncated:
            break

    env.close()
    print(f"Episode Reward: {episode_reward}")

if __name__ == "__main__":
    # Change the checkpoint name if needed
    run_trained_agent("best_model.pt")

    