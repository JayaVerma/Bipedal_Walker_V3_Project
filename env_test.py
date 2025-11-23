import gymnasium as gym

def main():
    env = gym.make("BipedalWalker-v3", render_mode="human")
    obs, info = env.reset()

    print("Environment loaded successfully!")
    print("Observation shape:", obs.shape)

    for step in range(200):
        action = env.action_space.sample()  # random actions
        obs, reward, done, truncated, info = env.step(action)

        if step % 20 == 0:
            print(f"Step {step}, Reward: {reward}")

        if done or truncated:
            print("Episode finished early.")
            break

    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()

    