import matplotlib.pyplot as plt

def plot_rewards(log_file="/Users/jayaverma/Deep_Reinforcement_Project/PPO_best_model/reward_log copy.txt"):
    episodes = []
    rewards = []

    with open(log_file, "r") as f:
        for line in f:
            episode, reward = line.strip().split(",")
            episodes.append(int(episode))
            rewards.append(float(reward))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label="Episode Reward", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_rewards()