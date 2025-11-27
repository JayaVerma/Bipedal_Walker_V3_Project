

---

# **BipedalWalker-v3: Deep Reinforcement Learning Project**

This repository contains implementations of three modern deep reinforcement learning algorithms — **PPO**, **SAC**, and **TD3** — trained and evaluated on the **BipedalWalker-v3** environment from Gymnasium.
The project was developed as part of my academic course work of **ESI 6684: Decision Making with Deep Reinforcement Learning (Fall 2025)**.

---

## 🚀 **Project Structure**

```
DEEP_REINFORCEMENT_LEARNING/
│── SAC/                     # SAC model checkpoints (if any)
│── TD3/                     # TD3 model checkpoints (if any)
│── PPO_best_model/          # PPO best model .pt file
│── report_result_section/   # Contains 3 videos of agent performance
│── utils/
│   ├── memory.py            # Memory buffer helper functions
│   ├── replay_buffer.py     # Replay buffer implementation
│
│── env_test.py              # Environment sanity-check script
│── model1.py                # Shared neural network model architectures
│
│── ppo.py                   # PPO algorithm implementation
│── sac.py                   # SAC algorithm implementation
│── td3.py                   # TD3 algorithm implementation
│
│── train.py                 # Training script for PPO
│── test_agent.py            # Test + visualize PPO agent
│
│── train_sac.py             # Training script for SAC
│── test_sac_agent.py        # Test + visualize SAC agent
│
│── train_td3.py             # Training script for TD3
│── test_td3_agent.py        # Test + visualize TD3 agent
│
│── plot_rewards.py          # Reward plotting utility
│── requirements.txt         # Project dependencies
└── .gitignore
```

---

## 🧠 **Implemented Algorithms**

The core algorithms are implemented **from scratch** (PyTorch-based) in the following files:

| Algorithm | Implementation File |
| --------- | ------------------- |
| **PPO**   | `ppo.py`            |
| **SAC**   | `sac.py`            |
| **TD3**   | `td3.py`            |

Each file contains:

* Policy and value network definitions
* Update rules
* Loss functions
* Interaction logic with the environment

---

## 🏃‍♂️ **Training Instructions**

Below are the exact commands to train and evaluate each agent.

---

### **1️⃣ Train & Test PPO Agent**

#### **Train PPO**

```
python train.py
```

This generates:

```
PPO_best_model/best_model.pt
```

#### **Test PPO**

```
python test_agent.py
```

This script:

* Loads `best_model.pt`
* Runs an evaluation episode
* Shows a visual rendering of the agent walking
* Prints per-episode reward

---

### **2️⃣ Train & Test SAC Agent**

#### **Train SAC**

```
python train_sac.py
```

This produces:

```
sac_policy_best.pt
```

#### **Test SAC**

```
python test_sac_agent.py
```

This script visualizes the SAC agent and prints reward results.

---

### **3️⃣ Train & Test TD3 Agent**

#### **Train TD3**

```
python train_td3.py
```

This generates:

```
td3_policy_best.pt
```

#### **Test TD3**

```
python test_td3_agent.py
```

This visualizes the TD3 agent and prints reward results.

---

## 🎥 **Agent Demonstration Videos**

In the folder:

```
report_result_section/
```

You will find 3 recorded videos:

* **ppo_run.mp4** — PPO walking behavior
* **sac_run.mp4** — SAC walking behavior
* **td3_run.mp4** — TD3 walking behavior

These are used directly in the project report.

---

## 📈 **Plotting Rewards**

You can generate reward curves using:

```
python plot_rewards.py
```

This will plot reward progression over training steps for the algorithm logs available.

---

## ⚙️ **Installing Dependencies**

Install all required packages:

```
pip install -r requirements.txt
```

If gymnasium requires a Box2D rebuild, run:

```
pip install gymnasium[box2d]
```

---

## 📦 **Reproducibility Notes**

* All training scripts use fixed random seeds
* Models are saved automatically during training
* Code is modular and based on PyTorch
* Environment: Gymnasium `BipedalWalker-v3`

---

## 📝 **Citation / Academic Integrity**

* PyTorch
* Gymnasium
* Algorithm papers (PPO 2017, SAC 2018, TD3 2018)

---

## 👩‍💻 **Author**

JAYA VERMA

---
