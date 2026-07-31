import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# Advertisement Recommendation
# using ε-Greedy Bandit
# -----------------------------

np.random.seed(42)
random.seed(42)

# Number of advertisements (arms)
n_ads = 5

# Number of user visits
n_rounds = 1000

# Exploration probability
epsilon = 0.1

# True click probabilities (unknown to the agent)
true_ctr = [0.10, 0.25, 0.40, 0.15, 0.30]

# Estimated values for each ad
Q = np.zeros(n_ads)

# Number of times each ad is selected
N = np.zeros(n_ads)

# Rewards collected
total_reward = 0

# Reward history
reward_history = []

# Selected advertisements
selected_ads = []

print("Starting Advertisement Recommendation...\n")

for t in range(n_rounds):

    # ε-Greedy Action Selection
    if random.random() < epsilon:
        action = random.randint(0, n_ads - 1)
    else:
        action = np.argmax(Q)

    # Simulate user click
    reward = 1 if random.random() < true_ctr[action] else 0

    # Update statistics
    N[action] += 1

    # Incremental Mean Update
    Q[action] = Q[action] + (reward - Q[action]) / N[action]

    total_reward += reward

    reward_history.append(total_reward)

    selected_ads.append(action)

print("Training Completed Successfully!\n")

print("Estimated Click Rates")

for i in range(n_ads):
    print(f"Ad {i+1}: {Q[i]:.3f}")

print("\nTimes Each Advertisement Selected")

for i in range(n_ads):
    print(f"Ad {i+1}: {int(N[i])}")

print("\nTotal Clicks:", total_reward)

print("\nBest Advertisement:", np.argmax(Q)+1)

# Plot cumulative reward
plt.figure(figsize=(8,5))
plt.plot(reward_history)
plt.title("Cumulative Reward")
plt.xlabel("Rounds")
plt.ylabel("Total Clicks")
plt.grid(True)
plt.show()

# Plot advertisement selections
plt.figure(figsize=(7,5))
plt.bar(range(1, n_ads+1), N)
plt.title("Advertisement Selection Count")
plt.xlabel("Advertisement")
plt.ylabel("Times Selected")
plt.show()