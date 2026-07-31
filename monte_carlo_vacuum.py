import numpy as np
import random
import matplotlib.pyplot as plt

# Grid size
ROWS = 5
COLS = 5

# Dirty cells (1 = dirty, 0 = clean)
grid = np.ones((ROWS, COLS), dtype=int)

# Start position
START = (0, 0)

# Actions
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# Parameters
EPISODES = 500
GAMMA = 0.9
EPSILON = 0.1

# Monte Carlo Tables
returns = {}
Q = {}

for r in range(ROWS):
    for c in range(COLS):
        for a in ACTIONS:
            Q[((r, c), a)] = 0
            returns[((r, c), a)] = []


def move(state, action):
    r, c = state

    if action == "UP":
        r = max(r - 1, 0)
    elif action == "DOWN":
        r = min(r + 1, ROWS - 1)
    elif action == "LEFT":
        c = max(c - 1, 0)
    elif action == "RIGHT":
        c = min(c + 1, COLS - 1)

    return (r, c)


reward_history = []

# ==========================
# Training
# ==========================
for ep in range(EPISODES):

    state = START
    room = np.ones((ROWS, COLS), dtype=int)

    episode = []
    total_reward = 0

    for step in range(100):

        # ε-greedy policy
        if random.random() < EPSILON:
            action = random.choice(ACTIONS)
        else:
            values = [Q[(state, a)] for a in ACTIONS]
            action = ACTIONS[np.argmax(values)]

        next_state = move(state, action)

        reward = -1  # energy cost

        # Clean dirty cell
        if room[next_state] == 1:
            reward += 10
            room[next_state] = 0

        episode.append((state, action, reward))

        total_reward += reward

        state = next_state

        # Stop if room is fully clean
        if room.sum() == 0:
            break

    reward_history.append(total_reward)

    # First-Visit Monte Carlo Update
    G = 0

    visited = set()

    for state, action, reward in reversed(episode):

        G = reward + GAMMA * G

        if (state, action) not in visited:

            returns[(state, action)].append(G)

            Q[(state, action)] = np.mean(returns[(state, action)])

            visited.add((state, action))

# ==========================
# Display Learned Policy
# ==========================

print("\nTraining Completed Successfully!\n")

policy = np.empty((ROWS, COLS), dtype=str)

symbols = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
}

for r in range(ROWS):
    for c in range(COLS):

        state = (r, c)

        values = [Q[(state, a)] for a in ACTIONS]

        best_action = ACTIONS[np.argmax(values)]

        policy[r][c] = symbols[best_action]

print("Optimal Cleaning Policy\n")

for row in policy:
    print(" ".join(row))

# ==========================
# Plot Rewards
# ==========================

plt.figure(figsize=(8,5))
plt.plot(reward_history)
plt.title("Monte Carlo Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()