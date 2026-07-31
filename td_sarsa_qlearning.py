import numpy as np
import random
import matplotlib.pyplot as plt

# Grid Size
ROWS = 5
COLS = 5

START = (0, 0)
GOAL = (4, 4)

OBSTACLES = [(1,1), (2,2), (3,1)]

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 500

# Q-Tables
Q_sarsa = np.zeros((ROWS, COLS, 4))
Q_qlearning = np.zeros((ROWS, COLS, 4))

# TD Value Function
V = np.zeros((ROWS, COLS))

reward_history = []


def move(state, action):

    r, c = state

    if action == 0:
        r = max(r-1, 0)

    elif action == 1:
        r = min(r+1, ROWS-1)

    elif action == 2:
        c = max(c-1, 0)

    elif action == 3:
        c = min(c+1, COLS-1)

    if (r, c) in OBSTACLES:
        return state

    return (r, c)


def reward(state):

    if state == GOAL:
        return 100

    return -1


# ==========================
# Training
# ==========================

for episode in range(EPISODES):

    state = START

    # ε-greedy initial action
    if random.random() < EPSILON:
        action = random.randint(0,3)
    else:
        action = np.argmax(Q_sarsa[state[0], state[1]])

    total_reward = 0

    while state != GOAL:

        next_state = move(state, action)

        r = reward(next_state)

        total_reward += r

        # TD(0)
        V[state] = V[state] + ALPHA * (
            r + GAMMA * V[next_state] - V[state]
        )

        # Next action (SARSA)
        if random.random() < EPSILON:
            next_action = random.randint(0,3)
        else:
            next_action = np.argmax(
                Q_sarsa[next_state[0], next_state[1]]
            )

        # SARSA Update
        Q_sarsa[state[0], state[1], action] += ALPHA * (
            r +
            GAMMA *
            Q_sarsa[next_state[0], next_state[1], next_action]
            -
            Q_sarsa[state[0], state[1], action]
        )

        # Q-Learning Update
        Q_qlearning[state[0], state[1], action] += ALPHA * (
            r +
            GAMMA *
            np.max(Q_qlearning[next_state[0], next_state[1]])
            -
            Q_qlearning[state[0], state[1], action]
        )

        state = next_state
        action = next_action

    reward_history.append(total_reward)

print("\nTraining Completed Successfully!\n")

print("TD Value Function\n")
print(np.round(V,2))

print("\nSARSA Q Table\n")
print(np.round(Q_sarsa,2))

print("\nQ-Learning Q Table\n")
print(np.round(Q_qlearning,2))

# ==========================
# Test Learned Policy
# ==========================

print("\nOptimal Path (Q-Learning)\n")

state = START

path = [state]

while state != GOAL:

    action = np.argmax(
        Q_qlearning[state[0], state[1]]
    )

    next_state = move(state, action)

    if next_state == state:
        break

    path.append(next_state)

    state = next_state

print(path)

# Reward Graph
plt.figure(figsize=(8,5))
plt.plot(reward_history)
plt.title("Learning Performance")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()