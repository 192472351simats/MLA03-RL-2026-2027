import numpy as np
import random

# Grid Size
ROWS = 5
COLS = 5

# Rewards
GOAL_REWARD = 100
OBSTACLE_REWARD = -100
STEP_REWARD = -1

# Learning Parameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
EPISODES = 500

# Actions
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# Initialize Q-table
Q = np.zeros((ROWS, COLS, len(actions)))

# Goal Position
goal = (4, 4)

# Obstacles
obstacles = [(1, 1), (2, 2), (3, 1)]


def move(state, action):
    r, c = state

    if action == 0:      # UP
        r = max(r - 1, 0)
    elif action == 1:    # DOWN
        r = min(r + 1, ROWS - 1)
    elif action == 2:    # LEFT
        c = max(c - 1, 0)
    elif action == 3:    # RIGHT
        c = min(c + 1, COLS - 1)

    return (r, c)


def reward(state):
    if state == goal:
        return GOAL_REWARD

    if state in obstacles:
        return OBSTACLE_REWARD

    return STEP_REWARD


# Training
for episode in range(EPISODES):

    state = (0, 0)

    while True:

        # ε-greedy action selection
        if random.uniform(0, 1) < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state[0], state[1]])

        next_state = move(state, action)

        r = reward(next_state)

        old_value = Q[state[0], state[1], action]

        next_max = np.max(Q[next_state[0], next_state[1]])

        new_value = old_value + ALPHA * (
            r + GAMMA * next_max - old_value
        )

        Q[state[0], state[1], action] = new_value

        state = next_state

        if state == goal or state in obstacles:
            break

print("\nTraining Completed Successfully!\n")

print("Q Table:\n")
print(Q)

print("\nOptimal Path:\n")

state = (0, 0)

path = [state]

while state != goal:

    action = np.argmax(Q[state[0], state[1]])

    next_state = move(state, action)

    if next_state == state:
        break

    path.append(next_state)

    state = next_state

    if state in obstacles:
        print("Robot hit an obstacle!")
        break

print(path)

if path[-1] == goal:
    print("\nGoal Reached Successfully!")