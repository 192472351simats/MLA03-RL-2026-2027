import numpy as np

# Grid Size
ROWS = 5
COLS = 5

# Warehouse Layout
# 0 = Empty
# 1 = Obstacle
# 2 = Goal (Pickup Point)

warehouse = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 2]
])

start = (0, 0)
goal = (4, 4)

actions = ["UP", "DOWN", "LEFT", "RIGHT"]

def get_next_state(state, action):
    r, c = state

    if action == "UP":
        r = max(r - 1, 0)
    elif action == "DOWN":
        r = min(r + 1, ROWS - 1)
    elif action == "LEFT":
        c = max(c - 1, 0)
    elif action == "RIGHT":
        c = min(c + 1, COLS - 1)

    if warehouse[r][c] == 1:
        return state

    return (r, c)

def reward(state):
    if state == goal:
        return 100
    return -1

def print_grid(robot):
    for i in range(ROWS):
        for j in range(COLS):
            if (i, j) == robot:
                print("R", end=" ")
            elif warehouse[i][j] == 1:
                print("X", end=" ")
            elif warehouse[i][j] == 2:
                print("G", end=" ")
            else:
                print(".", end=" ")
        print()
    print()

state = start
total_reward = 0

print("Warehouse Robot Navigation\n")

while state != goal:

    print_grid(state)

    best_action = None
    best_distance = 999

    for action in actions:
        next_state = get_next_state(state, action)

        distance = abs(goal[0] - next_state[0]) + abs(goal[1] - next_state[1])

        if distance < best_distance:
            best_distance = distance
            best_action = action

    state = get_next_state(state, best_action)

    r = reward(state)

    total_reward += r

    print("Action :", best_action)
    print("Reward :", r)
    print("----------------------")

print_grid(goal)

print("Goal Reached Successfully!")
print("Total Reward :", total_reward)