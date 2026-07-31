import numpy as np

# Grid Size
ROWS = 5
COLS = 5

# Rewards
GOAL_REWARD = 100
STEP_COST = -1
DISCOUNT = 0.9

# Goal Position
goal = (4, 4)

# Obstacles
obstacles = [(1, 1), (2, 2), (3, 1)]

# Initialize Value Function
V = np.zeros((ROWS, COLS))

# Actions: Up, Down, Left, Right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) not in obstacles


# Bellman Value Iteration
iterations = 100

for _ in range(iterations):

    new_V = V.copy()

    for r in range(ROWS):
        for c in range(COLS):

            if (r, c) == goal:
                new_V[r][c] = GOAL_REWARD
                continue

            if (r, c) in obstacles:
                continue

            values = []

            for dr, dc in actions:

                nr = r + dr
                nc = c + dc

                if valid(nr, nc):
                    values.append(STEP_COST + DISCOUNT * V[nr][nc])

            if values:
                new_V[r][c] = max(values)

    V = new_V

print("Optimal Value Function\n")
print(np.round(V, 2))

# Find Optimal Path

state = (0, 0)

path = [state]

while state != goal:

    r, c = state

    best_state = state
    best_value = -9999

    for dr, dc in actions:

        nr = r + dr
        nc = c + dc

        if valid(nr, nc):

            if V[nr][nc] > best_value:
                best_value = V[nr][nc]
                best_state = (nr, nc)

    if best_state == state:
        break

    state = best_state
    path.append(state)

print("\nOptimal Path")
print(path)

print("\nTotal Steps :", len(path) - 1)

if path[-1] == goal:
    print("Delivery Completed Successfully!")