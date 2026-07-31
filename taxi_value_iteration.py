import numpy as np

# Grid size
ROWS = 5
COLS = 5

# Discount factor
GAMMA = 0.9

# Step cost
STEP_COST = -1

# Goal location (Passenger Destination)
GOAL = (4, 4)

# Obstacles
OBSTACLES = [(1, 1), (2, 2), (3, 1)]

# Value function
V = np.zeros((ROWS, COLS))

# Possible actions
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def valid(r, c):
    return (
        0 <= r < ROWS and
        0 <= c < COLS and
        (r, c) not in OBSTACLES
    )


def next_state(state, action):

    r, c = state

    if action == "UP":
        r -= 1

    elif action == "DOWN":
        r += 1

    elif action == "LEFT":
        c -= 1

    elif action == "RIGHT":
        c += 1

    if not valid(r, c):
        return state

    return (r, c)


# ----------------------------
# Value Iteration
# ----------------------------

iterations = 100

for _ in range(iterations):

    newV = V.copy()

    for r in range(ROWS):

        for c in range(COLS):

            if (r, c) == GOAL:
                newV[r][c] = 100
                continue

            if (r, c) in OBSTACLES:
                continue

            values = []

            for action in ACTIONS:

                ns = next_state((r, c), action)

                reward = STEP_COST

                values.append(
                    reward +
                    GAMMA *
                    V[ns[0]][ns[1]]
                )

            newV[r][c] = max(values)

    V = newV

print("Optimal Value Function\n")
print(np.round(V, 2))

# ----------------------------
# Extract Optimal Policy
# ----------------------------

policy = np.full((ROWS, COLS), " ")

symbols = {
    "UP": "↑",
    "DOWN": "↓",
    "LEFT": "←",
    "RIGHT": "→"
}

for r in range(ROWS):

    for c in range(COLS):

        if (r, c) == GOAL:
            policy[r][c] = "G"
            continue

        if (r, c) in OBSTACLES:
            policy[r][c] = "X"
            continue

        best_action = None
        best_value = -9999

        for action in ACTIONS:

            ns = next_state((r, c), action)

            if V[ns[0]][ns[1]] > best_value:

                best_value = V[ns[0]][ns[1]]

                best_action = action

        policy[r][c] = symbols[best_action]

print("\nOptimal Policy\n")

for row in policy:
    print(" ".join(row))

# ----------------------------
# Taxi Navigation
# ----------------------------

print("\nTaxi Route\n")

state = (0, 0)

route = [state]

while state != GOAL:

    r, c = state

    best_action = None
    best_value = -9999

    for action in ACTIONS:

        ns = next_state(state, action)

        if V[ns[0]][ns[1]] > best_value:

            best_value = V[ns[0]][ns[1]]

            best_action = action

    state = next_state(state, best_action)

    route.append(state)

print(route)

print("\nDestination Reached Successfully!")