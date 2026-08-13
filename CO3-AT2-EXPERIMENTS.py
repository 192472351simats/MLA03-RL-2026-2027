import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)

# Policy Network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.001)

rewards = []

for episode in range(100):

    # Initial vehicles in 4 lanes
    vehicles = np.random.randint(1, 10, 4).astype(float)

    states = []
    actions = []
    episode_rewards = []

    for step in range(30):

        state = vehicles.copy()

        # Select action
        probabilities = model(
            state.reshape(1, -1), training=False
        )[0].numpy()

        action = np.random.choice(2, p=probabilities)

        # New vehicles arrive
        vehicles += np.random.randint(0, 2, 4)

        # Green signal
        if action == 0:
            vehicles[0] = max(0, vehicles[0] - 3)
            vehicles[1] = max(0, vehicles[1] - 3)
        else:
            vehicles[2] = max(0, vehicles[2] - 3)
            vehicles[3] = max(0, vehicles[3] - 3)

        waiting = np.sum(vehicles)
        reward = -waiting

        states.append(state)
        actions.append(action)
        episode_rewards.append(reward)

    # Calculate returns
    returns = []
    G = 0

    for r in reversed(episode_rewards):
        G = r + 0.99 * G
        returns.insert(0, G)

    returns = np.array(returns, dtype=np.float32)

    # Normalize returns
    returns = (returns - returns.mean()) / (
        returns.std() + 1e-8
    )

    # Policy update
    with tf.GradientTape() as tape:

        loss = 0

        for s, a, G in zip(states, actions, returns):

            prob = model(
                np.array(s).reshape(1, -1)
            )[0, a]

            loss -= tf.math.log(prob + 1e-8) * G

    gradients = tape.gradient(
        loss, model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    rewards.append(np.mean(episode_rewards))

    if episode % 20 == 0:
        print(
            "Episode:", episode,
            "Average Waiting Reward:",
            round(rewards[-1], 2)
        )
