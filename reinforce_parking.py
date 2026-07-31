# ==================================================
# Experiment 13:
# REINFORCE Algorithm for Autonomous Parking System
# ==================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



# ==========================================
# Autonomous Parking Environment
# ==========================================

class ParkingEnvironment:


    def __init__(self):

        self.parking_position = 5
        self.reset()



    def reset(self):

        # Vehicle starting position

        self.position = np.random.randint(
            0,
            10
        )

        return np.array(
            [self.position]
        )



    def step(self, action):

        # Actions:
        # 0 -> Move Left
        # 1 -> Move Right
        # 2 -> Park


        if action == 0:

            self.position -= 1


        elif action == 1:

            self.position += 1



        elif action == 2:

            distance = abs(
                self.position -
                self.parking_position
            )


            if distance == 0:

                return (
                    np.array([self.position]),
                    100,
                    True
                )

            else:

                return (
                    np.array([self.position]),
                    -distance,
                    True
                )



        self.position = max(
            0,
            min(9,self.position)
        )


        distance = abs(
            self.position -
            self.parking_position
        )


        reward = -distance


        return (
            np.array([self.position]),
            reward,
            False
        )



# ==========================================
# Policy Network
# ==========================================


class PolicyNetwork:


    def __init__(self):

        self.model = Sequential()


        self.model.add(
            Dense(
                32,
                activation="relu",
                input_dim=1
            )
        )


        self.model.add(
            Dense(
                32,
                activation="relu"
            )
        )


        self.model.add(
            Dense(
                3,
                activation="softmax"
            )
        )


        self.optimizer = Adam(
            learning_rate=0.001
        )



    def action_probability(self,state):

        state = state.reshape(1,-1)

        return self.model(state)



# ==========================================
# REINFORCE Training
# ==========================================


env = ParkingEnvironment()

policy = PolicyNetwork()


episodes = 400

gamma = 0.95


reward_history = []



for episode in range(episodes):


    state = env.reset()


    states = []

    actions = []

    rewards = []


    done = False



    while not done:


        probs = policy.action_probability(
            state
        )


        action = np.random.choice(
            3,
            p=probs.numpy()[0]
        )


        next_state,reward,done = env.step(
            action
        )


        states.append(state)

        actions.append(action)

        rewards.append(reward)


        state = next_state



    # Discounted Returns

    G = 0

    returns = []


    for r in reversed(rewards):

        G = r + gamma * G

        returns.insert(
            0,
            G
        )


    returns = np.array(
        returns
    )


    returns = (
        returns -
        np.mean(returns)
    ) / (
        np.std(returns)+1e-8
    )



    # Policy update

    with tf.GradientTape() as tape:


        loss = 0


        for s,a,G in zip(
            states,
            actions,
            returns
        ):


            probabilities = (
                policy.action_probability(s)
            )


            selected = probabilities[0][a]


            loss += (
                -tf.math.log(selected)
                * G
            )



    gradients = tape.gradient(
        loss,
        policy.model.trainable_variables
    )


    policy.optimizer.apply_gradients(
        zip(
            gradients,
            policy.model.trainable_variables
        )
    )


    reward_history.append(
        sum(rewards)
    )



print(
    "\nTraining Completed Successfully!"
)



# ==========================================
# Testing Parking Agent
# ==========================================


state = env.reset()

parking_path = [
    state[0]
]


done = False


while not done:


    probs = policy.action_probability(
        state
    )


    action = np.argmax(
        probs.numpy()
    )


    state,reward,done = env.step(
        action
    )


    parking_path.append(
        state[0]
    )



print("\nParking Path:")

print(parking_path)



if reward == 100:

    print(
        "Parking Successful!"
    )

else:

    print(
        "Parking Failed"
    )



# Learning Curve

plt.plot(
    reward_history
)

plt.title(
    "REINFORCE Parking Learning"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Reward"
)

plt.grid()

plt.show()