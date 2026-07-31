# ==================================================
# Experiment 12:
# Policy-Based RL for Industrial Robotic Arm
# REINFORCE Policy Gradient Algorithm
# ==================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import random


# ================================================
# Robotic Arm Environment
# ================================================

class RoboticArmEnv:


    def __init__(self):

        self.reset()



    def reset(self):

        # Arm starting position

        self.position = 0

        # Object location

        self.object_position = 4

        return np.array([self.position])



    def step(self, action):


        # Actions:
        # 0 -> Move Left
        # 1 -> Move Right
        # 2 -> Pick


        if action == 0:

            self.position -= 1


        elif action == 1:

            self.position += 1


        elif action == 2:


            if self.position == self.object_position:

                reward = 100

                done = True

                return (
                    np.array([self.position]),
                    reward,
                    done
                )


        # Boundary

        self.position = max(
            0,
            min(5,self.position)
        )


        distance = abs(
            self.position -
            self.object_position
        )


        reward = -distance


        done = False


        return (
            np.array([self.position]),
            reward,
            done
        )



# ================================================
# Policy Network
# ================================================


class PolicyNetwork:


    def __init__(self):


        self.model = Sequential()


        self.model.add(
            Dense(
                16,
                input_dim=1,
                activation="relu"
            )
        )


        self.model.add(
            Dense(
                16,
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



    def predict(self,state):

        state = state.reshape(1,-1)

        return self.model(state)



# ================================================
# REINFORCE Training
# ================================================


env = RoboticArmEnv()

policy = PolicyNetwork()


episodes = 300

gamma = 0.95


reward_history = []



for episode in range(episodes):


    state = env.reset()


    states = []

    actions = []

    rewards = []



    done = False



    while not done:


        probability = policy.predict(state)


        action = np.random.choice(
            3,
            p=probability.numpy()[0]
        )


        next_state, reward, done = env.step(action)



        states.append(state)

        actions.append(action)

        rewards.append(reward)


        state = next_state



    # Calculate discounted rewards

    discounted_rewards = []

    G = 0


    for reward in reversed(rewards):

        G = reward + gamma * G

        discounted_rewards.insert(
            0,
            G
        )


    discounted_rewards = np.array(
        discounted_rewards
    )


    discounted_rewards = (
        discounted_rewards -
        np.mean(discounted_rewards)
    ) / (
        np.std(discounted_rewards)+1e-8
    )



    # Policy update

    with tf.GradientTape() as tape:


        loss = 0


        for s,a,G in zip(
            states,
            actions,
            discounted_rewards
        ):


            probabilities = policy.predict(s)


            selected_probability = probabilities[0][a]


            loss += -tf.math.log(
                selected_probability
            ) * G



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



# ================================================
# Testing Robot Arm
# ================================================


state = env.reset()

path = [state[0]]

done = False


while not done:


    probabilities = policy.predict(state)


    action = np.argmax(
        probabilities.numpy()
    )


    state,reward,done = env.step(action)


    path.append(
        state[0]
    )


print("\nRobot Arm Movement:")

print(path)



if done:

    print(
        "\nObject Picked Successfully!"
    )



# Reward Graph

plt.plot(
    reward_history
)

plt.title(
    "Policy Gradient Learning"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Reward"
)

plt.grid()

plt.show()