# ==================================================
# Experiment 14:
# Actor-Critic (A2C) for Smart Elevator Scheduling
# ==================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ==========================================
# Elevator Environment
# ==========================================

class ElevatorEnvironment:


    def __init__(self):

        self.floors = 5

        self.reset()



    def reset(self):

        self.elevator_floor = 0

        self.passenger_floor = np.random.randint(
            0,
            self.floors
        )

        self.destination = np.random.randint(
            0,
            self.floors
        )

        return self.state()



    def state(self):

        return np.array([
            self.elevator_floor / 5,
            self.passenger_floor / 5,
            self.destination / 5
        ])



    def step(self, action):

        # Actions:
        # 0 Down
        # 1 Up
        # 2 Stop/Open Door


        if action == 0:

            self.elevator_floor -= 1


        elif action == 1:

            self.elevator_floor += 1


        self.elevator_floor = max(
            0,
            min(4,self.elevator_floor)
        )


        reward = -1


        done = False


        # Passenger served

        if (
            action == 2 and
            self.elevator_floor == self.destination
        ):

            reward = 100

            done = True



        return (
            self.state(),
            reward,
            done
        )



# ==========================================
# Actor-Critic Network
# ==========================================


class ActorCritic(Model):


    def __init__(self,actions):

        super().__init__()


        self.common = Dense(
            64,
            activation="relu"
        )


        # Actor
        self.actor = Dense(
            actions,
            activation="softmax"
        )


        # Critic
        self.critic = Dense(
            1,
            activation="linear"
        )



    def call(self,state):

        x = self.common(state)

        return (
            self.actor(x),
            self.critic(x)
        )



# ==========================================
# A2C Training
# ==========================================


env = ElevatorEnvironment()


actions = 3


model = ActorCritic(actions)


optimizer = Adam(
    learning_rate=0.001
)


episodes = 300


gamma = 0.95


reward_history = []



for episode in range(episodes):


    state = env.reset()

    state = state.reshape(1,-1)


    total_reward = 0


    done = False



    while not done:


        with tf.GradientTape() as tape:


            probabilities,value = model(
                state
            )


            action = np.random.choice(
                actions,
                p=probabilities.numpy()[0]
            )


            next_state,reward,done = env.step(
                action
            )


            next_state = next_state.reshape(
                1,-1
            )


            _,next_value = model(
                next_state
            )


            target = reward + (
                gamma *
                next_value *
                (1-int(done))
            )


            advantage = target - value


            actor_loss = (
                -tf.math.log(
                    probabilities[0][action]
                )
                *
                advantage
            )


            critic_loss = tf.square(
                advantage
            )


            loss = (
                actor_loss +
                critic_loss
            )



        gradients = tape.gradient(
            loss,
            model.trainable_variables
        )


        optimizer.apply_gradients(
            zip(
                gradients,
                model.trainable_variables
            )
        )


        state = next_state

        total_reward += reward



    reward_history.append(
        total_reward
    )



print(
    "A2C Training Completed Successfully!"
)



# ==========================================
# Testing
# ==========================================


state = env.reset()

print(
    "\nInitial Elevator State:"
)

print(state)


done = False

steps = 0


while not done and steps < 20:


    state_input = state.reshape(
        1,-1
    )


    probs,value = model(
        state_input
    )


    action = np.argmax(
        probs.numpy()
    )


    state,reward,done = env.step(
        action
    )


    steps += 1



print(
    "\nElevator Task Completed"
)

print(
    "Steps:",
    steps
)



# Graph

plt.plot(
    reward_history
)

plt.title(
    "Actor-Critic Elevator Learning"
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Reward"
)

plt.grid()

plt.show()