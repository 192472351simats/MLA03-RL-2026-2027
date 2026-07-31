# =====================================================
# Experiment 11:
# Smart Traffic Signal Control using Deep RL
# Part 1 - Environment + DQN
# =====================================================

import random
import numpy as np

from collections import deque

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



# =====================================================
# Traffic Environment
# =====================================================


class TrafficEnvironment:


    def __init__(self):

        # Vehicles waiting at:
        # North, South, East, West

        self.max_queue = 20

        self.reset()



    def reset(self):

        self.queues = np.random.randint(
            0,
            10,
            size=4
        )

        self.time = 0

        return self.get_state()



    def get_state(self):

        return np.array(
            self.queues / self.max_queue,
            dtype=float
        )



    def step(self, action):

        """
        Actions:

        0 -> North-South Green
        1 -> East-West Green

        """


        old_waiting = np.sum(self.queues)



        # Vehicle generation

        arrivals = np.random.randint(
            0,
            4,
            size=4
        )


        self.queues += arrivals



        # Traffic light operation

        if action == 0:

            # North-South vehicles pass

            self.queues[0] = max(
                0,
                self.queues[0]-5
            )

            self.queues[1] = max(
                0,
                self.queues[1]-5
            )


        else:

            # East-West vehicles pass

            self.queues[2] = max(
                0,
                self.queues[2]-5
            )

            self.queues[3] = max(
                0,
                self.queues[3]-5
            )



        new_waiting = np.sum(self.queues)



        # Reward:
        # Lower waiting time = higher reward


        reward = old_waiting - new_waiting



        self.time += 1


        done = False


        if self.time >= 100:

            done = True



        return (
            self.get_state(),
            reward,
            done
        )





# =====================================================
# DQN Network
# =====================================================


class DQNAgent:



    def __init__(
            self,
            state_size,
            action_size):


        self.state_size = state_size

        self.action_size = action_size



        self.memory = deque(
            maxlen=5000
        )


        self.gamma = 0.95


        self.epsilon = 1.0

        self.epsilon_min = 0.01

        self.epsilon_decay = 0.995


        self.learning_rate = 0.001



        self.model = self.build_model()



    def build_model(self):


        model = Sequential()



        model.add(
            Dense(
                32,
                input_dim=self.state_size,
                activation="relu"
            )
        )



        model.add(
            Dense(
                32,
                activation="relu"
            )
        )



        model.add(
            Dense(
                self.action_size,
                activation="linear"
            )
        )



        model.compile(

            optimizer=Adam(
                learning_rate=self.learning_rate
            ),

            loss="mse"

        )


        return model




    def remember(
            self,
            state,
            action,
            reward,
            next_state,
            done):


        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )




    def act(self,state):


        if random.random() < self.epsilon:

            return random.randint(
                0,
                self.action_size-1
            )


        q_values = self.model.predict(
            state,
            verbose=0
        )


        return np.argmax(q_values[0])




# =====================================================
# Initialize System
# =====================================================


env = TrafficEnvironment()


state_size = 4

action_size = 2



agent = DQNAgent(
    state_size,
    action_size
)



print(
    "Traffic Signal Environment Created"
)

print(
    "State Size:",
    state_size
)

print(
    "Actions:",
    action_size
)


print(
    "DQN Model Initialized Successfully"
)