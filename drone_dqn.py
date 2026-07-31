# ==========================================
# Experiment 10:
# DQN for Autonomous Drone Delivery System
# Part 1 - Environment and DQN Model
# ==========================================

import random
import numpy as np

from collections import deque

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# ==========================================
# Environment Parameters
# ==========================================

GRID_SIZE = 5

START = (0,0)

DELIVERY_POINT = (4,4)

MAX_BATTERY = 25


# Actions

ACTIONS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT"
]


# ==========================================
# Drone Delivery Environment
# ==========================================

class DroneEnvironment:

    def __init__(self):

        self.reset()


    def reset(self):

        self.position = START

        self.battery = MAX_BATTERY

        return self.get_state()


    def get_state(self):

        x,y = self.position

        return np.array([
            x/GRID_SIZE,
            y/GRID_SIZE,
            self.battery/MAX_BATTERY
        ])


    def step(self,action):

        x,y = self.position


        # Movement

        if action == 0:       # UP
            x -= 1

        elif action == 1:     # DOWN
            x += 1

        elif action == 2:     # LEFT
            y -= 1

        elif action == 3:     # RIGHT
            y += 1



        # Boundary Check

        x = max(0,min(GRID_SIZE-1,x))

        y = max(0,min(GRID_SIZE-1,y))


        self.position = (x,y)


        # Battery Consumption

        self.battery -= 1


        reward = -1

        done = False



        # Delivery completed

        if self.position == DELIVERY_POINT:

            reward = 100

            done = True



        # Battery finished

        if self.battery <= 0:

            reward = -100

            done = True



        return (
            self.get_state(),
            reward,
            done
        )



# ==========================================
# DQN Agent
# ==========================================


class DQNAgent:


    def __init__(self,state_size,action_size):


        self.state_size = state_size

        self.action_size = action_size


        # Replay Memory

        self.memory = deque(maxlen=5000)


        # Learning parameters

        self.gamma = 0.95

        self.epsilon = 1.0

        self.epsilon_min = 0.01

        self.epsilon_decay = 0.995

        self.learning_rate = 0.001


        self.model = self.create_model()



    def create_model(self):


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
            loss="mse",
            optimizer=Adam(
                learning_rate=self.learning_rate
            )
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


        if random.random() <= self.epsilon:

            return random.randrange(
                self.action_size
            )


        prediction = self.model.predict(
            state,
            verbose=0
        )


        return np.argmax(prediction[0])



# ==========================================
# Initialize Environment
# ==========================================


env = DroneEnvironment()


state_size = 3

action_size = 4


agent = DQNAgent(
    state_size,
    action_size
)


print("Drone Delivery Environment Created")

print("State Size :",state_size)

print("Action Size :",action_size)

print("DQN Model Initialized Successfully")