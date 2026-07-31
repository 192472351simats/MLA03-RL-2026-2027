import random
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ==========================
# Hyperparameters
# ==========================

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 32

MEMORY_SIZE = 2000

EPISODES = 300

# ==========================
# DQN Agent
# ==========================

class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size

        self.action_size = action_size

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.gamma = GAMMA

        self.epsilon = EPSILON

        self.epsilon_min = EPSILON_MIN

        self.epsilon_decay = EPSILON_DECAY

        self.learning_rate = LEARNING_RATE

        self.model = self.build_model()

    def build_model(self):

        model = Sequential()

        model.add(Dense(24,
                        input_dim=self.state_size,
                        activation="relu"))

        model.add(Dense(24,
                        activation="relu"))

        model.add(Dense(self.action_size,
                        activation="linear"))

        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.learning_rate)
        )

        return model

    def remember(self,
                 state,
                 action,
                 reward,
                 next_state,
                 done):

        self.memory.append(
            (state,
             action,
             reward,
             next_state,
             done)
        )

    def act(self, state):

        if np.random.rand() <= self.epsilon:

            return random.randrange(self.action_size)

        q_values = self.model.predict(
            state,
            verbose=0
        )

        return np.argmax(q_values[0])

    def replay(self, batch_size):

        minibatch = random.sample(
            self.memory,
            batch_size
        )

        for state, action, reward, next_state, done in minibatch:

            target = reward

            if not done:

                target = reward + self.gamma * np.amax(
                    self.model.predict(
                        next_state,
                        verbose=0
                    )[0]
                )

            target_f = self.model.predict(
                state,
                verbose=0
            )

            target_f[0][action] = target

            self.model.fit(
                state,
                target_f,
                epochs=1,
                verbose=0
            )

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

# ==========================
# Create Environment
# ==========================

env = gym.make(ENV_NAME)

state_size = env.observation_space.shape[0]

action_size = env.action_space.n

agent = DQNAgent(
    state_size,
    action_size
)

print("Environment :", ENV_NAME)
print("State Size :", state_size)
print("Action Size :", action_size)

print("\nDQN Agent Initialized Successfully\n")