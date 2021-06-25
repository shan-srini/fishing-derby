import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy

DQN_RESULT_FILE_PATH = './runs_dqn/run4/dqn_weights.h5f'

env = gym.make('FishingDerby-v0')
SHAPE = env.observation_space.shape
# only 6 relevant actions that we want to consider to lower action_space
# this eliminates the usage of diagonal move combinations
ACTIONS = env.action_space.n

""" Inspiration from: https://github.com/nicknochnack/KerasRL-OpenAI-Atari-SpaceInvadersv0/blob/main/Space%20Invaders%20Walkthrough.ipynb """
# CONSTANTS
LEARNING_ITERATIONS = 30000
TEST_ITERATIONS = 1
DISCOUNT_FACTOR = .8
LEARNING_RATE = .0001
EXPLORE_PROB = .3

LEARN = True

# issue with uint8?
tf.compat.v1.disable_eager_execution()

def generate_model():
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3, SHAPE[0], SHAPE[1], SHAPE[2])))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), strides=(2,2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(ACTIONS, activation='linear'))
    return model

MODEL = generate_model()

def generate_agent():
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=EXPLORE_PROB, value_test=EXPLORE_PROB, nb_steps=10000)
    memory = SequentialMemory(limit=10000, window_length=3)
    # DQN source code
    # https://github.com/keras-rl/keras-rl/blob/216c3145f3dc4d17877be26ca2185ce7db462bad/rl/agents/dqn.py
    # An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
    dqn = DQNAgent(model=MODEL, nb_actions=ACTIONS, memory=memory, policy=policy, gamma=DISCOUNT_FACTOR)
    return dqn

dqn = generate_agent()
dqn.compile(Adam(lr=LEARNING_RATE))
if LEARN:
    dqn.fit(env, nb_steps=LEARNING_ITERATIONS, visualize=False, verbose=1)
    dqn.save_weights(f"{DQN_RESULT_FILE_PATH}")
else:
    dqn.load_weights(f"{DQN_RESULT_FILE_PATH}")
    scores = dqn.test(env, nb_episodes=TEST_ITERATIONS, visualize=False)
    print(np.mean(scores.history['episode_reward']))