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

DQN_RESULT_FILE_PATH = '../DQN_RESULTS/run1/dqn_weights.h5f'

env = gym.make('FishingDerby-v0')
SHAPE = env.observation_space.shape
# only 6 relevant actions that we want to consider to lower action_space
# this eliminates the usage of diagonal move combinations
ACTIONS = 6

# CONSTANTS
ITERATIONS = 100000
DISCOUNT_FACTOR = .9
LEARNING_RATE = .005
EXPLORE_PROB = .4

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
    #policy = LinearAnnealedPolicy(EpsGreedyQPolicy(EXPLORE_PROB), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    policy = EpsGreedyQPolicy(EXPLORE_PROB)
    memory = SequentialMemory(limit=100000, window_length=3)
    # DQN source code
    # https://github.com/keras-rl/keras-rl/blob/216c3145f3dc4d17877be26ca2185ce7db462bad/rl/agents/dqn.py
    # An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
    dqn = DQNAgent(model=MODEL, nb_actions=ACTIONS, memory=memory, policy=policy, gamma=DISCOUNT_FACTOR)
    return dqn

dqn = generate_agent()
dqn.compile(Adam(lr=LEARNING_RATE))
dqn.fit(env, nb_steps=ITERATIONS, visualize=False, verbose=1)

dqn.save_weights(f"{DQN_RESULT_FILE_PATH}")
dqn.load_weights(f"{DQN_RESULT_FILE_PATH}")

scores = dqn.test(env, nb_episodes=1, visualize=True)
print(np.mean(scores.history['episode_reward']))