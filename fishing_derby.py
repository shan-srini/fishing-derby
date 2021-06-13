import gym
import copy
import random

# CONSTANTS
ITERATIONS = 100000
DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = 0.2 # Eps greedy exploration
LEARNING_RATE = 0.01
# is max moves required? Game ends from bot in reasonable time...
MAX_MOVES = 100000

# Load environment for Fishing Derby
env = gym.make('FishingDerby-v0')
env.reset()
# initialize Q
init_q = [0] * env.action_space.n
rows = env.observation_space.shape[0]
cols = env.observation_space.shape[1]
Q = [[copy.copy(init_q) for __ in range(cols)] for _ in range(rows)]

# # Inspecting the environment
# # Discrete(18)
# print(env.action_space)
# # Box (array dimensional) Box(high=0, low=255, shape=(210, 160, 3), uint8)
# print(env.observation_space.shape)

for _ in range(ITERATIONS):
    done = False
    ii = 0 
    while ii < MAX_MOVES and not done:
        ii += 1
        # epsilon greedy
        if random.random() < EXPLORE_PROB:
            action = env.action_space.sample()
        else:
            action = env.action_space.sample() # find best action given state most likely max(Q[current_hook_x][current_hook_y])

        env.render()
        # env.action_space returns all actions, sample picks a random action
        # step returns observation: env.observation_space, reward: float, done: bool, info: dict
        observation, reward, done, info = env.step(action)
        
    env.reset()
env.close()

"""
    Questions:
        To what extent will the reward given to us (my_score - opp_score) actually be beneficial
            Do we need to do any adjustments to this???
            How deep the hook is, how far the closest fish is
        How do we extract a policy? What does it look like/mean?
            Maybe similar to thought questions on HW: a small surrounding environment of the fishing hook
    Notes:
        Upping EXPLORE_PROB may help? 
        Classifier from rgb to objects to find surrounding environment of hook:
            232 232 74 = (most likely) FISH ... needs more investigation...
"""

# got this from gym code https://github.com/openai/gym/blob/1d31c12437e8bd7f466139a479705819fff8c111/gym/envs/atari/atari_env.py#L79
# if self._obs_type == 'ram':
#             self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
#         elif self._obs_type == 'image':
#             self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)