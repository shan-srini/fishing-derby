import gym
import copy
import random
import time

# CONSTANTS
ITERATIONS = 1
DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = 0.2 # Eps greedy exploration
LEARNING_RATE = 0.01
# is max moves required? Game ends from bot in reasonable time...
MAX_MOVES = 1000

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

lookup_colors = { (167, 26, 26): "red", (24, 26, 167): "blue",
       (117, 128, 240): "purple", (72, 160, 72): "green",
       (66, 72, 200): "purple/blue", (232, 232, 74): "yellow",
       (0, 0, 0): "black", (45, 50, 184): "blue", 
       (0, 0, 148): "blue", (228, 111, 111): "pink" }

lookup_colors_shorthand = { (167, 26, 26): "R", (24, 26, 167): "B",
       (117, 128, 240): "P", (72, 160, 72): "G",
       (66, 72, 200): "P", (232, 232, 74): "Y",
       (0, 0, 0): "BLK", (45, 50, 184): "B", 
       (0, 0, 148): "B", (228, 111, 111): "P" }
       
       
def get_color(r, g, b):
    """ Get the string representation of a color """
    return lookup_colors_shorthand[(r, g, b)]

def observation_to_colors(arr):
    """ 2d array of observation to 2d array of colors """
    ret = [[get_color(rgb[0], rgb[1], rgb[2]) for rgb in row] for row in arr]
    return ret

def get_print_outs(obs):
    obs_colors = observation_to_colors(observation)
    for row in range(len(obs_colors)):
        print(row)
        for col in range(len(obs_colors[0])):
            print(col)
            print(obs_colors[row][col] + " ", end='')
        print("\n")

def detect_fishing_rod(obs):
    previous = "B"
    for cc in range(28, 80):
        for rr, row in enumerate(reversed(obs)):
            is_fish = obs[rr][cc] == "Y" and (obs[rr][cc - 1] == "Y" or obs[rr][cc - 2] == "Y" or obs[rr][cc + 1] == "Y" or obs[rr][cc - 1] == "Y")
            is_rod = obs[rr][cc] == "Y" and obs[rr][cc - 1] == "B" and obs[rr][cc + 1] == "B" and obs[rr][cc - 2] == "B" and obs[rr][cc + 2] == "B"
            if rr < 187 and is_rod and not is_fish:
                print("row: " + str(rr) + " col: " + str(cc))
            previous = obs[rr][cc]

for _ in range(ITERATIONS):
    done = False
    ii = 0
    once = False 
    while ii < MAX_MOVES and not done:
        ii += 1
        # epsilon greedy
        if False and random.random() < EXPLORE_PROB:
            action = env.action_space.sample()
        else:
            action = 5# env.action_space.sample() # find best action given state most likely max(Q[current_hook_x][current_hook_y])

        env.render()
        # env.action_space returns all actions, sample picks a random action
        # step returns observation: env.observation_space, reward: float, done: bool, info: dict
        observation, reward, done, info = env.step(action)
        obs_colors = observation_to_colors(observation)
        detect_fishing_rod(obs_colors)
        time.sleep(.5)
        """
        if once == False and ii == 100:
            get_print_outs(obs_colors)
            once = True
        """
        #l = detect_fishing_rod(obs_colors)
        #l.add(f"red: {rgb[0]} green: {rgb[1]} blue: {rgb[2]}")
        #print('step')
    env.reset()
env.close()


# def detect_fishing_rod(obs):
#     """ using a 2d array of rgb values, detects the end of the fishing rod
#     @return x: int, y: int of the end of fishing rod
#     """
#     # obs = obs[::-1]
#     # print(obs)
#     MIN_ROD_LENGTH = 2
#     def detect(x, y):
#         """ helper to detect the rod in one spot by checking the pixel is yellow and surrounding is water """
#         if obs[x][y] == 'yellow':
#             left_water = (x - 1 >= 0) and obs[x-1][y] == 'blue'
#             right_water = (x + 1 < len(obs)) and obs[x+1][y] == 'blue'
#             return True
#         return False
#     # iterate observation looking for fishing rod
#     found_rod = False
#     count_rod = 0
#     for rr, row in enumerate(obs):
#         for cc, val in enumerate(row):
#             if detect(rr, cc):
#                 found_rod = True
#                 count_rod += 1
#             elif found_rod:
#                 if count_rod >= MIN_ROD_LENGTH:
#                     return rr - 1, cc - 1
#             else:
#                 found_rod = False
#     # raise BaseException("Rod not found")

"""
{'red: 167 green: 26 blue: 26', = red
'red: 24 green: 26 blue: 167' = blue
'red: 117 green: 128 blue: 240' = purple, 
'red: 72 green: 160 blue: 72' = green
'red: 66 green: 72 blue: 200' = purple/blue, 
'red: 232 green: 232 blue: 74' = yellow, 
'red: 0 green: 0 blue: 0' = black, 
'red: 45 green: 50 blue: 184', = blue
'red: 0 green: 0 blue: 148', = blue
'red: 228 green: 111 blue: 111'} = pink
"""

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