import gym
import copy
import random
import time

# CONSTANTS
ITERATIONS = 10000
DISCOUNT_FACTOR = 0.7
EXPLORE_PROB = 0.3 # Eps greedy exploration
LEARNING_RATE = 0.4
# is max moves required? Game ends from bot in reasonable time...
MAX_MOVES = 10000

# Load environment for Fishing Derby
env = gym.make('FishingDerby-v0')
env.reset()
# initialize Q
init_q = [0] * env.action_space.n
rows = env.observation_space.shape[0]
cols = env.observation_space.shape[1]
Q = {}

# # Inspecting the environment
# # Discrete(18)
# print(env.action_space)
# # Box (array dimensional) Box(high=0, low=255, shape=(210, 160, 3), uint8)
# print(env.observation_space.shape)
lookup_colors = { (167, 26, 26): "R", (24, 26, 167): "B",
       (117, 128, 240): "P", (72, 160, 72): "G",
       (66, 72, 200): "P", (232, 232, 74): "Y",
       (0, 0, 0): "BLK", (45, 50, 184): "B", 
       (0, 0, 148): "B", (228, 111, 111): "P" }
       
def get_color(r, g, b):
    """ Get the string representation of a color """
    return lookup_colors[(r, g, b)]

def observation_to_colors(arr):
    """ 2d array of observation to 2d array of colors """
    ret = [[get_color(rgb[0], rgb[1], rgb[2]) for rgb in row] for row in arr]
    return ret

def detect_fishing_rod(obs):
    for cc in range(28, 76):
        for rr, row in enumerate(reversed(obs)):
            rr = len(obs) - 1 - rr
            is_fish = row[cc] == "Y" and (row[cc - 1] == "Y" or row[cc - 2] in ["Y", "BLK"] or row[cc + 1] in ["Y", "BLK"] or row[cc - 1] in ["Y", "BLK"])
            is_rod = row[cc] == "Y" and obs[rr - 1][cc] == "Y" and obs[rr - 2][cc] == "Y" and obs[rr - 3][cc] == "Y" and obs[rr - 4][cc] == "Y"
            if rr < 187 and is_rod and not is_fish:
                return rr, cc
    return -1, -1

def detect_fish(obs, rr, cc):
    return obs[rr][cc-1] == "Y" or obs[rr][cc+1] == "Y"

def detect_shark_or_bottom(obs, rr, cc):
    return "BLK" in [obs[rr][cc-1], obs[rr][cc+1], obs[rr-1][cc], obs[rr+1][cc], obs[rr-2][cc-1], obs[rr+2][cc+1]]

def compute_reward(obs, rr, cc, old_score, new_score):
    ret = 0
    if detect_fish(obs, rr, cc):
        ret += 100 # 100 reward for hooking a fish
    if detect_shark_or_bottom(obs, rr, cc):
        ret -= 50
    if rr >= 186: # dont go super low with no fish! it is not rewarding!
        ret -= 50
    return ret

def get_surrounding_at(level, obs, rr, cc):
    top = obs[rr-level][cc]
    bottom = obs[rr+level][cc]
    left = obs[rr][cc-level]
    right = obs[rr][cc+level]
    diag_right_up = obs[rr-level][cc+level]
    diag_left_up = obs[rr-level][cc-level]
    diag_right_down = obs[rr+level][cc+level]
    diag_left_down = obs[rr+level][cc-level]
    return f"{top} {bottom} {left} {right} {diag_right_up} {diag_left_up} {diag_right_down} {diag_left_down}"

def get_state(obs, rr, cc):
    """ visibility of 2x2 around the fishing rod """
    level_1 = get_surrounding_at(1, obs, rr, cc)
    level_2 = get_surrounding_at(2, obs, rr, cc)
    return f"{level_1} {level_2}"

def find_best_action(state, obs, rr, cc):
    # Check if the corresponding 2x2 environment is in Q
    if state in Q:
        return Q[state].index(max(Q[state]))
    # if state isn't in Q then add it with initial and pick a random move
    else:
        Q[state] = copy.copy(init_q)
        return env.action_space.sample()

for _ in range(ITERATIONS):
    # is the current iteration done?
    done = False
    # for max moves
    ii = 0
    # start the game
    observation, new_score, done, info = env.step(0)
    # starting position
    rod_rr, rod_cc = 81, 43
    while ii < MAX_MOVES and not done:
        ii += 1
        # surrounding env of the current rod location
        state = get_state(observation, rod_rr, rod_cc)
        # epsilon greedy
        # env.action_space returns all actions, sample picks a random action
        if random.random() < EXPLORE_PROB:
            action = env.action_space.sample()
        else:
            action = find_best_action(state, observation, rod_rr, rod_cc)
        # make the move
        # step returns observation: env.observation_space, reward: float, done: bool, info: dict
        observation, new_score, done, info = env.step(action)
        # world translated to colors
        obs_colors = observation_to_colors(observation)
        # new position for rod
        next_rod_rr, next_rod_cc = detect_fishing_rod(obs_colors)
        # sometimes rod detection fails! If it does skip this iteration
        if next_rod_rr == -1:
            continue
        # the surrounding env of next rod location
        state_next = get_state(observation, next_rod_rr, next_rod_cc)
        # the next action based on the surrounding env of the next rod position
        next_action = find_best_action(state_next, observation, next_rod_rr, next_rod_cc)
        
        if state not in Q:
            # sometimes with a random exploration the state isn't actually in Q yet
            # because find_best_action wasn't called
            Q[state] = copy.copy(init_q) 
        
        q_current = Q[state][action]
        q_next = Q[state_next][next_action]
        reward = compute_reward(obs_colors, next_rod_rr, next_rod_cc)
        new_q = q_current + LEARNING_RATE * (reward + (DISCOUNT_FACTOR * q_next) - q_current)
        Q[state][action] = new_q
        
        # DEBUG
        # if new_q:
        #     print(f"got a new q value! {new_q}")
        # if reward:
        #     print(f"got a reward! {reward}")
        
        env.render()

        # next iteration
        rod_rr = next_rod_rr
        rod_cc = next_rod_cc
        
    env.reset()
    print(f'iteration {_} Q is \n')
    print(Q)
env.close()

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