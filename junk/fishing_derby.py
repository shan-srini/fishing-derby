import gym
import copy
import random
import time
import json # for printing logs

# CONSTANTS
ITERATIONS = 10000
DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = .1 # Eps greedy exploration
LEARNING_RATE = 0.1
# is max moves required? Game ends from bot in reasonable time...
MAX_MOVES = 100000

# Load environment for Fishing Derby
env = gym.make('FishingDerby-v0')
env.reset()
# env.seed(4100)
# initialize Q
init_q = [0] * 6
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
    return "Y" in [obs[rr][cc-1], obs[rr][cc+1], obs[rr+1][cc]]

def detect_shark_or_bottom(obs, rr, cc):
    depth = 2
    above = obs[rr-1] + obs[rr-2]
    cur_row = obs[rr]
    below = obs[rr+1] + obs[rr+2]
    check =  above[cc-depth:cc+depth+1] + cur_row[cc-depth:cc+depth+1] + below[cc-depth:cc+depth+1]
    # print(check)
    return "BLK" in check

def compute_reward(obs, rr, cc):
    ret = 0
    # if score_diff:
    #     ret += score_diff * 50
    #     print(score_diff)
    # if detect_fish(obs, rr, cc):
    #     ret += 100 # 100 reward for hooking a fish
    # if detect_shark_or_bottom(obs, rr, cc):
    #    ret -= 25
    # if rr >= 184: # dont go super low with no fish! it is not rewarding!
    #     ret -= 50
    return ret

def get_area(start_row, end_row, start_col, end_col, obs):
    ret = list()
    for ii in range(start_row, end_row):
        for jj in range(start_col, end_col):
            try: ret.extend(obs[ii][jj])
            except: pass
    return ret
    # return [arr[row][start_col:end_col] for row in range(start_row, end_row)]

def get_max_color(start_row, end_row, start_col, end_col, obs):
    ret = {
        "BLK": 0,
        "Y": 0,
        "B": 0
    }
    ret = 0
    for ii in range(start_row, end_row):
        for jj in range(start_col, end_col):
            # if obs[ii][jj] in ret:
            try: 
                color = obs[ii][jj]
                if color == (232, 232, 74):
                    ret += 1
            except: pass
    # find key with max value
    return ret // 5
    # return ret["Y"] // 3
    return max(ret, key=ret.get)

def get_state(obs, rr, cc):
    """ visibility of 2x2 around the fishing rod """
    # level of depth to go get the surrounding env of the end of fishing rod
    level = 20
    # surrounding = get_area(start_row=rr-level, end_row=rr+level+1, start_col=cc-level, end_col=cc+level+1, obs=obs)
    # # top left
    # top_left = get_max_color(start_row=rr-level, end_row=rr+(level//2), start_col=cc-level, end_col=cc+(level // 2), obs=obs)
    # # top right
    # top_right = get_max_color(start_row=rr-level, end_row=rr+(level//2), start_col=cc+(level//2), end_col=cc+level+1, obs=obs)
    # # bottom left
    # bottom_left = get_max_color(start_row=rr+(level//2), end_row=rr+level+1, start_col=cc-level, end_col=cc+(level//2), obs=obs)
    # # bottom right
    # bottom_right = get_max_color(start_row=rr+(level//2), end_row=rr+level+1, start_col=cc+(level//2), end_col=cc+level+1, obs=obs)
    # return f"top_left:{top_left} top_right:{top_right} bottom_left:{bottom_left} bottom_right:{bottom_right}"
    surrounding = get_area(start_row=rr-level, end_row=rr+level+1, start_col=cc-level, end_col=cc+level+1, obs=obs)
    # top left
    top = get_max_color(start_row=rr-level, end_row=rr, start_col=cc-level, end_col=cc+level+1, obs=obs)
    # top right
    right = get_max_color(start_row=rr-level, end_row=rr+level+1, start_col=cc, end_col=cc+level+1, obs=obs)
    # bottom left
    bottom = get_max_color(start_row=rr, end_row=rr+level+1, start_col=cc-level, end_col=cc+level+1, obs=obs)
    # bottom right
    left = get_max_color(start_row=rr-level, end_row=rr+level+1, start_col=cc-level, end_col=cc, obs=obs)
    ret =  f"top:{top} right:{right} bottom:{bottom} left:{left}"
    return ret

"""
A A A A A A A
A A A A A A A
A A A R A A A
A A A A A Y Y
B A A A A Y Y
A A A A A Y Y

top left = 3
top right = 3
bottom left = 3
bottom right = 3
"""

def find_best_action(state, obs, rr, cc):   
    # Check if the corresponding 2x2 environment is in Q
    # if state isn't in Q then add it
    if state not in Q:
        Q[state] = copy.copy(init_q)
    return Q[state].index(max(Q[state]))

for _ in range(ITERATIONS):
    # is the current iteration done?
    done = False
    # for max moves
    ii = 0
    # start the game
    observation, score, done, info = env.step(0)
    obs_colors = observation_to_colors(observation)
    # starting position
    rod_rr, rod_cc = 81, 43
    while ii < MAX_MOVES and not done:
        ii += 1
        # surrounding env of the current rod location
        state = get_state(obs_colors, rod_rr, rod_cc)
        # epsilon greedy
        # env.action_space returns all actions, sample picks a random action
        if False and random.random() < EXPLORE_PROB:
            action = random.randrange(0, 6)
        else:
            action = find_best_action(state, observation, rod_rr, rod_cc)
        # make the move
        # step returns observation: env.observation_space, reward: float, done: bool, info: dict
        reward = 0
        observation, score, done, info = env.step(action)
        reward += score
        observation, score, done, info = env.step(action)
        reward += score
        observation, score, done, info = env.step(action)
        reward += score
        observation, score, done, info = env.step(action)
        reward += score
        observation, score, done, info = env.step(action)
        reward += score
        observation = observation[:187]
        # world translated to colors
        # obs_colors = observation_to_colors(observation)
        # new position for rod
        next_rod_rr, next_rod_cc = detect_fishing_rod(obs_colors)
        # sometimes rod detection fails! If it does skip this iteration
        if next_rod_rr == -1:
            continue
        # the surrounding env of next rod location
        state_next = get_state(obs_colors, next_rod_rr, next_rod_cc)
        # the next action based on the surrounding env of the next rod position
        next_action = find_best_action(state_next, observation, next_rod_rr, next_rod_cc)
        
        if state not in Q:
            # sometimes with a random exploration the state isn't actually in Q yet
            # because find_best_action wasn't called
            Q[state] = copy.copy(init_q) 
        
        q_current = Q[state][action]
        q_next = Q[state_next][next_action]
        # reward = compute_reward(obs_colors, next_rod_rr, next_rod_cc)
        new_q = q_current + LEARNING_RATE * (reward + (DISCOUNT_FACTOR * q_next) - q_current)
        Q[state][action] = new_q
        
        env.render()
        
        # DEBUG
        # if new_q:
        #     print(f"got a new q value! {new_q}")
        # if reward:
        #     print(f"got a reward! {reward}")

        # next iteration
        rod_rr = next_rod_rr
        rod_cc = next_rod_cc
        
    env.reset()
    # log output
    with open('./Q_log', 'a') as f:
        f.write(f"iteration {_}")
        f.write("\n\n\n")
        f.write(json.dumps(Q, indent=2))
        f.write("\n\n\n")
    print(f"state space at length {len(Q)} Iteration is: {_}")
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