import gym
import copy
import random
import time

# CONSTANTS
ITERATIONS = 100
DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = 0.4 # Eps greedy exploration
LEARNING_RATE = 0.01
# is max moves required? Game ends from bot in reasonable time...
MAX_MOVES = 1000

# Load environment for Fishing Derby
env = gym.make('FishingDerby-v0')
env.reset()
# initialize Q
init_q = [0] * env.action_space.n
# init_q[8] = .5
rows = env.observation_space.shape[0]
cols = env.observation_space.shape[1]
# Q = [[copy.copy(init_q) for __ in range(cols)] for _ in range(rows)]
Q = {}

# # Inspecting the environment
# # Discrete(18)
# print(env.action_space)
# # Box (array dimensional) Box(high=0, low=255, shape=(210, 160, 3), uint8)
# print(env.observation_space.shape)
def get_print_outs(obs):
    obs_colors = observation_to_colors(observation)
    for row in range(len(obs_colors)):
        print(row)
        for col in range(len(obs_colors[0])):
            print(col)
            print(obs_colors[row][col] + " ", end='')
        print("\n")

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
    for cc in range(31, 76):
        for rr, row in enumerate(reversed(obs)):
            rr = len(obs) - 1 - rr
            is_fish = row[cc] == "Y" and (row[cc - 1] == "Y" or row[cc - 2] in ["Y", "BLK"] or row[cc + 1] in ["Y", "BLK"] or row[cc - 1] in ["Y", "BLK"])
            is_rod = row[cc] == "Y" and obs[rr - 1][cc] == "Y" and obs[rr - 2][cc] == "Y" and obs[rr - 3][cc] == "Y" and obs[rr - 4][cc] == "Y"
            if rr < 187 and is_rod and not is_fish:
                # print("row: " + str(rr) + " col: " + str(cc))
                return rr, cc
    return -1, -1

def detect_fish(obs, rr, cc):
    return obs[rr][cc-1] == "Y" or obs[rr][cc+1] == "Y"

def detect_shark(obs, rr, cc):
    return "BLK" in [obs[rr][cc-1], obs[rr][cc+1], obs[rr-1][cc], obs[rr+1][cc], obs[rr-2][cc-1], obs[rr+2][cc+1]]

def compute_reward(obs, rr, cc, old_score, new_score):
    # return 100 if fish_is_on_rod else rewards[rr][cc]
    ret = 0
    if new_score > old_score:
        ret += new_score - old_score * 50 # if action incremented the score that's fantastic
    if detect_fish(obs, rr, cc):
        ret += 100 # 100 reward for hooking a fish
    if detect_shark(obs, rr, cc):
        ret -= 50
    return ret

def get_state(obs, rr, cc):
    return f"{obs[rr - 1][cc]} {obs[rr+1][cc]} {obs[rr][cc-1]} {obs[rr][cc+1]} {obs[rr - 2][cc]} {obs[rr+2][cc]} {obs[rr][cc-2]} {obs[rr][cc+2]}"

def find_best_action(obs, rr, cc):
    state = get_state(obs, rr, cc)
    if state in Q:
        return Q[state].index(max(Q[state]))
    else:
        Q[state] = copy.copy(init_q)
        return env.action_space.sample()

for _ in range(ITERATIONS):
    done = False
    ii = 0
    observation, old_score, done, info = env.step(0) # noop
    # starting position
    rod_rr, rod_cc = 81, 43
    print(f"Iteration: {_}")
    while ii < MAX_MOVES and not done:
        ii += 1
        # epsilon greedy
        # env.action_space returns all actions, sample picks a random action
        if random.random() < EXPLORE_PROB:
            action = env.action_space.sample()
        else:
            action = find_best_action(observation, rod_rr, rod_cc)
        # make the move
        # step returns observation: env.observation_space, reward: float, done: bool, info: dict
        observation, new_score, done, info = env.step(action)
        # new world for rod
        obs_colors = observation_to_colors(observation)
        # new position for rod
        next_rod_rr, next_rod_cc = detect_fishing_rod(obs_colors)
        if next_rod_rr == -1:
            next_rod_rr = rod_rr
            next_rod_cc = rod_cc
        next_action = find_best_action(observation, next_rod_rr, next_rod_cc)
        
        state = get_state(observation, rod_rr, rod_cc)
        state_next = get_state(observation, next_rod_rr, next_rod_cc)
        try:
            q_current = Q[state][action]
        except:
            q_current = Q[state_next][find_best_action(observation, rod_rr, rod_cc)]
        q_next = Q[state_next][find_best_action(observation, next_rod_rr, next_rod_cc)]
        reward = compute_reward(obs_colors, next_rod_rr, next_rod_cc, old_score, new_score)
        new_q = q_current + LEARNING_RATE * (reward + (DISCOUNT_FACTOR * q_next) - q_current)
        Q[state][action] = new_q
        if new_q:
            print(f"got a new q value! {new_q}")
        if reward:
            print('got a reward!')
        
        env.render()
        print(get_print_outs(observation))

        time.sleep(100)
        # print("next")
        # print(Q)
        
        old_score = new_score
        rod_rr = next_rod_rr
        rod_cc = next_rod_cc

        #l = detect_fishing_rod(obs_colors)
        #l.add(f"red: {rgb[0]} green: {rgb[1]} blue: {rgb[2]}")
        #print('step')
    env.reset()
env.close()
print(Q)

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