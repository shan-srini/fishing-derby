import gym
import copy
import random
import time
import numpy as np
import math
import json # for logs

# CONSTANTS
ITERATIONS = 10000
DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = .2 # Eps greedy exploration
LEARNING_RATE = 0.2
# is max moves required? Game ends from bot in reasonable time...
MAX_MOVES = 100000

# CODE CONSTANTS
UP = 'up'
RIGHT = 'right'
DOWN = 'down'
LEFT = 'left'
HOOKED = 'hooked'

# Global
# Load environment for Fishing Derby
env = gym.make('FishingDerby-v0')
env.reset()
# not sure which one to use
random.seed(4100); env.seed(4100)
# initialize Q
init_q = [0] * 6
Q = {
    UP: copy.copy(init_q),
    RIGHT: copy.copy(init_q),
    DOWN: copy.copy(init_q),
    LEFT: copy.copy(init_q),
    HOOKED: copy.copy(init_q)
}
# IS_HOOKED = False
# set custom defaults for Q values
Q[UP][2] = .1
Q[RIGHT][3] = .1
Q[LEFT][4] = .1
Q[DOWN][5] = .1
Q[HOOKED][0] = .5
"""
.... .................  
"""

def find_best_action(state):
    return np.argmax(Q[state])
    # Check if the corresponding 2x2 environment is in Q
    # if state isn't in Q then add it
    # if state not in Q:
    #     Q[state] = copy.copy(init_q)
    # return Q[state].index(max(Q[state]))

def get_rod_position():
    ram = env.unwrapped._get_ram()
    rod_rr, rod_cc = int(ram[32]), int(ram[67])
    return rod_rr, rod_cc

def get_fish_locations():
    ram = env.unwrapped._get_ram()
    f1 = str((int(ram[74]), 216))
    f2 = str((int(ram[73]), 221))
    f3 = str((int(ram[72]), 231))
    f4 = str((int(ram[71]), 237))
    f5 = str((int(ram[70]), 244))
    f6 = str((int(ram[69]), 253))
    shark = str((int(ram[75])))
    return [f1, f2, f3, f4, f5, f6, shark]

def dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def get_state():
    rr, cc = get_rod_position()
    # find index of closest fish
    fishes = ''.join(get_fish_locations())
    ret = f"{rr} {cc} {fishes}"
    if ret not in Q:
        Q[ret] = copy.copy(init_q)
    return ret
    distances = [dist(x1=rr, y1=cc, x2=fish[0], y2=fish[1]) for fish in fishes]
    # favor lower fish by doing things like distances[5] -= .5 or something
    # pick the closest fish
    closest_fish_ix = np.argmin(distances)
    # store state with which direction has closest fish
    # testing... this freezes the game IF a fish is presumably caught
    # if distances[closest_fish_ix] < 3: 
        # print("fish_ix", closest_fish_ix)
        # print(distances[closest_fish_ix])
        # print(rr, cc)
        # print(fishes[closest_fish_ix][0], fishes[closest_fish_ix][1])
        # vert_dif = rr - int(fishes[closest_fish_ix][0])
        # horz_dif = cc - fishes[closest_fish_ix][1]
        # print("vert ", str(vert_dif), " horz ", str(horz_dif))
        # import time
        # time.sleep(5)


    vert_dif = fishes[closest_fish_ix][0] - rr
    horz_dif = fishes[closest_fish_ix][1] - cc
    if vert_dif == 0 and horz_dif == 0:
        # pass # hooked! maybe do string state of hooked with the shark x location?
        return HOOKED
    elif abs(vert_dif) < abs(horz_dif) or horz_dif == 0:
        # pass # fish is UP if vert_dif > 0
        if vert_dif > 0:
            return UP
        # pass # fish is DOWN if vert_dif < 0
        else:
            return DOWN
    else: # abs(vert_dif) > abs(horz_dif)
        # pass # fish is RIGHT if vert_dif > 0
        if vert_dif > 0:
            return RIGHT
        # pass # fish is LEFT if vert_dif < 0
        else:
            return LEFT

def make_moves_for_frames(count):
    # step returns observation: env.observation_space, reward: float, done: bool, info: dict
    total_reward = 0
    for _ in range(count):
        observation, score, done, info = env.step(action)
        total_reward += score
    return observation, total_reward, done, info

avg_score = 0
for _ in range(ITERATIONS):
    # is the current iteration done?
    done = False
    # for max moves
    ii = 0
    # start the game
    observation, score, done, info = env.step(0)
    # starting position
    rod_rr, rod_cc = get_rod_position()
    score = 0
    while ii < MAX_MOVES and not done:
        env.render()
        ii += 1
        # surrounding env of the current rod location
        state = get_state()
        # print(state)
        # import time
        # time.sleep(1)
        # epsilon greedy
        # env.action_space returns all actions, sample picks a random action
        if random.random() < EXPLORE_PROB:
            action = random.randrange(0, 6)
        else:
            action = find_best_action(state)
        # print(action)
        # make the move
        observation, reward, done, info = make_moves_for_frames(1)
        reward = max(0, reward)
        score += reward
        # new position for rod
        next_rod_rr, next_rod_cc = get_rod_position()
        # state of next location
        state_prime = get_state()
        # the next action based on the surrounding env of the next rod position
        action_prime = find_best_action(state_prime)
        
        q_current = Q[state][action]
        q_prime = Q[state_prime][action_prime]
        q_update = q_current + LEARNING_RATE * (reward + (DISCOUNT_FACTOR * q_prime) - q_current)
        Q[state][action] = q_update
        
        # next iteration
        rod_rr = next_rod_rr
        rod_cc = next_rod_cc
    env.reset()
    print(f"iteration {_} score:", score)
    # log output
env.close()
with open('./Q_log_large_state', 'a') as f:
    f.write("\n\n\n")
    f.write(json.dumps(Q, indent=2))
    f.write("\n\n\n")

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