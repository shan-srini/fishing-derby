import gym
import copy
import random
import time
import numpy as np
from math import sqrt
import json # for logs

# CONSTANTS
ITERATIONS = 100
DISCOUNT_FACTOR = 0.8
EXPLORE_PROB = .1 # Eps greedy exploration
LEARNING_RATE = 0.1
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
IS_HOOKED = False
# set custom defaults for Q values
"""
.... .................  
"""

def find_best_action(state,rr, cc):   
    # Check if the corresponding 2x2 environment is in Q
    # if state isn't in Q then add it
    if state not in Q:
        Q[state] = copy.copy(init_q)
    return Q[state].index(max(Q[state]))

def get_rod_position():
    ram = env.unwrapped._get_ram()
    rod_rr, rod_cc = ram[32], ram[67]
    return rod_rr, rod_cc

def get_fish_locations():
    ram = env.unwrapped._get_ram()
    f1 = (ram[74], 216)
    f2 = (ram[73], 221)
    f3 = (ram[72], 231)
    f4 = (ram[71], 237)
    f5 = (ram[70], 244)
    f6 = (ram[69], 253)
    return [f1, f2, f3, f4, f5, f6]

def dist(x1, y1, x2, y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2)

def get_state():
    rr, cc = get_rod_position()
    # find index of closest fish
    fishes = get_fish_locations()
    distances = [dist(x1=rr, y1=cc, x2=fish[0], y2=fish[1]) for fish in fishes]
    # favor lower fish by doing things like distances[5] -= .5 or something
    # pick the closest fish
    closest_fish_ix = np.argmin(distances)
    # store state with which direction has closest fish
    # testing... this freezes the game IF a fish is presumably caught
    if distances[closest_fish_ix] < 3: 
        print("fish_ix", closest_fish_ix)
        print(distances[closest_fish_ix])
        print(rr, cc)
        print(fishes[closest_fish_ix][0], fishes[closest_fish_ix][1])
        import time
        time.sleep(2)
    return

def make_moves_for_frames(count):
    # step returns observation: env.observation_space, reward: float, done: bool, info: dict
    total_reward = 0
    for _ in range(count):
        observation, score, done, info = env.step(action)
        total_reward += score
    return observation, total_reward, done, info

for _ in range(ITERATIONS):
    # is the current iteration done?
    done = False
    # for max moves
    ii = 0
    # start the game
    observation, score, done, info = env.step(0)
    # starting position
    rod_rr, rod_cc = get_rod_position()


    while ii < MAX_MOVES and not done:
        env.render()
        ii += 1
        # surrounding env of the current rod location
        state = 'hi' #get_state(observation, rod_rr, rod_cc)
        get_state()

        # epsilon greedy
        # env.action_space returns all actions, sample picks a random action
        if random.random() < EXPLORE_PROB:
            action = random.randrange(0, 6)
        else:
            action = find_best_action(state, rod_rr, rod_cc)
        # make the move
        observation, reward, done, info = make_moves_for_frames(1)
        
        # observation = observation[:187]
        
        # new position for rod
        next_rod_rr, next_rod_cc = get_rod_position()
        # the surrounding env of next rod location
        state_next ='hi'# get_state(observation, next_rod_rr, next_rod_cc)
        # the next action based on the surrounding env of the next rod position
        next_action = find_best_action(state_next, next_rod_rr, next_rod_cc)
        
        q_current = Q[state][action]
        q_next = Q[state_next][next_action]
        new_q = q_current + LEARNING_RATE * (reward + (DISCOUNT_FACTOR * q_next) - q_current)
        Q[state][action] = new_q
        
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