import gym
import copy
import random
import time
import numpy as np
import math
import sys
import json # for logs

# where to store results
STORE_Q_FILE_PATH ='./naive/Q'
LEARN = True

# CONSTANTS
ITERATIONS = 1
TEST_ITERATIONS = 1
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
Q[HOOKED][0] = .1

def find_best_action(state):
    return np.argmax(Q[state])

def get_rod_position():
    ram = env.unwrapped._get_ram()
    rod_rr, rod_cc = int(ram[32]), int(ram[67])
    return rod_rr, rod_cc

def get_fish_locations():
    ram = env.unwrapped._get_ram()
    f1 = (int(ram[74]), 216)
    f2 = (int(ram[73]), 221)
    f3 = (int(ram[72]), 231)
    f4 = (int(ram[71]), 237)
    f5 = (int(ram[70]), 244)
    f6 = (int(ram[69]), 253)
    return [f1, f2, f3, f4, f5, f6]

def dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def get_state():
    rr, cc = get_rod_position()
    # find index of closest fish
    fishes = get_fish_locations()
    distances = [dist(x1=rr, y1=cc, x2=fish[0], y2=fish[1]) for fish in fishes]
    # favor lower fish by doing things like distances[5] -= .5 or something
    distances[5] -= .1
    distances[4] -= .01
    distances[3] -= .01
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

def make_moves_for_frames(count, action):
    # step returns observation: env.observation_space, reward: float, done: bool, info: dict
    total_reward = 0
    for _ in range(count):
        observation, score, done, info = env.step(action)
        total_reward += score
    return observation, total_reward, done, info

def train():
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
            # epsilon greedy
            if random.random() < EXPLORE_PROB:
                action = random.randrange(0, 6)
            else:
                action = find_best_action(state)
            # print(action)
            # make the move
            observation, reward, done, info = make_moves_for_frames(1, action)
            reward = max(0, reward)
            score+=reward
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
        # print(f"iteration{_} score:", score)
    env.close()
    # save output
    with open(f"{STORE_Q_FILE_PATH}", 'w') as f:
        f.write(json.dumps(Q, indent=2))

def test():
    global Q
    total_scores = []
    try:
        with open(f"{STORE_Q_FILE_PATH}", 'r') as f:
            Q = json.loads(f.read())
    except:
        pass
        # raise FileNotFoundError(f"Need Q in {FILE_PATH}")
    for ii in range(TEST_ITERATIONS):
        done = False
        iteration_score = 0
        while not done:
            state = get_state()
            action = find_best_action(state)
            observation, reward, done, info = make_moves_for_frames(1, action)
            iteration_score += reward
    total_scores.append(iteration_score)
    print(np.mean(total_scores))
    env.close()
    return



if __name__=='__main__':
    if LEARN:
        train()
    else:
        test()