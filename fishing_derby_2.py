import gym
import copy
import random
import time
import json

random.seed(4100)
ITERATIONS = 100

env = gym.make('FishingDerby-ram-v0')
env.reset()

"""

obs_space[32] is x pos of rod
obs_space[67] is y pos of rod

obs_space[33] is x pos of opponent rod
obs_space[68] is y pos of opponent rod

obs_space[69] is x pos of fish 6
obs_space[70] is x pos of fish 2
obs_space[71] is x pos of fish 3
obs_space[72] is x pos of fish 4
obs_space[73] is x pos of fish 5
obs_space[74] is x pos of fish 6
"""
last = []
p = set()
test = [69, 70, 71, 72, 73, 74]
for _ in range(ITERATIONS):
    a = env.step(0)
    env.render()
    a = a[0]

    print(a)
    print("------------------")
  
   

  
    time.sleep(3)
    
"""
    if _ == 1:
        for l in range(len(a)):
            if a[l] > last[l]:
                p.add(l)

    for i in range(len(p)):
        if a[i] < last[i] and p.__contains__(i):
            p.remove(i)
        #print(str(i) + " num: " + str(a[i]))
       
    last = a

print(p)
"""
print(p)
env.reset()
"""    
    print("---------")
    print("x is: " + str(a[33]) + " y is: " + str(a[68]))
    print(a)

    for nu in range(len(a)):
        print("index: " + str(nu) + " num: " + str(a[nu]))
"""