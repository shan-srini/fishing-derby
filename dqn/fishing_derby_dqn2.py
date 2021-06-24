"""
https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb
apt-get install -y wvfb ffmpeg ... might need brew installs

pip install 'imageio==2.4.0'
pip install pyvirtualdisplay
pip install tf-agents
"""
# IMPORTS
from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# Hyperparameters
TOTAL_ITERATIONS = 10000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-2
log_interval = 200

num_eval_episodes = 10
eval_interval = 1000

env = 