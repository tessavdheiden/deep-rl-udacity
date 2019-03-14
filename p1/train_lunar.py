import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(tf.__version__)
print(np.__version__)

import inspect

from training.utils import get_var_name, training_sessions, plot_multiple_sessions


import gym
env = gym.make('LunarLander-v2')

seed = 0
env.seed(seed)
action_size = env.action_space.n
state_size = env.observation_space.shape[0]

print('State shape: ', state_size)
print('Number of actions: ', action_size)

from models.dqn_agent import DQNAgent
from models.ddpg_agent import DDPGAgent
from training.q_learning import dqn
from training.ddpg_learning import ddpg

vanilla = DQNAgent(state_size, action_size, seed, network_type='linear', config='single')
duel = DQNAgent(state_size, action_size, seed, network_type='duel', config='single')
double = DQNAgent(state_size, action_size, seed, network_type='linear', config='double')
prio = DQNAgent(state_size, action_size, seed, network_type='linear', config='single', memory_type='prioritized')

n_episodes = 200
max_t = 100
n_training_sessions = 1


scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std = training_sessions(vanilla, env, n_episodes, max_t, n_training_sessions)
scores_duel_x, scores_duel_mean, scores_duel_std = training_sessions(duel, env, n_episodes, max_t, n_training_sessions)
scores_double_x, scores_double_mean, scores_double_std = training_sessions(double, env, n_episodes, max_t, n_training_sessions)
scores_prio_x, scores_prio_mean, scores_prio_std = training_sessions(prio, env, n_episodes, max_t, n_training_sessions)

fig = plt.figure()
ax = fig.add_subplot(111)

plot_multiple_sessions(scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std, label=get_var_name(vanilla), color='gray')
plot_multiple_sessions(scores_duel_x, scores_duel_mean, scores_duel_std, label=get_var_name(duel), color='blue')
plot_multiple_sessions(scores_double_x, scores_double_mean, scores_double_std, label=get_var_name(double), color='red')
plot_multiple_sessions(scores_prio_x, scores_prio_mean, scores_prio_std, label=get_var_name(prio), color='green')
plt.legend()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

