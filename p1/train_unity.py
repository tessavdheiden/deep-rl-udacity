import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
from training.utils import get_var_name, training_sessions_unity, plot_multiple_sessions, get_root_dir

print(torch.__version__)
print(tf.__version__)
print(np.__version__)

from training.q_learning import dqn_unity
from unityagents import UnityEnvironment
from models.dqn_agent import DQNAgent

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86")
seed=0
#env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

train_mode = True
action_size = brain.vector_action_space_size
env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
state = env_info.vector_observations[0]
state_size = len(state)

vanilla = DQNAgent(state_size, action_size, seed, network_type='linear', config='single')
duel = DQNAgent(state_size, action_size, seed, network_type='duel', config='single')
double = DQNAgent(state_size, action_size, seed, network_type='linear', config='double')
prio = DQNAgent(state_size, action_size, seed, network_type='linear', config='single', memory_type='prioritized')
double_prio = DQNAgent(state_size, action_size, seed, network_type='linear', config='double', memory_type='prioritized')

n_episodes = 2000
max_t = 100
n_training_sessions = 1

scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std = training_sessions_unity(vanilla, env, brain_name, env_info, get_var_name(vanilla), n_episodes, max_t, n_training_sessions)
scores_duel_x, scores_duel_mean, scores_duel_std = training_sessions_unity(duel, env, brain_name, env_info, get_var_name(duel), n_episodes, max_t, n_training_sessions)
scores_double_x, scores_double_mean, scores_double_std = training_sessions_unity(double, env, brain_name, env_info, get_var_name(double),n_episodes, max_t, n_training_sessions)
scores_prio_x, scores_prio_mean, scores_prio_std = training_sessions_unity(prio, env, brain_name, env_info, get_var_name(prio),n_episodes, max_t, n_training_sessions)
scores_double_prio_x, scores_double_prio_mean, scores_double_prio_std = training_sessions_unity(double_prio, env, brain_name, env_info,get_var_name(double_prio), n_episodes, max_t, n_training_sessions)


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plot_multiple_sessions(scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std, label=get_var_name(vanilla), color='gray')
plot_multiple_sessions(scores_duel_x, scores_duel_mean, scores_duel_std, label=get_var_name(duel), color='blue')
plot_multiple_sessions(scores_double_x, scores_double_mean, scores_double_std, label=get_var_name(double), color='red')
plot_multiple_sessions(scores_prio_x, scores_prio_mean, scores_prio_std, label=get_var_name(prio), color='green')
plot_multiple_sessions(scores_double_prio_x, scores_double_prio_mean, scores_double_prio_std, label=get_var_name(double_prio), color='orange')
plt.legend()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(get_root_dir() + '/benchmark_unity_environment.png')
plt.show()
