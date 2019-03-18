import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
from training.utils import get_var_name, training_session_unity, plot_multiple_sessions, get_root_dir

print(torch.__version__)
print(tf.__version__)
print(np.__version__)

from training.ddpg import ddpg_unity
from unityagents import UnityEnvironment
from models.ddpg_agent import Agent, SoftAgent

env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64", no_graphics = True)
seed=0

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

train_mode = True
action_size = brain.vector_action_space_size
env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
state = env_info.vector_observations[0]
state_size = len(state)

#ddpg = Agent(state_size, action_size, seed)
soft = SoftAgent(state_size, action_size, seed)

n_episodes = 500
max_t = 1000
n_training_sessions = 1

#scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std, scores_vanilla_y = training_session_unity(ddpg, env, brain_name, env_info, get_var_name(ddpg), n_episodes, max_t)
scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std, scores_vanilla_y = training_session_unity(soft, env, brain_name, env_info, get_var_name(soft), n_episodes, max_t)


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
#plot_multiple_sessions(scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std, label=get_var_name(ddpg), color='gray')
plot_multiple_sessions(scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std, label=get_var_name(soft), color='blue')
plt.plot(np.arange(len(scores_vanilla_y)), scores_vanilla_y, alpha=.1)
plt.legend()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(get_root_dir() + '/benchmark_unity_environment.png')
plt.show()