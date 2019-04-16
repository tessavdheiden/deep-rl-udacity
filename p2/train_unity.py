import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
from training.utils import get_var_name, training_session_unity, plot_multiple_sessions, get_root_dir

print(torch.__version__)
print(tf.__version__)
print(np.__version__)

from unityagents import UnityEnvironment
from models.ddpg_agent import Agent, SoftAgent
from models.a2c_agent import ACAgent
from training.ddpg import ddpg
from training.a2c import a2c


env = UnityEnvironment(file_name="Reacher_Linux_multi/Reacher_Linux/Reacher.x86_64", no_graphics = True)
seed=0

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

train_mode = True
action_size = brain.vector_action_space_size
env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
state = env_info.vector_observations[0]
state_size = len(state)
num_agents = len(env_info.agents)

n_episodes = 2000
max_t = 1000
n_training_sessions = 1

model_type = 'ddpg'
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
if model_type == 'ddpg':
    ddpg_agent = Agent(state_size, action_size, seed)
    scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std, scores_vanilla_y = training_session_unity(ddpg,
                                                                                                         ddpg_agent,
                                                                                                         env,
                                                                                                         brain_name,
                                                                                                         env_info,
                                                                                                         get_var_name(
                                                                                                             ddpg),
                                                                                                         n_episodes,
                                                                                                         max_t)
    plot_multiple_sessions(scores_vanilla_x, scores_vanilla_mean, scores_vanilla_std, label=get_var_name(ddpg),
                           color='gray')
    plt.plot(np.arange(len(scores_vanilla_y)), scores_vanilla_y, alpha=.1)
elif model_type == 'a2c':
    a2c_agent = ACAgent(state_size, action_size, seed, num_agents, rollout_length=5,lr=1e-4,lr_decay=.95,gamma=.95,value_loss_weight=1,gradient_clip=5)
    scores_a2c_x, scores_a2c_mean, scores_a2c_std, scores_a2c_y = training_session_unity(a2c, a2c_agent, env, brain_name, env_info, get_var_name(ddpg), n_episodes, max_t)
    plot_multiple_sessions(scores_a2c_x, scores_a2c_mean, scores_a2c_std, label=get_var_name(a2c), color='blue')
    plt.plot(np.arange(len(scores_a2c_y)), scores_a2c_y, alpha=.1)

plt.legend()
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(get_root_dir() + '/unity_environment_model_{}_batch_size_128.png'.format(model_type))
plt.show()