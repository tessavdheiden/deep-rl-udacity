import os
import numpy as np
import inspect

def get_root_dir():
   path_this_file = os.path.dirname(os.path.realpath(__file__))
   return ('/').join(path_this_file.split('/')[:-1])

def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    name = [k for k, v in callers_local_vars if v is var]
    return name[0]

from training.ddpg import ddpg

def training_session_unity(agent, env, brain_name, env_info, model_name, n_episodes, max_t, buckets=5):
    num_agents = len(env_info.agents)
    scores_y = np.asarray(ddpg(agent, env, num_agents, True, brain_name, model_name, n_episodes, max_t))

    bucket_size = n_episodes // buckets
    scores_mean = np.zeros(buckets+1)
    scores_std = np.zeros(buckets+1)
    scores_x = np.arange(buckets+1) * (n_episodes // buckets)
    for bucket in range(buckets):
        start = bucket*bucket_size
        end = start + bucket_size
        scores_mean[bucket+1] = scores_y[start:end].mean(0)
        scores_std[bucket+1] = scores_y[start:end].std(0)

    return scores_x, scores_mean, scores_std, scores_y

import matplotlib.pyplot as plt

def plot_multiple_sessions(scores_x, scores_mean, scores_std, label='no name', color='gray'):

    plt.plot(scores_x, scores_mean, color=color, label=label)
    plt.fill_between(scores_x, scores_mean - scores_std, scores_mean + scores_std, color=color, alpha=0.2)



