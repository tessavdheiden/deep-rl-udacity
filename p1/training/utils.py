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

from training.q_learning import dqn, dqn_unity

def training_sessions_unity(agent, env, brain_name, env_info, model_name, n_episodes, max_t, n_training_sessions):
    scores_list = []
    for session in range(n_training_sessions):
        scores = np.asarray(dqn_unity(agent, env, brain_name, env_info, model_name, n_episodes, max_t))
        scores_list.append(scores)

    scores_list = np.asarray(scores_list)
    scores_x = np.arange(scores_list.shape[1])
    scores_mean = scores_list.mean(0)
    scores_std = scores_list.std(0)
    return scores_x, scores_mean, scores_std


def training_sessions(agent, env, n_episodes, max_t, n_training_sessions, model_name):
    scores_list = []
    for session in range(n_training_sessions):
        scores = np.asarray(dqn(agent, env, n_episodes, max_t, model_name))
        scores_list.append(scores)

    scores_list = np.asarray(scores_list)
    scores_x = np.arange(scores_list.shape[1])
    scores_mean = scores_list.mean(0)
    scores_std = scores_list.std(0)
    return scores_x, scores_mean, scores_std

import matplotlib.pyplot as plt

def plot_multiple_sessions(scores_x, scores_mean, scores_std, label='no name', color='gray'):
    plt.plot(scores_x, scores_mean, color=color, label=label)
    plt.fill_between(scores_x, scores_mean - scores_std, scores_mean + scores_std, color=color, alpha=0.2)



