from collections import deque
import numpy as np
import torch

from training.utils import get_var_name, get_root_dir

def ddpg_unity(agent, env, brain_name, env_info, model_name, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01,
              eps_decay=0.995):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        for t in range(max_t):
            action = agent.act(state, eps)  # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(),
                       get_root_dir() + '/stored_weights/checkpoint_actor_{}.pth'.format(model_name))
            torch.save(agent.critic_local.state_dict(),
                       get_root_dir() + '/stored_weights/checkpoint_critic_{}.pth'.format(model_name))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(),
                       get_root_dir() + '/stored_weights/checkpoint_actor_{}.pth'.format(model_name))
            torch.save(agent.critic_local.state_dict(),
                       get_root_dir() + '/stored_weights/checkpoint_critic_{}.pth'.format(model_name))
            break

    return scores