from collections import deque
import numpy as np
import torch
import time

from training.utils import get_var_name, get_root_dir


def ddpg(agent, env, num_agents, train_mode, brain_name, model_name, n_episodes=2000, max_t=1000, print_every=10, learn_every=20, num_learn=10, goal_score=30):
    total_scores_deque = deque(maxlen=100)
    total_scores = []

    for i_episode in range(1, n_episodes + 1):
        # Reset Env and Agent
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        agent.reset()

        start_time = time.time()

        for t in range(max_t):
            actions = agent.act(states)

            env_info = env.step(actions)[brain_name]  # send all actions to the environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)

            dones = env_info.local_done  # see if episode finished

            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)  # send actions to the agent

            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step

            if t % learn_every == 0:
                for _ in range(num_learn):
                    agent.start_learn()

            if np.any(dones):  # exit loop if episode finished
                break

        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        total_scores_deque.append(mean_score)
        total_scores.append(mean_score)
        total_average_score = np.mean(total_scores_deque)
        duration = time.time() - start_time

        print('\rEpisode {}\tTotal Average Score: {:.2f}\tMean: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tDuration: {:.2f}'
              .format(i_episode, total_average_score, mean_score, min_score, max_score, duration))

        if i_episode % print_every == 0:
            torch.save(agent.actor_local.state_dict(), get_root_dir() +  '/stored_weights/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), get_root_dir() +  '/stored_weights/checkpoint_critic.pth')
            print('\rEpisode {}\tTotal Average Score: {:.2f}'.format(i_episode, total_average_score))

        if total_average_score >= goal_score and i_episode >= 100:
            print('Problem Solved after {} epsisodes!! Total Average score: {:.2f}'.format(i_episode,
                                                                                           total_average_score))
            torch.save(agent.actor_local.state_dict(),
                       get_root_dir() + '/stored_weights/checkpoint_actor_{}.pth'.format(model_name))
            torch.save(agent.critic_local.state_dict(),
                       get_root_dir() + '/stored_weights/checkpoint_critic_{}.pth'.format(model_name))
            break

    return total_scores
