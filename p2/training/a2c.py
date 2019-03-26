from unityagents import UnityEnvironment
import numpy as np
import torch
import time
from collections import deque
from training.utils import get_root_dir


def a2c(agent, env, num_agents, train_mode, brain_name, model_name, n_episodes=2000, max_t=1000, print_every=10, learn_every=20, num_learn=10, goal_score=30, rollout_length=5):
    total_scores_deque = deque(maxlen=100)
    total_scores = []

    for i_episode in range(1, n_episodes + 1):
        # Reset Env and Agent
        env_info = env.reset(train_mode=train_mode)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        #agent.reset()

        start_time = time.time()
        steps_taken = 0
        for t in range(max_t):
            steps_taken +=1
            actions, log_probs, state_values = agent.act(states)      # select actions for 20 envs
            env_inst = env.step(actions)[brain_name]     # send the actions to the environment
            next_states = env_inst.vector_observations                          # get the next states
            rewards = env_inst.rewards                                          # get the rewards
            dones = env_inst.local_done                                         # see if episode has finished
            not_dones = [1-done for done in dones]        

            for state, action, reward, next_state, not_done, log_prob, state_value in zip(states, actions, rewards, next_states, not_dones, log_probs, state_values):
                agent.step(state, action, reward, next_state, not_done, log_prob, state_value)  # send actions to the agent

            #if steps_taken % rollout_length == 0:
            #    agent.update_model()

                
            scores += rewards                                                   # update the scores
            states = next_states                                                # roll over the states to next time step


            scores += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step

            if t % learn_every == 0:
                for _ in range(num_learn):
                    agent.start_learn()
                del agent.memory[:]

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
            torch.save(agent.model.state_dict(), '/stored_weights/checkpoint_model.pth')
            print('\rEpisode {}\tTotal Average Score: {:.2f}'.format(i_episode, total_average_score))

        if total_average_score >= goal_score and i_episode >= 100:
            print('Problem Solved after {} epsisodes!! Total Average score: {:.2f}'.format(i_episode,
                                                                                           total_average_score))
            torch.save(agent.model.state_dict(),
                       get_root_dir() + '/stored_weights/checkpoint_model_{}.pth'.format(model_name))

            break

    return total_scores
