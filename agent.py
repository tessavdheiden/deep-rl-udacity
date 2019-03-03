import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9
    
    def epsilon_greedy(self, state):
        policy                  = np.ones(self.nA) * (self.epsilon/self.nA)
        best_action_idx         = np.argmax(self.Q[state])
        policy[best_action_idx] = (1 - self.epsilon) + (self.epsilon / self.nA)
        return policy

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action_probabilities  = self.epsilon_greedy(state)
        return np.random.choice(np.arange(self.nA), p=action_probabilities)
    
    def discounted_reward(self, state):
        return self.gamma * np.max(self.Q[state])
    
    def sarsa_max_update(self, s, a, r, new_s): 
        self.Q[s][a] = self.Q[s][a] + (self.alpha * (r + self.discounted_reward(new_s) -  self.Q[s][a]))
    
    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.sarsa_max_update(state, action, reward, next_state)

