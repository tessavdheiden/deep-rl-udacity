import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.a2c_model import ActorCriticNetwork

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024        # minibatch size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item()

class ACAgent:
    
    def __init__(self,
                 state_dim,                 # dimension of the state vector
                 action_dim,                # dimension of the action vector
                 num_envs,                  # number of parallel agents (20 in this experiment)
                 seed,
                 rollout_length=5,          # steps to sample before bootstraping
                 lr=1e-4,                   # learning rate
                 lr_decay=.95,              # learning rate decay rate
                 gamma=.99,                 # reward discount rate
                 value_loss_weight = 1.0,   # strength of value loss
                 gradient_clip = 5,         # threshold of gradient clip that prevent exploding
                 ):
        self.model = ActorCriticNetwork(state_dim, action_dim).to(device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.total_steps = 0
        self.n_envs = num_envs
        self.value_loss_weight = value_loss_weight
        self.gradient_clip = gradient_clip
        self.rollout_length = rollout_length
        self.total_steps = 0
        self.memory = []
        self.memory2 = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, seed)
        
    def act(self, state):
        """
        Sample action along with outputting the log probability and state values, given the states
        """
        state = torch.from_numpy(state).float().to(device=device)
        action, log_prob, state_value = self.model(state)
        action, log_prob, state_value = action.cpu().data.numpy(), log_prob.cpu().data.numpy(), state_value.cpu().data.numpy()
        return action, log_prob, state_value

    def step(self, state, action, reward, next_state, not_dones, log_probs, state_values):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.append([action, reward, log_probs, not_dones, state_values])
        self.memory2.add(action, reward, log_probs, not_dones, state_values)

    def start_learn(self):
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory
            self.learn(experiences)

    def learn(self, experience):
        """
        Updates the actor critic network given the experience
        experience: list [[action,reward,log_prob,done,state_value]]
        """
        processed_experience = [None]* (len(experience) - 1)
        
        _advantage = torch.tensor(np.zeros((self.n_envs, 1))).float().to(device=device)   # initialize advantage Tensor
        _return = experience[-1][-1].detach()                                             # get returns
        for i in range(len(experience)-2,-1,-1):                                          # iterate from the last step
            _action, _reward, _log_prob, _not_done, _value = experience[i]                # get training data
            _not_done = torch.tensor(_not_done,device=device).unsqueeze(1).float()        # masks indicating the episodes not finished
            _reward = torch.tensor(_reward,device=device).unsqueeze(1)                    # get the rewards of the parallel agents
            _next_value = experience[i+1][-1]                                             # get the next states
            _return = _reward + self.gamma * _not_done * _return                          # compute discounted return
            _advantage = _reward + self.gamma * _not_done * _next_value.detach() - _value.detach() # advantage
            processed_experience[i] = [_log_prob, _value, _return,_advantage]
            
        log_prob, value, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_experience))
        policy_loss = -log_prob * advantages                                  # loss of the actor
        value_loss = 0.5 * (returns - value).pow(2)                           # loss of the critic (MSE)
        
        self.optimizer.zero_grad()
        (policy_loss + self.value_loss_weight * value_loss).mean().backward() # total loss
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip) # clip gradient
        self.optimizer.step()
        
        self.total_steps += self.rollout_length * self.n_envs

    
    def save(self, path="./trained_model.checkpoint"):
        """
        Save state_dict of the model
        """
        torch.save({"state_dict":self.model.state_dict}, path)
        
    def load(self, path):
        """
        Load model and decay learning rate
        """
        state_dict = torch.load(path)['state_dict']
        self.model.load_state_dict(state_dict())
        
        # If recoverred for training, the learning rate is decreased
        self.lr *= self.lr_decay
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr

import random
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size


        self.experience = namedtuple("Experience", field_names=["action", "reward", "log_prob", "not_done", "state_value"])
        self.seed = random.seed(seed)

    def add(self, action, reward, log_prob, not_done, state_value):
        """Add a new experience to memory."""
        e = self.experience(action, reward, log_prob, not_done, state_value)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        #states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        log_probs = torch.from_numpy(np.vstack([e.log_prob for e in experiences if e is not None])).float().to(device)
        not_dones = torch.from_numpy(np.vstack([e.not_done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        state_values = torch.from_numpy(np.vstack([e.state_value for e in experiences if e is not None])).float().to(
            device)

        return (actions, rewards, log_probs, not_dones, state_values)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)