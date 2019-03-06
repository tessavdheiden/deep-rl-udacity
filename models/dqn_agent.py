import numpy as np
import random
from collections import namedtuple, deque

from models.q_network import QNetwork, QDuelingNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 8         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, config='single',  network_type='linear', memory_type='random'):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.config = config

        # Q-Network
        if network_type == 'duel':
            self.qnetwork_local = QDuelingNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QDuelingNetwork(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        elif network_type == 'linear': 
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory_type = memory_type
        if memory_type == 'prioritized':
            self.memory = NaivePrioritizedBuffer(BUFFER_SIZE, BATCH_SIZE)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.__len__() > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next_values = self.qnetwork_target(next_states).detach()
        if self.config == 'single':
            Q_targets_next = Q_targets_next_values.max(1)[0].unsqueeze(1)
        elif self.config == 'double':
            best_actions = self.qnetwork_local(next_states).max(1)[1] # values, indices = tensor.max(0)
            idx = best_actions.unsqueeze(1)
            Q_targets_next = torch.gather(Q_targets_next_values[best_actions], 1, idx) 

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(1, actions)

        # Compute loss
        #loss = F.mse_loss(Q_expected, Q_targets) * 1 #
        loss = ((Q_targets - Q_expected).pow(2)*weights)

        # Minimize the loss
        self.optimizer.zero_grad()
        if self.memory_type == 'prioritized':
            prios = loss + 1e-5
            self.memory.update_priorities(indices, prios.data.cpu().numpy())
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, torch.zeros_like(dones), torch.ones_like(dones))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, batch_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = BUFFER_SIZE
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity,batch_size), dtype=np.float32)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        states = torch.from_numpy(np.vstack([s for s in states if s is not None])).float().to(device).squeeze()
        actions = batch[1]
        actions = torch.from_numpy(np.vstack([a for a in actions if a is not None])).long().to(device)
        rewards = batch[2]
        rewards = torch.from_numpy(np.vstack([a for a in rewards if a is not None])).float().to(device).squeeze()
        next_states = np.concatenate(batch[3])
        next_states = torch.from_numpy(np.vstack([s for s in next_states if s is not None])).float().to(device).squeeze()
        dones = batch[4]
        dones = torch.from_numpy(np.vstack([e for e in dones if e is not None]).astype(np.uint8)).float().to(device)

        weights = torch.from_numpy(np.vstack([s for s in weights if s is not None])).float().to(device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)