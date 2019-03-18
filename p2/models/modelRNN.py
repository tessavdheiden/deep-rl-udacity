import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorRNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(ActorRNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.lstm = nn.LSTM(fc_units, fc_units, 1, dropout=0.0)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.lstm(states)
        x = F.relu(self.fc1(x))
        return F.tanh(self.fc2(x))


class CriticRNN(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticRNN, self).__init__()
        hidden_dim = 64
        self.seed = torch.manual_seed(seed)
        self.encode_state = nn.LSTM(state_size, hidden_dim, 1, dropout=0.0)
        self.fc1 = nn.Linear(hidden_dim, fcs1_units)

        self.encode_action = nn.LSTM(action_size, hidden_dim, 1, dropout=0.0)
        self.fc2 = nn.Linear(hidden_dim, fc2_units)

        self.fc3 = nn.Linear(fcs1_units + fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encode_state.weight.data.uniform_(*hidden_init(self.encode_state))
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.encode_action.weight.data.uniform_(*hidden_init(self.encode_action))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = self.lstm(states)
        action = self.lstm(actions)
        xs = F.leaky_relu(self.fcs1(state))
        xa = F.leaky_relu(self.fcs2(action))
        x = torch.cat((xs, xa), dim=1)
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
