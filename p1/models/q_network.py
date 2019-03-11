import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models.layer_utils import make_fc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QDuelingNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QDuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        dims = [state_size, fc1_units, fc2_units]
        self.fc = make_fc(dims) 
        #self.adv = nn.Linear(fc2_units, action_size)
        #self.val = nn.Linear(fc2_units, 1)
        self.adv = nn.Sequential(
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size)
        )

        self.val = nn.Sequential(
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc(state)
        adv = self.adv(x)
        val = self.val(x)
        q = val - adv.mean(1, keepdim=True) + adv
        return q

class QActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        dims1 = [state_size, fc1_units, fc2_units]
        self.fc = make_fc(dims1)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc(state)
        probs = F.softmax(self.fc3(x))
        m = Categorical(probs)
        action = m.sample()
        return probs, action

class QCriticNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        dims1 = [state_size, fc1_units, fc2_units]
        self.fc1 = make_fc(dims1)
        dims2 = [action_size + fc2_units, fc2_units]
        self.fc2 = make_fc(dims2)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state, action):
        """Build a network that maps state -> action values."""
        xs = self.fc1(state)
        action = action.float()
        x = torch.cat([xs, action], dim=1)
        x = self.fc2(x)
        return self.fc3(x)
