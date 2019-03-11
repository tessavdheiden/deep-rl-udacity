import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_fc(dims, activation=None, batch_norm=False):
    layers = [nn.Linear(dims[0], dims[1])]
    for i in range(1, len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]).to(device))
        if activation == 'relu':
            layers.append(nn.ReLU().to(device))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dims[i+1]).to(device))
    fc = nn.Sequential(*layers).to(device)
    return fc
