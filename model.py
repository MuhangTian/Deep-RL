'''To store models (which are building blocks for learning algorithms)'''
import random

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 210, 160, 3            # input dimensions (height, width, channels)
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # the flattened size is 8960 assuming dims and convs above
        self.fc1 = nn.Linear(8960, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, naction)

    def forward(self, X, prev_state=None):
        """
        X - bsz x T x iC x iH x iW observations (in order)
        returns:
          bsz x T x naction action logits, prev_state
        """
        bsz, T = X.size()[:2]

        Z = F.gelu(self.conv3( # bsz*T x hidden_dim x H3 x W3
              F.gelu(self.conv2(
                F.gelu(self.conv1(X.view(-1, self.iC, self.iH, self.iW)))))))

        # flatten with MLP
        Z = F.gelu(self.fc1(Z.view(bsz*T, -1))) # bsz*T x hidden_dim
        Z = Z.view(bsz, T, -1)
        
        return self.fc2(Z), prev_state
    
    def get_action(self, x, prev_state):
        """
        x - 1 x 1 x ic x iH x iW
        returns:
          int index of action
        """
        logits, prev_state = self(x, prev_state)
        # take highest scoring action
        action = logits.argmax(-1).squeeze().item()
        return action, prev_state
    
    
class QNetwork(nn.Module):
    def __init__(self, naction, args) -> None:
        super().__init__()
        self.iH, self.iW, self.iC = 84, 84, 1
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, naction)

    def forward(self, X):
        bsz, T, S = X.size()[:3]
        X = X.view(-1, self.iC, self.iH, self.iW)
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))

        X = X.view(bsz*S, -1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        
        X = X.view(bsz, S, -1)
        X = torch.mean(X, dim=1)

        return X

    def get_action(self, x, prev_state):
        logits = self(x)
        if torch.rand(1).item() < 0.05:     # epsilon-greedy with epsilon=0.05
            action = random.randint(0,8)
        else:
            action = logits.argmax().item()
        
        return action, prev_state
        