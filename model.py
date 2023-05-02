'''To store models (which are building blocks for learning algorithms)'''
from torch import nn
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


class CriticNetworkSimple(nn.Module):
    '''Value network to act as the critic in actor-critic algorithms'''
    def __init__(self) -> None:
        super().__init__()
        self.iH, self.iW, self.iC = 210, 160, 3             # input dimensions (height, width, channels)
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(8960, 256)
        self.fc2 = nn.Linear(256, 1)            # output is a single value (value of state)
    
    def forward(self, X):
        bsz, T = X.size()[:2]

        Z = F.gelu(self.conv3( # bsz*T x hidden_dim x H3 x W3
              F.gelu(self.conv2(
                F.gelu(self.conv1(X.view(-1, self.iC, self.iH, self.iW)))))))

        Z = F.gelu(self.fc1(Z.view(bsz*T, -1))) # bsz*T x hidden_dim
        Z = Z.view(bsz, T, -1)
        
        return self.fc2(Z)


class ActorNetworkSimple(nn.Module):
    '''Actor network to act as the actor in actor-critic algorithms'''
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
        Z = self.fc2(Z)
        Z = F.softmax(Z, dim=-1)
        
        return Z, prev_state
    
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


class ActorNetworkSimple(nn.Module):
    '''Actor network to act as the actor in actor-critic algorithms'''
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
        Z = self.fc2(Z)
        Z = F.softmax(Z, dim=-1)
        
        return Z, prev_state
    
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


class ActorNetworkCNN(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 210, 160, 3
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(8960, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, naction)

    def forward(self, X, prev_state=None):
        bsz, T = X.size()[:2]

        Z = F.gelu(self.bn1(self.conv1(X.view(-1, self.iC, self.iH, self.iW))))
        Z = F.gelu(self.bn2(self.conv2(Z)))
        Z = F.gelu(self.bn3(self.conv3(Z)))
        Z = F.gelu(self.bn4(self.conv4(Z)))

        Z = F.gelu(self.fc1(Z.view(bsz*T, -1)))
        Z = Z.view(bsz, T, -1)
        Z = self.fc2(Z)
        Z = F.softmax(Z, dim=-1)

        return Z, prev_state

    def get_action(self, x, prev_state):
        logits, prev_state = self(x, prev_state)
        action = logits.argmax(-1).squeeze().item()
        return action, prev_state