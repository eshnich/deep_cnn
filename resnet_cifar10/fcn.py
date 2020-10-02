import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, depth, size, width, num_classes=10):

        super(Net, self).__init__()
        self.size = size
        self.width =width
        self.layer1 = nn.Linear(3*self.size**2, width*self.size**2)
        self.layer2 = nn.Linear(width*self.size**2, num_classes)

    def forward(self, x):
        o = x.reshape(-1, 3*self.size**2)
        o = F.relu(self.layer1(o))
        return self.layer2(o)
