import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, depth, size, width,  nonlinear=False, num_classes=2):

        super(Net, self).__init__()
        self.size = size
        self.nonlinear=nonlinear
        self.width = width
        self.size = size
        self.depth = depth

        self.first_layer = nn.Linear(3*self.size**2, self.width*self.size**2, bias=False)
        nn.init.kaiming_normal_(self.first_layer.weight.data, a=0.01, nonlinearity='leaky_relu')

        module = nn.Linear(self.width*self.size**2, self.width*self.size**2, bias=False)
        nn.init.kaiming_normal_(module.weight.data, a=0.01, nonlinearity='leaky_relu')
        self.layers = nn.ModuleList([deepcopy(module) for idx in range(self.depth-1)])

        self.final_layer = nn.Linear(self.width*self.size**2, num_classes)
        nn.init.kaiming_normal_(self.final_layer.weight.data, nonlinearity='linear')

    def forward(self, x):
        o = x.reshape(-1, 3*self.size**2)
        o = F.leaky_relu(self.first_layer(o))
        for i in range(len(self.layers)):
            layer = self.layers[i]
            o = layer(o)
            if self.nonlinear:
                o = F.leaky_relu(o)
        o = self.final_layer(o)
        return o
