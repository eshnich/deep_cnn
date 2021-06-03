import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, depth, size, width=16, nonlinear = False, filters=3, num_classes=2):

        super(Net, self).__init__()
        self.depth = depth
        self.size = size
        self.width = width
        self.filter_size = filters
        self.num_classes=num_classes
        padding = int((filters - 1)/2)
        
        self.first_layer = nn.Conv2d(3, self.width, self.filter_size, padding=padding, stride=1, bias=False)
        nn.init.kaiming_normal_(self.first_layer.weight.data, a=0.01, nonlinearity='leaky_relu')
        
        module = nn.Conv2d(self.width, self.width, self.filter_size, padding=padding, stride=1, bias=False)
        nn.init.kaiming_normal_(module.weight.data, a=0.01, nonlinearity='leaky_relu')
        
        self.nonlinear = nonlinear
        bn = nn.BatchNorm2d(self.width)
        self.bn_layers = nn.ModuleList([deepcopy(bn) for idx in range(self.depth)])
        self.layers = nn.ModuleList([deepcopy(module) for idx in range(self.depth-1)])
        
        self.final_layer = nn.Conv2d(self.width, self.num_classes, self.filter_size, padding=padding, stride=1, bias=False)
        nn.init.kaiming_normal_(self.final_layer.weight.data, nonlinearity='linear')
        print("kaiming norm")
        
        
    def forward(self, x):
        o = F.leaky_relu(self.bn_layers[0](self.first_layer(x)))
        #o = F.leaky_relu(self.first_layer(x))
        for i in range(len(self.layers)):
            layer = self.layers[i]
            o = layer(o)
            o = self.bn_layers[i+1](o)
            if self.nonlinear:
                o = F.leaky_relu(o)
        #o = o.reshape(-1, 3*self.size**2)
        o = self.final_layer(o)
        o = o.reshape(-1, self.num_classes, self.size**2)
        o = torch.mean(o, 2)
        return o
