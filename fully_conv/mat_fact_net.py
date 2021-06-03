import torch.nn as nn
import torch
from copy import deepcopy
import torch.nn.functional as F

class MatFactLayer(nn.Module):
    
    def __init__(self, size, filters_in, filters_out, bias=True):
        super(MatFactLayer, self).__init__()
        bound = 5e-2
        self.size = size
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.bias = bias
        init = torch.Tensor(filters_out, filters_in, size, size).uniform_(-bound, bound)
        self.weights = nn.Parameter(init)
        if self.bias:
            self.bias_weights = nn.Parameter(torch.zeros(filters_out,size, size))

    def forward(self, x):
        o = torch.cat([torch.sum(torch.matmul(x[0], self.weights), dim=1, keepdim=True).reshape(-1, self.filters_out, self.size, self.size) for i in range(x.size()[0])])
        if self.bias:
            o += self.bias_weights
        return o

class MatFactNet(nn.Module):

    def __init__(self, depth, size, nonlinear = False):

        super(MatFactNet, self).__init__()
        self.depth = depth
        self.size = size
        self.width = 8
        self.first_layer = MatFactLayer(self.size, 3, self.width,  bias=True)
        module = MatFactLayer(self.size, self.width, self.width, bias=True)
        self.nonlinear = nonlinear
        self.layers = nn.ModuleList([deepcopy(module) for idx in range(self.depth-1)])
        self.final_layer = MatFactLayer(self.size, self.width, 2, bias=True)
        #self.final_layer = nn.Linear(3*self.size**2, 2)

    def forward(self, x):
        o = self.first_layer(x)
        for layer in self.layers:
            o = layer(o)
            if self.nonlinear:
                o = F.leaky_relu(o)
        o = self.final_layer(o)
        o = o.reshape(-1, 2, self.size**2)
        o = torch.mean(o, 2)
        return o

