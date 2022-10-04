import math
import random
#from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGG(nn.Module):
    def __init__(self, configuration, image_channel=3, num_classes=10, batch_norm=True, pooling_size=2, param_list=False, width_factor=1):
        super(VGG, self).__init__()
        self.convolution = self._convolution_layer(configuration, ch=image_channel, bn=batch_norm, ps=pooling_size, param_list=param_list, width_factor=width_factor)
        self.classifier = nn.Linear(int(512 * width_factor), num_classes)

    def forward(self, x):
        out = self.convolution(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _convolution_layer(self, cfg, ch, bn, ps, param_list, width_factor):
        layers = []
        in_channels = ch
        if ch == 1:
            layers.append(nn.ZeroPad2d(2))
        if param_list:
            convLayer = Conv2dList
        else:
            convLayer = nn.Conv2d
        for x in cfg:
            if (isinstance(x, str)):
                if (x == 'M'):
                    layers += [nn.MaxPool2d(kernel_size=ps, stride=2, padding=ps // 2 + ps % 2 - 1)]
                elif (x == 'A'):
                    layers += [nn.AvgPool2d(kernel_size=ps, stride=2, padding=ps // 2 + ps % 2 - 1)]
                else:
                    layers += [SubSampling(kernel_size=ps, stride=2)]
            else:
                x = int(x * width_factor)
                if bn:
                    layers += [convLayer(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [convLayer(in_channels, x, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


    def _fully_connected_creator(self, nbr_fully_connected, num_classes, size):
            ## the idea here is to creat the different convolution layers with a max pooling ever so often
            layers = []
            for i in range (0,nbr_fully_connected-1):
                print(size)
                layers += [nn.Linear(size, size),nn.ReLU(inplace=True)]
            layers += [nn.Linear(size, num_classes), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

class Conv2dList(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='constant'):
        super().__init__()

        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)

        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if bias is not None:
            bias = nn.Parameter(torch.empty(out_channels, ))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        n = max(1, 128 * 256 // (in_channels * kernel_size * kernel_size))
        weight = nn.ParameterList([nn.Parameter(weight[j: j + n]) for j in range(0, len(weight), n)])

        setattr(self, 'weight', weight)
        setattr(self, 'bias', bias)

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride

    def forward(self, x):

        weight = self.weight
        if isinstance(weight, nn.ParameterList):
            weight = torch.cat(list(self.weight))

        return F.conv2d(x, weight, self.bias, self.stride, self.padding)

class SubSampling(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(SubSampling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)[..., 0, 0]

def configuration_builder(convolution_channels,convolution_depths,types_of_pooling):
    configuration = []
    for i in range(0,len(types_of_pooling)):
        for j in range(0,convolution_depths[i]):
            configuration += [convolution_channels[i]] # first convolution
        configuration += [types_of_pooling[i]]
    return configuration

class SimpleFCC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleFCC, self).__init__()
        # self parameters

        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

