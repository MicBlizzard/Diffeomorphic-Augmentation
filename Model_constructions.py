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

import Model_definitions




convolution_channels = [64,128,256,512,512]
convolution_depths = [1,1,1,2,2]
types_of_pooling = ['M','M','M','M','M']
pool_factors = [2,2,2,2,2]
#net = Model_definitions.VGG(Model_definitions.configuration_builder(convolution_channels,convolution_depths,types_of_pooling))
net = Model_definitions.SimpleFCC(3,100,2)
#untrained_net = Model_definitions.SimpleFCC(2,100,2)
PATH = "./model/net.pth"
#PATH_DIC = "net_dic.pth"
#PATH_untrained = "model_untrained.pth"
#torch.save(net.state_dict(), PATH)
torch.save(net, PATH)
#torch.save(net.state_dict(), PATH_DIC)
#torch.save(untrained_net, PATH_untrained)
#
# Net = torch.load(PATH)
# Net.eval()
#
# statement = True
# for p1, p2 in zip(net.parameters(), Net.parameters()):
#     if p1.data.ne(p2.data).sum() > 0:
#         statement = False
#
# print(statement)

# Find the difference in the two models
