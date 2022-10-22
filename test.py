import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import Model_definitions_Leo
import Data_treatement
import Transformations
import Useful_functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PATH_Untrained ="./model/netuntrained.pth"
# Net = torch.load(PATH_Untrained)
# Net.eval()
# Net.to(device)

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

PATH = "./model/vggbn_trained.pt"
Final_PATHS = "./model/vggbn_trained"
Model1 = Model_definitions_Leo.VGG("VGG11")
Model2 = Model_definitions_Leo.VGG("VGG11")
Model3 = Model_definitions_Leo.VGG("VGG11")
M = [Model1,Model2,Model3]
checkpoint = torch.load(PATH)
index = 1
for c in checkpoint:
    temp = torch.load(Final_PATHS +str(index) +".pt")
    M[index-1].load_state_dict(temp)
    M[index - 1].eval()
    index += 1

#Type_of_System = ["Vectors","Stripe"]
#Type_of_System = ["Vectors","NODVSM"]
Type_of_System = ["Images","CIFAR10"]

testset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Test_Import_NP()

batch_size = 4

testloader = DataLoader(dataset = testset, batch_size=batch_size, shuffle=False, num_workers=0)

#correct_untrained = 0
totals = np.zeros(10)
correct = np.zeros(10)

# since we're not training, we don't need to calculate the gradients for our outputs
for m in M:
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            print(i)
        #for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs_trained = m(images)
            #outputs_untrained = net_untrained(images)
            #print(outputs_trained)
            #print(outputs_untrained)
            #print(labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted_trained = torch.max(outputs_trained.data, 1)
            #_, predicted_untrained = torch.max(outputs_untrained.data, 1)
            for i,l in enumerate(labels,0):
                totals[l] += 1
                if (l==predicted_trained[i]):
                    correct[l] += 1
            #correct_untrained += (predicted_untrained == labels).sum().item()
    for c in range(0,10):
        print('Accuracy of the trained network for class ',c,' is :',correct[c]/totals[c])
    # print(f'Accuracy of the untrained network on the 10000 test images: {100 * correct_untrained // total} %')
    # print(f'Relative accuracy of the trained network on the 10000 test images: {100 * (correct_trained-correct_untrained) // (total-correct_untrained)} %')





















# #
# # # !/usr/bin/env python
# # # coding: utf-8
# #
# # # In[ ]:
# #
# #
# # # Diffeomorphism stability
# #
# # # D_f is <||>
# #
# # # for loop on 500 images and 500 diffeos
# # delta = 1
# # c = 3
# # N = 32
# # T = 4 * (delta ** 2) / (math.pi * (N ** 2) * np.log(c))
# #
# # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# # X = []
# # y = []
# # for i in range(0, 500):
# #     X += [testset[i][0]]
# #     y += [testset[i][1]]
# #
# # X = torch.stack(X)
# # y = torch.tensor(y)
# #
# # # In[ ]:
# #
# #
# # Df = 0
# # Norm = 0
# # Nf = 0
# #
# # N = 20
# # for i in range(0, N):
# #     C_list_new = torch.rand(500, c, c)  # one C matrix per image
# #     X_tilde = Trans(X, T, c, C_list_new)
# #
# #     with torch.no_grad():
# #         delta = ((torch.norm(net(X_tilde) - net(X), p='fro') / len(X)).item())
# #         norm = (torch.norm((X_tilde) - (X), p='fro') / len(X)).item()
# #         # print(delta)
# #         # print(norm)
# #         Norm += norm
# #         Df += delta
# #
# # Df /= N
# # Norm /= N
# #
# # print(Df)
# # print(Norm)
# #
# # for i in range(0, N):
# #     Noise = torch.rand(500, 3, 32, 32)  # one C matrix per image
# #     for j in range(0, 500):
# #         Noise[j] *= Norm / torch.norm(Noise[j], p='fro').item()
# #
# #     X_tilde = X + Noise[j]
# #
# #     with torch.no_grad():
# #         delta = ((torch.norm(net(X_tilde) - net(X), p='fro') / len(X)).item())
# #         print(delta)
# #         Nf += delta
# #
# # Nf /= N
# # print(Nf)
#
#
