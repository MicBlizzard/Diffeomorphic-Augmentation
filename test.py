import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Data_treatement
import Transformations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Type_of_System = ["Vectors","Stripe"]
#Type_of_System = ["Vectors","NODVSM"]
Type_of_System = ["Images","CIFAR10"]

#location = "data/CIFAR10"
#trainset, testset, Data_type = Data_treatement.Import(location)
testset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Test_Import_NP()

batch_size = 4

testloader = DataLoader(dataset = testset, batch_size=batch_size, shuffle=False, num_workers=0)

PATH = "./model/net.pth"
net_untrained = torch.load(PATH)
net_untrained.eval()

PATH = "./model/net_trained.pth"
net = torch.load(PATH)
net.eval()

correct_trained = 0
correct_untrained = 0
total = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i, data in enumerate(testloader,0):
    #for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs_trained = net(images)
        outputs_untrained = net_untrained(images)
        #print(outputs_trained)
        #print(outputs_untrained)
        #print(labels)
        # the class with the highest energy is what we choose as prediction
        _, predicted_trained = torch.max(outputs_trained.data, 1)
        _, predicted_untrained = torch.max(outputs_untrained.data, 1)
        total += labels.size(0)
        correct_trained += (predicted_trained == labels).sum().item()
        correct_untrained += (predicted_untrained == labels).sum().item()
print(f'Accuracy of the trained network on the 10000 test images: {100 * correct_trained // total} %')
print(f'Accuracy of the untrained network on the 10000 test images: {100 * correct_untrained // total} %')
print(f'Relative accuracy of the trained network on the 10000 test images: {100 * (correct_trained-correct_untrained) // (total-correct_untrained)} %')


##### testing how "likely" adversarial diffeos are #####
cut_off = 2
temperature = 0.003
if (Type_of_System[0] == "Vectors"):
    temperature = 0
N_attack = 2
gradient_ascent_step_size = 0.01

diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
distr = Transformations.diffeo_distr(batch_size,diffeo_shape,device)
shape = (batch_size,) + diffeo_shape
batch_size = 1
Norms = []
Likelyhoods = []
criterion = nn.CrossEntropyLoss()
diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)

for i, data in enumerate(testloader,0):
#for data in testloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    parameters = torch.tensor(Transformations.diffeo_parameters(distr, shape, diffeo_shape, device, temperature), requires_grad=True).to(device)
    Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
    outputs = net(images)
    predicted = torch.max(outputs.data, 1).to(device)
    if(predicted == labels):
        print("initially the same")
        outputs_temp = net(inputs_temp)
        predicted_temp = torch.max(outputs.data, 1).to(device)
        while (predicted_temp == labels):
            l = criterion(outputs_temp, labels)
            l.backward()
            with torch.no_grad():
                parameters += gradient_ascent_step_size * (parameters.grad)
            inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
            outputs_temp = net(inputs_temp)
            predicted_temp = torch.max(outputs.data, 1).to(device)
#
#         _, predicted_trained = torch.max(outputs_trained.data, 1)
#         _, predicted_untrained = torch.max(outputs_untrained.data, 1)
#         total += labels.size(0)
#
#         correct_trained += (predicted_trained == labels).sum().item()
#         correct_untrained += (predicted_untrained == labels).sum().item()
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
