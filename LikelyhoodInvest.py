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
cut_off = 6
temperature = 0.001
if (Type_of_System[0] == "Vectors"):
    temperature = 0
N_attack = 2
gradient_ascent_step_size = 0.1

diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
#distr = Transformations.diffeo_distr(batch_size,diffeo_shape,device)
shape = (batch_size,) + diffeo_shape
batch_size = 1
Norms = []
Likelyhoods = []
Nres = 1000
#Ms = np.linspace(0,1, num=200, endpoint=True, retstep=False, dtype=None, axis=0)
Ms = np.linspace(0.5, 1, num=Nres, endpoint=True, dtype=None, axis=0)
passed = np.zeros(Nres)
#passed = [0,0,0,0,0]
Number_of_retries = 1
criterion = nn.CrossEntropyLoss()
diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape
for i, data in enumerate(trainloader,0):
#for i, data in enumerate(testloader,0):
#for data in testloader:
    #parameters = torch.randn(shape, requires_grad=True).to(device)
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    #parameters = (2*torch.rand(shape)-torch.ones(shape)).to(device)
    attempt = 0
    m = 0
    while (m < 0.5) and (attempt < Number_of_retries):
        parameters = (10**(-6)+9*(10**(-6))*torch.rand(shape)).to(device)
        print("the parameters of the ",attempt," attempt are ", parameters)
        parameters.requires_grad_(requires_grad= True)
        print("for the ", i , "th picture and the ", attempt ,"attempt")
        #parameters = torch.tensor(Transformations.diffeo_parameters(distr, shape, diffeo_shape, device, temperature), requires_grad=True).to(device)
        Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
        inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
        outputs = Model3(inputs)
        predicted = torch.max(outputs.data, 1).indices.to(device)
        #print ("predicted vs labels", predicted, labels)
        if(predicted == labels):
            #Useful_functions.imshow(inputs[0])
            #print("initially the same")
            outputs_temp = Model3(inputs_temp)
            predicted_temp = torch.max(outputs_temp.data, 1).indices.to(device)
            index = 1
            while (predicted_temp == labels):
                l = criterion(outputs_temp, labels)
                l.backward()
                #print("parameters before",Transf.parameters)
                with torch.no_grad():
                    parameters += gradient_ascent_step_size * (parameters.grad)
                #print("parameters after", Transf.parameters)
                inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                outputs_temp = Model3(inputs_temp)
                predicted_temp = torch.max(outputs_temp.data, 1).indices.to(device)
                index += 1
                #print(index)
            with torch.no_grad():
                #Useful_functions.imshow(inputs_temp[0])
                ### calculate the likelyhood of the diffeomorphism ####
                m = (min(Useful_functions.likelyhood_of_diffeomorphisms(parameters,temperature)))
            for j in range(0,Nres):
                if (m > Ms[j]):
                    passed[j] += 1
            # if (m> 0.5):
            #     passed[0] += 1
            #     print("pass 0.5 ratio is", passed[0] / (i+1), "and i+1 is ", i+1)
            #     if (m> 0.6):
            #         passed[1] += 1
            #         print("pass 0.6 ratio is", passed[1] / (i+1), "and i+1 is ", i+1)
            #         if(m>0.7):
            #             passed[2] += 1
            #             print("pass 0.7 ratio is", passed[2] / (i+1), "and i+1 is ", i+1)
            #             if (m >0.8):
            #                 passed[3] += 1
            #                 print("pass 0.8 ratio is", passed[3] / (i+1), "and i+1 is ", i+1)
            #                 if (m > 0.9):
            #                     passed[4] += 1
            #                     print("pass 0.9 ratio is", passed[4]/(i+1), "and i+1 is ",i+1)
        else:
            m = 1
        attempt += 1
    if ((i+1)%200 == 0):
        print("at the ", i, " th step, the rates of passage are ",passed)
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
