import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import Model_definitions_Leo
import Data_treatement
import Transformations
import Useful_functions
import train
import matplotlib.pyplot as plt

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
    index += 1
#M = [Model1]
#Type_of_System = ["Vectors","Stripe"]
#Type_of_System = ["Vectors","NODVSM"]
Type_of_System = ["Images","CIFAR10"]
# #
#
# PATH = "./model/resnet18_trained.pt"
# Final_PATHS = "./model/resnet18_trained"
# Model1 = Model_definitions_Leo.ResNet18()
# Model2 = Model_definitions_Leo.ResNet18()
# Model3 = Model_definitions_Leo.ResNet18()
# M = [Model1,Model2,Model3]
# checkpoint = torch.load(PATH)
# index = 1
# for c in checkpoint:
#     temp = torch.load(Final_PATHS +str(index) +".pt")
#     M[index-1].load_state_dict(temp)
#     index += 1
# #M = [Model1]
# #Type_of_System = ["Vectors","Stripe"]
# #Type_of_System = ["Vectors","NODVSM"]
# Type_of_System = ["Images","CIFAR10"]


M = [Model1,Model2]
#location = "data/CIFAR10"
#trainset, testset, Data_type = Data_treatement.Import(location)
#testset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Test_Import_NP()
trainset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Train_Import_NP()

batch_size = 1
#
trainloader = DataLoader(dataset = trainset, batch_size=batch_size, shuffle=False, num_workers=0)

##### testing how "likely" adversarial diffeos are #####

cut_off = 6
temperature = 0.001
if (Type_of_System[0] == "Vectors"):
    temperature = 0
N_attack = 2
gradient_ascent_step_size = 0.1
gradient_step_step = 1

diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
#distr = Transformations.diffeo_distr(batch_size,diffeo_shape,device)
shape = (batch_size,) + diffeo_shape
batch_size = 1
Nres = 2
Nimages = 50
Norms = np.zeros((3,Nimages,Nres,10))
Losses = np.zeros((3,Nimages,Nres,10))

#GA = np.linspace(0.01, 1, num=Nres, endpoint=True, dtype=None, axis=0)
GA = np.logspace(-2,-1, num=Nres, endpoint=True,base=10.0, dtype=None, axis=0)

passed = np.zeros(Nres)
Number_of_retries = 1
criterion = nn.CrossEntropyLoss()
diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape
Cs = torch.zeros((3,Nres) + shape).to(device)
for n,m in enumerate(M,0):
    Ms = []
    Ms_per_class = [[], [], [], [], [], [], [], [], [], []]
    Nrealex = 0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = m(inputs)
        #with torch.no_grad():
        predicted = torch.max(outputs.data, 1).indices.to(device)
        p = torch.max(outputs.data, 1)
        li = criterion(outputs, labels)
        if(predicted==labels):
            Nrealex += 1
            for n_a, a in enumerate(GA, 0):
                parameters = (10 **(-6)*torch.ones(shape)).to(device)
                #parameters = (10**(-6)+9*(10**(-6))*torch.rand(shape)).to(device)
                parameters.requires_grad_(requires_grad= True)
                Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                outputs = m(inputs_temp)
                predicted = torch.max(outputs.data, 1).indices.to(device)
                index = 1
                for k in range(0,1):
                #while ((predicted == labels)):
                    print(index)
                    Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    outputs = m(inputs_temp)
                    predicted = torch.max(outputs.data, 1).indices.to(device)
                    l = criterion(outputs, labels)
                    l.backward()
                    #print(l)
                    delta_L = l-li
                    #print(delta_L)

                    with torch.no_grad():
                        parameters += a * (parameters.grad)

                        # if (index == 1)
                        #   parameters += a * (parameters.grad)
                        #else:
                        #   parameters += a*(gamma*parameters.grad + (1-gamma)*old)
                        # old = parameters.grad
                        ##### escape methods
                        #parameters += min(index,10)*a * (parameters.grad)
                        #parameters += np.sqrt(index)*a * (parameters.grad)
                        #parameters += (index) * a * (parameters.grad)

                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    outputs = m(inputs_temp)
                    predicted = torch.max(outputs.data, 1).indices.to(device)
                    index += 1
                    #parameters.grad.zero_()
                    if ( torch.norm((parameters)).item() > 500)or(index > 80/a): # arbitrary limit...
                        print("faulty image",Nrealex,n_a)
                        break

                print(Nrealex)
                with torch.no_grad():
                    Cs[n,n_a,:,:,:] += torch.abs(parameters)
                    Norms[n,Nrealex-1,n_a,labels.item()] = torch.norm((parameters)).item()
                    Losses[n,Nrealex-1,n_a,labels.item()] = (l-li).item()
                        #print(Norms)
                        # Ms += [torch.norm((parameters)).item()]
                        # Ms_per_class[labels.item()] += [torch.norm((parameters)).item()]
                    #print(Ms)
            if (Nrealex >= Nimages):
                break

    for j in range(0,10):
        for k in range(0,Nres):
            Average = np.sum(Norms[n,:,k,j])
            print('the average with resolution ', GA[k], 'of the ',j,'th class is ',Average)

        print("")

    print("the C's for model ", n+1, "are ",Cs[n,:,:,:,:]/Nimages)

    X = np.linspace(0,35,36)

fig, axs = plt.subplots(1, len(M))
fig.suptitle('Stepsizes for different networks')
for n in range(0,len(M)):
    for i in range(0,Nres):
        Y = torch.reshape(Cs[n, i, 0, :, :] / Nimages,(36,1))
        axs[n].plot(X, Y, label = "step size of "+str(GA[i]))
    axs[n].legend()
plt.show()


# fig, axs = plt.subplots(1, len(M))
# fig.suptitle(r'\delta C(\alpha) convergence for different networks')
# for n in range(0,len(M)):
#   Y = np.zeros(Nres)
#     for i in range(0,Nres):
#         Y[i] = Cs[n, i, 0, 0, 0].item() / Cs[n, i, 0, 0, 0].item()
#     axs[n].plot(GA, Y, label = "step size of "+str(GA[i]))
#     axs[n].legend()
# plt.show()



np.save('./results/B1AverageCs', Cs.cpu().detach().numpy())
np.save('./results/B1norms', Norms)
np.save('./results/B1Losses', Losses)



#
# ########## Saving ##########
#
# np.save('./results/B2Parameters', Parameters)
# np.save('./results/B2Norms', Norms)
# np.save('./results/B2Losses', Losses)
# np.save('./results/B2Losses', Number_of_steps)
#


################## Data treatement ###################

Parameters = np.load('./results/B2Parameters.npy')
Norms = np.load('./results/B2Norms.npy')
Losses =  np.load('./results/B2Losses.npy')
Number_of_steps = np.load('./results/B2Losses.npy')


SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['text.usetex'] = True

X = np.linspace(0, 35, 36)
fig, axs = plt.subplots(1, len(M))
fig.suptitle('Difference between max loss direction and random direction')
fig.set_size_inches( 16.5,5.5)

for j in range(0, 2):
    for n in range(0, len(M)):
            Average = np.sum(abs(Parameters[n, :, Nres-1, j,:, :, :]),(0,1))
            print(Average.shape)
            Y = np.reshape(Average , (36, 1))
            if (j == 0):
                axs[n].plot(X, Y, label="Maximal loss direction")
            else:
                axs[n].plot(X, Y, label="Random direction")
            axs[n].set_xlabel(r'$ i+c\cdot j$')
            axs[n].set_ylabel(r'$<|C_{i,j}|>$')
            axs[n].legend()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
plt.show()
#
# data_n = [[],[]]
# for type in range(0,2):
#     print("New type")
#     for i in range(0,10):
#         N = np.count_nonzero(Norms[0, :,  Parameter_resolution - 1, type, i])
#         print(np.sum(Norms[0, :,  Parameter_resolution - 1, type, i])/N)
#         data_n[type] += [np.sum(Norms[0, :,  Parameter_resolution - 1, type, i])/N]
#
# X = np.linspace(0,9,10)
# fig, axs = plt.subplots(1, len(M))
# fig.suptitle('Difference between max loss direction and random direction')
# fig.set_size_inches( 16.5,5.5)
#
# labels = ["max loss direction","random direction"]
# colors = ['b','r']
# for n in range(0, len(M)):
#     for type in range(0,2):
#         axs[n].bar(X+ type*0.25, data_n[type], label=labels[type], color = colors[type], width = 0.25)
#         axs[n].set_xlabel(r'class (of CIFAR10)')
#         axs[n].set_ylabel(r'$<\|C\|_{f}>$')
#         axs[n].legend()
# plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
# plt.show()
#
