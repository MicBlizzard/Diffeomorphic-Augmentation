import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import Model_definitions_Leo
import Data_treatement
import Transformations
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############ Model(s) setup ###########

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

################# Data Setup ##########

Type_of_System = ["Images","CIFAR10"]

trainset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Train_Import_NP()

batch_size = 1
criterion = nn.CrossEntropyLoss()
#
trainloader = DataLoader(dataset = trainset, batch_size=batch_size, shuffle=False, num_workers=0)

cut_off = 6
temperature = 0.001
Parameter_resolution = 1 # in this case we are looking at the step_size
Number_of_correctly_labeled_images = 1 # Number of images for which we do our statistics on
Number_of_initial_conditions = 1

diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape

Gradient_ascent_step = np.linspace(0.3,0.4,10)
#Gradient_ascent_step = np.linspace(0.5,10, num=Parameter_resolution, endpoint=True, dtype=None, axis=0)
#Gradient_ascent_step = np.logspace(0,2, num=Parameter_resolution, endpoint=True, dtype=None, axis=0)

#print((3, Number_of_correctly_labeled_images, Parameter_resolution, 2, 10))
Parameters = []

for i in range(0,3):
    Para_temp = []
    for N_i in range(0,Number_of_initial_conditions):
        Para_temp_temp = []
        for j in range(0,len(Gradient_ascent_step)):
            Para_temp_temp += [np.zeros(shape)]
        Para_temp += [Para_temp_temp]
    Parameters +=[Para_temp]
print(Parameters)

Norms = np.zeros((3,Number_of_correctly_labeled_images,Parameter_resolution,10))
Losses = np.zeros((3,Number_of_correctly_labeled_images,Parameter_resolution,10))
Number_of_steps = np.zeros((3,Number_of_correctly_labeled_images,Parameter_resolution,10))

########### Testing ##########

for n,m in enumerate(M,0):
    Number_of_real_images = 0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = m(inputs)
        predicted = torch.max(outputs.data, 1).indices.to(device)
        li = criterion(outputs, labels)
        if(predicted==labels):
            Number_of_real_images += 1
            print(Number_of_real_images)
            for n_a, a in enumerate(Gradient_ascent_step,0):
                for n_i in range(0,Number_of_initial_conditions):
                    parameters = (10 ** (-6) * torch.ones(shape)).to(device)
                    #delta = (10 ** (-9) * torch.randn(shape)).to(device)
                    #parameters += delta
                    parameters.requires_grad_(requires_grad=True)
                    Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                    B  = np.zeros(shape)
                    Parameters[n][n_i][n_a] = np.concatenate((Parameters[n][n_i][n_a],Transf.parameters.detach().numpy()))
                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    outputs = m(inputs_temp)
                    l = criterion(outputs, labels)
                    predicted = torch.max(outputs.data, 1).indices.to(device)
                    index = 1
                    delta_L = l-li
                    while ((predicted == labels)):
                        #print(index)
                        inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                        outputs = m(inputs_temp)
                        predicted = torch.max(outputs.data, 1).indices.to(device)
                        l = criterion(outputs, labels)
                        l.backward()
                        with torch.no_grad():
                            Transf.parameters += a * parameters.grad
                            Parameters[n][n_i][n_a] = np.concatenate((Parameters[n][n_i][n_a],Transf.parameters.detach().numpy()))
                            #Parameters = np.concatenate(Parameters, np.array([Transf.parameters.detach().numpy()]))

                        parameters.grad.zero_()
                        inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                        outputs = m(inputs_temp)
                        predicted = torch.max(outputs.data, 1).indices.to(device)
                        l = criterion(outputs, labels)
                        delta_L = l-li
                        index += 1
                        print(index)
                        if (torch.norm((parameters)).item() > 1500) or (index > 300/a):  # arbitrary limit...
                            print("faulty image", Number_of_real_images, n_a)
                            break

                    #print(torch.norm((Transf.parameters)).item(), delta_L.item(), index)
                    print((Transf.parameters).detach().numpy())
                    #(Parameters[n][Number_of_real_images - 1][n_a][labels.item()]) = ((Transf.parameters).detach().numpy())
                    # Norms[n, Number_of_real_images - 1, n_a, labels.item()] = torch.norm((Transf.parameters)).item()
                    # Losses[n, Number_of_real_images - 1, n_a, labels.item()] = delta_L.item()
                    # Number_of_steps[n, Number_of_real_images - 1, n_a, labels.item()] = index

            if (Number_of_real_images >= Number_of_correctly_labeled_images):
                break

###### Showing and plotting ###########
#     for a in range(0, 10):
#         for b in range(0, Parameter_resolution):
#             Average = np.sum(Norms[n, :,b,a])/Number_of_correctly_labeled_images
#             print('the average norm with resolution ', Gradient_ascent_step[b], 'for class', a, ' is', Average)
#         print("")
#
# ########## Saving ##########
#
#
# np.save('./results/B1Parameters', Parameters)
# np.save('./results/B1Norms', Norms)
# np.save('./results/B1Losses', Losses)
# np.save('./results/B1Number_of_steps', Number_of_steps)
#
#
# ################## Data treatement ###################
#
# Parameters = np.load('./results/B1Parameters.npy')
# Norms = np.load('./results/B1Norms.npy')
# Losses =  np.load('./results/B1Losses.npy')
# Number_of_steps = np.load('./results/B1Number_of_steps.npy')
#
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

# X = np.linspace(0, 35, 36)
fig, axs = plt.subplots(1, len(M))
fig.suptitle('Norms of generated diffeos for different step sizes')
fig.set_size_inches( 16.5,5.5)

for i in range(1,2):
    for j in range(0, 1):
        for n_a, a in enumerate(Gradient_ascent_step,0):
             for n_i in range(0,Number_of_initial_conditions):
                for n in range(0, len(M)):
                    Y = Parameters[n][n_i][n_a][:,i,j]
                    X = np.linspace(0,len(Y),len(Y))
                    axs[n].plot(X, Y, label=r'$<\|C\|_{f}>(\alpha)$ for image '+str(i))
                    axs[n].set_xlabel(r'$ \alpha $')
                    axs[n].set_ylabel(r'$<\|C\|_{f}>$')
                    axs[n].legend()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
plt.show()

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
# fig.suptitle('Difference between max lossparameters += (index) * a * (parameters.grad) direction and random direction')
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
