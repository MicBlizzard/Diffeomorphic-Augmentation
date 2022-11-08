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

#
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
#

M = [Model2,Model3]
################# Data Setup ##########

Type_of_System = ["Images","CIFAR10"]

trainset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Train_Import_NP()

batch_size = 1
criterion = nn.CrossEntropyLoss()
#
trainloader = DataLoader(dataset = trainset, batch_size=batch_size, shuffle=False, num_workers=0)

cut_off = 6
temperature = 0.001
Parameter_resolution = 2 # in this case we are looking at the step_size
Number_of_correctly_labeled_images = 50 # Number of images for which we do our statistics on

diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape

Gradient_ascent_step = np.logspace(2,3, num=Parameter_resolution, endpoint=True, dtype=None, axis=0)

#print((3, Number_of_correctly_labeled_images, Parameter_resolution, 2, 10))
Parameters = np.zeros((3, Number_of_correctly_labeled_images, Parameter_resolution, 2, 10)+shape)
Norms = np.zeros((3,Number_of_correctly_labeled_images,Parameter_resolution,2,10))
Losses = np.zeros((3,Number_of_correctly_labeled_images,Parameter_resolution,2,10))
Number_of_steps = np.zeros((3,Number_of_correctly_labeled_images,Parameter_resolution,2,10))


########### Testing ##########

Type_of_test = "B2"
#
for n,m in enumerate(M,0):
    Number_of_real_images = 0
    m.eval()
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = m(inputs)
        predicted = torch.max(outputs.data, 1).indices.to(device)
        li = criterion(outputs, labels)
        if(predicted==labels):
            Number_of_real_images += 1
            print(Number_of_real_images)
            parameters = (10 **(-6)*torch.ones(shape)).to(device)
            parameters.requires_grad_(requires_grad= True)
            Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
            inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
            outputs = m(inputs_temp)
            l = criterion(outputs, labels)
            predicted = torch.max(outputs.data, 1).indices.to(device)
            l.backward()
            max_loss_direction_for_image_i = torch.tensor((parameters.grad)).to(device)
            temp_norm = torch.norm(max_loss_direction_for_image_i).item()
            non_zero = torch.abs(torch.sign(parameters.grad)).to(device)
            print(non_zero)
            random_direction_for_image_i = (torch.randn(shape)*non_zero).to(device)
            random_direction_for_image_i = random_direction_for_image_i*temp_norm/torch.norm(random_direction_for_image_i).item()
            parameters.grad.zero_()
            for n_a, a in enumerate(Gradient_ascent_step,0):
                delta_L = l-li
                for j in range(0,2):
                    index = 1
                    while ((predicted == labels)):
                        Transf.parameters = (10 ** (-6) * torch.ones(shape)).to(device)
                        inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                        outputs = m(inputs_temp)
                        predicted = torch.max(outputs.data, 1).indices.to(device)
                        l = criterion(outputs, labels)
                        #print(j)
                        if (j==0):
                            with torch.no_grad():
                                Transf.parameters = (index) * a * max_loss_direction_for_image_i
                        if (j==1):
                            with torch.no_grad():
                                Transf.parameters = (index) * a * random_direction_for_image_i
                                #print("in the j = 1 loop")
                        inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                        #print(Transf.parameters)
                        outputs = m(inputs_temp)
                        predicted = torch.max(outputs.data, 1).indices.to(device)
                        l = criterion(outputs, labels)
                        delta_L = l-li
                        index += 1
                        if (index%500==0):
                            print(torch.norm((Transf.parameters)).item())
                        if (torch.norm((Transf.parameters)).item() > 150):
                            print("faulty image", Number_of_real_images, n_a,"for a walk of type", j)
                            break
                    #print(torch.norm((Transf.parameters)).item(), delta_L.item(), index)
                    (Parameters[n][Number_of_real_images - 1][n_a][j][labels.item()]) = ((Transf.parameters).detach().numpy())
                    Norms[n, Number_of_real_images - 1, n_a,j, labels.item()] = torch.norm((Transf.parameters)).item()
                    Losses[n, Number_of_real_images - 1, n_a,j, labels.item()] = delta_L.item()
                    Number_of_steps[n, Number_of_real_images - 1, n_a,j, labels.item()] = index

                    Transf.parameters = (10 ** (-6) * torch.ones(shape)).to(device)
                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    outputs = m(inputs_temp)
                    predicted = torch.max(outputs.data, 1).indices.to(device)

            if (Number_of_real_images >= Number_of_correctly_labeled_images):
                break

###### Showing and plotting ###########

    for a in range(0, 10):
        for b in range(0, Parameter_resolution):
            for c in range(0,2):
                Average = np.sum(Norms[n, :,b,c,a])/Number_of_correctly_labeled_images
                print('the average norm with resolution ', Gradient_ascent_step[b], 'of the ', a, 'th class is when doing a type ',c,'walk is', Average)
            print("")


for j in range(0,2):
    X = np.linspace(0, 35, 36)
    fig, axs = plt.subplots(1, len(M))
    fig.suptitle('Stepsizes for different networks and walk of type '+str(j))
    for n in range(0, len(M)):
        for n_a in range(0, Parameter_resolution):

                Average = np.sum(abs(Parameters[n, :, n_a, j,:, :, :]),(0,1))
                print(Average.shape)
                Y = np.reshape(Average , (36, 1))
                axs[n].plot(X, Y, label="step size of " + str(Gradient_ascent_step[n_a]))
                axs[n].legend()
    plt.show()

########## Saving ##########

np.save('./results/B2Parameters', Parameters)
np.save('./results/B2Norms', Norms)
np.save('./results/B2Losses', Losses)
np.save('./results/B2NSteps', Number_of_steps)

################## Data treatement ###################

Parameters = np.load('./results/B2Parameters.npy')
Norms = np.load('./results/B2Norms.npy')
Losses =  np.load('./results/B2Losses.npy')
Number_of_steps = np.load('./results/B2NSteps.npy')

SMALL_SIZE = 19
MEDIUM_SIZE = 21
BIGGER_SIZE = 23

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.grid'] = True
Radi_list = []
for i in range(0,cut_off):
    for j in range(0,cut_off):
        if (np.sqrt((i+1)**2+(j+1)**2) <= cut_off +0.5 ):
            Radi_list += [np.sqrt((i+1)**2+(j+1)**2)]

True_radii = []
[True_radii.append(x) for x in Radi_list if x not in True_radii]
True_radii.sort()
l = len (True_radii)
Ynew = np.zeros(l)
k = torch.arange(1, cut_off + 1, device=device)
i, j = torch.meshgrid(k, k)
R = (i.pow(2) + j.pow(2)).sqrt()

#M = [1]

###### FIGURE 3 - Norms as a function of s #######

X = np.linspace(0, 35, 36)
fig, axs = plt.subplots(1, len(M))
fig.suptitle(r'$\left<n_{f}\right>(s)$ for $\beta_{0}$ and $\gamma_{0}$ Algorithms')
fig.set_size_inches(8,6.5)

type = ['VGG11','ResNet18']

for n in range(0, len(M)):
    for j in range(0, 2):
        Y = np.sum((Norms[n, :, :, j,:]), (0, 2)) / (Number_of_correctly_labeled_images)
        if (len(M) > 1):
            if (j == 0):
                axs[n].plot(Gradient_ascent_step, Y, label=r'${\varphi}^{(n_{f})} = \beta^{(n_{f})}_{0}$ '+type[n])
            else:
                axs[n].plot(Gradient_ascent_step, Y, label=r'$\varphi^{(n_{f})} = \gamma^{(n_{f})}_{0}$ '+type[n])
            axs[n].set_xlabel(r'$ s $')
            axs[n].set_ylabel(r'$\left<n_{f}\right>(s)$')
            axs[n].grid()
            axs[n].legend()
        else:
            if (j == 0):
                axs.plot(True_radii, Ynew, label=r'${\varphi}^{(n_{f})} = \beta^{(n_{f})}_{0}$')
            else:
                axs.plot(True_radii, Ynew, label=r'$\varphi^{(n_{f})} = \gamma^{(n_{f})}_{0}$')
            axs.set_xlabel(r'$ s $')
            axs.set_ylabel(r'$\left<n_{f}\right>(s)$')
            axs.grid()
            axs.legend()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.90 ,top=0.9, wspace=0.4, hspace=0.2)
plt.show()


###### FIGURE 4 - Distribution for random walk and maxloss direction #######

X = np.linspace(0, 35, 36)
fig, axs = plt.subplots(1, len(M))
fig.suptitle(r'$\left<n_{f}\right>(s)$ for $\beta_{0}$ and $\gamma_{0}$ Algorithms')
fig.set_size_inches(8,6.5)

type = ['VGG11','ResNet18']

for n in range(0, len(M)):
    for j in range(0, 2):
        Y = np.sum((Number_of_steps[n, :, :, j,:]), (0, 2)) / (Number_of_correctly_labeled_images)
        if (len(M) > 1):
            if (j == 0):
                axs[n].plot(Gradient_ascent_step, Y, label=r'${\varphi}^{(n_{f})} = \beta^{(n_{f})}_{0}$ '+type[n])
            else:
                axs[n].plot(Gradient_ascent_step, Y, label=r'$\varphi^{(n_{f})} = \gamma^{(n_{f})}_{0}$ '+type[n])
            axs[n].set_xlabel(r'$ s $')
            axs[n].set_ylabel(r'$\left<n_{f}\right>(s)$')
            axs[n].grid()
            axs[n].legend()
        else:
            if (j == 0):
                axs.plot(True_radii, Ynew, label=r'${\varphi}^{(n_{f})} = \beta^{(n_{f})}_{0}$')
            else:
                axs.plot(True_radii, Ynew, label=r'$\varphi^{(n_{f})} = \gamma^{(n_{f})}_{0}$')
            axs.set_xlabel(r'$ s $')
            axs.set_ylabel(r'$\left<n_{f}\right>(s)$')
            axs.grid()
            axs.legend()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.90 ,top=0.9, wspace=0.4, hspace=0.2)
plt.show()
#
# ###### FIGURE 5 - Distribution for random walk and maxloss direction #######
#
# for n in range(0, len(M)):
#     for j in range(0, 2):
#         Y = np.sum((Number_of_steps[n, :, :, j, :]), (0, 2)) / (Number_of_correctly_labeled_images)
#         Y = np.sum((Number_of_steps[n, :, :, j,:]), (0, 2)) / (Number_of_correctly_labeled_images)
#         if (len(M) > 1):
#             if (j == 0):
#                 axs[n].plot(Gradient_ascent_step, Y, label="Maximal loss direction")
#             else:
#                 axs[n].plot(Gradient_ascent_step, Y, label="Random direction")
#             axs[n].set_xlabel(r'$ r $')
#             axs[n].set_ylabel(r'$\left<\left[C^{(n_{f})}\right]^{2}\right>(r)$')
#             axs[n].grid()
#             axs[n].legend()
#         else:
#             if (j == 0):
#                 axs.plot(True_radii, Ynew, label=r'${\varphi}^{(n_{f})} = \beta^{(n_{f})}_{0}$')
#             else:
#                 axs.plot(True_radii, Ynew, label=r'$\varphi^{(n_{f})} = \gamma^{(n_{f})}_{0}$')
#             axs.set_xlabel(r'$ r $')
#             axs.set_ylabel(r'$\left<\left[\varphi^{(n_{f})}\right]^{2}\right>(r)$')
#             axs.grid()
#             axs.legend()
# plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85 ,top=0.9, wspace=0.4, hspace=0.4)
# plt.show()
#
#
# ##### FIG 2 #######
#
# X = np.linspace(0, 35, 36)
# fig, axs = plt.subplots(1, len(M))
# fig.suptitle(r'Distribution of $\left<\left[\varphi^{(n_{f})}\right]^{2}\right>(r)$ for $\beta_{0}$ and $\gamma_{0}$ algorithms with $s = 0.1$')
# fig.set_size_inches(8,6.5)
#
# print(len(M))
# for j in range(0, 2):
#     for n in range(0, len(M)):
#         Y = np.sum((Parameters[n, :, 0, j,:, :, :]), (0, 1)) / (Number_of_correctly_labeled_images)
#         #Y = np.reshape(Y, 36)
#         Ynew = np.zeros(l)
#         for n_r, r in enumerate(True_radii, 0):
#             BOOL = (R == r)
#             #print(BOOL)
#             Ynew[n_r] = np.sum(Y[0][BOOL])/len(Y[0][BOOL])
#
#         if (len(M) > 1):
#             if (j == 0):
#                 axs[n].plot(True_radii, Ynew, label="Maximal loss direction")
#             else:
#                 axs[n].plot(True_radii, Ynew, label="Random direction")
#             axs[n].set_xlabel(r'$ r $')
#             axs[n].set_ylabel(r'$\left<\left[C^{(n_{f})}\right]^{2}\right>(r)$')
#             axs[n].grid()
#             axs[n].legend()
#         else:
#             if (j == 0):
#                 axs.plot(True_radii, Ynew, label=r'${\varphi}^{(n_{f})} = \beta^{(n_{f})}_{0}$')
#             else:
#                 axs.plot(True_radii, Ynew, label=r'$\varphi^{(n_{f})} = \gamma^{(n_{f})}_{0}$')
#             axs.set_xlabel(r'$ r $')
#             axs.set_ylabel(r'$\left<\left[\varphi^{(n_{f})}\right]^{2}\right>(r)$')
#             axs.grid()
#             axs.legend()
# plt.subplots_adjust(left=0.15, bottom=0.1, right=0.85 ,top=0.9, wspace=0.4, hspace=0.4)
# plt.show()
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

# #
# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# ax.bar(X + 0.00, data_n[0], color = 'b', width = 0.25)
# ax.bar(X + 0.25, data_n[1], color = 'g', width = 0.25)
# plt.show()