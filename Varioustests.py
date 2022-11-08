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

# #### VGG #####
PATH = "./model/vggbn_trained.pt"
Final_PATHS = "./model/vggbn_trained"
Model1 = Model_definitions_Leo.VGG("VGG11")
Model2 = Model_definitions_Leo.VGG("VGG11")
Model3 = Model_definitions_Leo.VGG("VGG11")
size_of_net = 11
type_of_net = 'VGG'
# #
# ### RESNET #####
# PATH = "./model/resnet18_trained.pt"
# Final_PATHS = "./model/resnet18_trained"
# Model1 = Model_definitions_Leo.ResNet18()
# Model2 = Model_definitions_Leo.ResNet18()
# Model3 = Model_definitions_Leo.ResNet18()
# size_of_net = 18
# type_of_net = 'Resnet'
M = [Model1,Model2,Model3]
checkpoint = torch.load(PATH)
index = 1

for c in checkpoint:
    temp = torch.load(Final_PATHS +str(index) +".pt")
    M[index-1].load_state_dict(temp)
    index += 1

M = [Model1]
################# Data Setup ##########

# #Type_of_System = ["Vectors","Stripe"]
# #Type_of_System = ["Vectors","NODVSM"]
Type_of_System = ["Images","CIFAR10"]

trainset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Train_Import_NP()

batch_size = 1
criterion = nn.CrossEntropyLoss()
#
trainloader = DataLoader(dataset = trainset, batch_size=batch_size, shuffle=False, num_workers=0)

cut_off = 6
temperature = 0.001
Number_of_correctly_labeled_images = 50 # Number of images for which we do our statistics on
Number_of_initial_conditions = 30
Number_of_Radii = 40
diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape

NON_ZERO_MATRIX = np.array([[1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,0.],[1.,1.,1.,1.,1.,0.],[1.,1.,1.,1.,0.,0.],[1.,1.,0.,0.,0.,0.]])
# delta = np.zeros((6,6))
# delta[0][0] = -1/2
ZERO_MATRIX = -1*NON_ZERO_MATRIX+ np.ones(diffeo_shape)

#### Arrays of information ######
Radii = np.linspace(0.00001,30,Number_of_Radii)
Parameters = np.zeros((3, Number_of_correctly_labeled_images,Number_of_Radii,Number_of_initial_conditions)+shape)

########## Testing ##########

for n_m, m in enumerate(M,0):
    Number_of_real_images = 0
    for n_d, data in enumerate(trainloader,0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = m(inputs)
        predicted = torch.max(outputs.data, 1).indices.to(device)
        li = criterion(outputs, labels)
        if(predicted==labels):
            Number_of_real_images += 1
            for n_r,r in enumerate(Radii,0):
                print(n_r)
                for n_i in range(0,Number_of_initial_conditions):

                    ### Generating the ball
                    p_temp = np.random.normal(0,1,28)
                    d = np.linalg.norm(p_temp)
                    p = r*p_temp/d
                    parameters = torch.zeros((1,6,6)).to(device)
                    N = 0
                    for i in range(0,cut_off):
                        for j in range(0,cut_off):
                            if (NON_ZERO_MATRIX[i,j]==1):
                                parameters[0,i,j] = p[N]
                                N += 1
                    parameters.requires_grad_(requires_grad=True)
                    Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type,dimension, index_of_discriminating_factor, diffeo)
                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    outputs = m(inputs_temp)
                    l = criterion(outputs, labels)
                    l.backward()
                    with torch.no_grad():
                       Parameters[n_m,Number_of_real_images-1,n_r,n_i,:,:,:] = parameters.grad
                    parameters.grad.zero_()
            if (Number_of_real_images >= Number_of_correctly_labeled_images):
                break

##### Showing and plotting ###########

    # for a in range(0, 10):
    #     for b in range(0, Parameter_resolution):
    #         Average = np.sum(Norms[n, :,b,a])/Number_of_correctly_labeled_images
    #         print('the average norm with resolution ', Gradient_ascent_step[b], 'for class', a, ' is', Average)
    #     print("")

########## Saving ##########

np.save('./results/B1Balls', Parameters)

################# Data treatement ###################

## Loading variables ####

Parameters = np.load('./results/B1Balls.npy')

#### Figure implementations ######

def number_marker(n):
    if (n//10 == 1):
        return 'th'
    else:
        if (n % 10 == 1):
            return 'st'
        if (n % 10 == 2):
            return 'nd'
        if (n % 10 == 3):
            return 'rd'
        return 'th'

#############  ##############

SMALL_SIZE = 20
MEDIUM_SIZE = 23
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['text.usetex'] = True

##### Fig 1 - Distribution of C_{ij} averaged on images for different N_iter #######

fig, axs = plt.subplots(1, len(M))
fig.suptitle(r'Distribution of the normalized $<\left[C^{(n_{f})}_{i,j}\right]^{2}>$ for '+ str(len(M)) +' '+ type_of_net + str(size_of_net) + r' Net(s) for different $ \alpha $.')
fig.set_size_inches( 16.5,5.5)

for n_m in range(0, len(M)):
    Y_temp = []
    for n_r in range(0, Number_of_Radii):
        Y = np.sum(np.square(Parameters[n_m,:,n_r,:,:,:,:]),(0,1))/(Number_of_correctly_labeled_images*Number_of_initial_conditions)
        m1 = np.sum(Y) / 28
        Y = Y / m1
        Y = Y - np.ones(diffeo_shape)
        Y = np.square(Y)
        Y_temp += [np.sum(Y) / 28]
    Y = np.array(Y_temp)
    if(len(M)==1):
        axs.plot(Radii,Y, label= ' TO DO')
    else:
        axs[n_m].plot(Radii,Y, label= ' TO DO')
    if(len(M)==1):
        axs.set_xlabel(r'TO DO')
        axs.set_ylabel(r'TO DO')
        axs.legend()
        axs.grid()
    else:
        axs[n_m].set_xlabel(r'TO DO')
        axs[n_m].set_ylabel(r'TO DO')
        axs[n_m].legend()
        axs[n_m].grid()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
plt.show()

##### Fig 2 -  #######

fig, axs = plt.subplots(1, len(M))
fig.suptitle(r'Distribution of the normalized $<\left[C^{(n_{f})}_{i,j}\right]^{2}>$ for '+ str(len(M)) +' '+ type_of_net + str(size_of_net) + r' Net(s) for different $ \alpha $.')
fig.set_size_inches( 16.5,5.5)

for n_m in range(0, len(M)):
    Y_temp = []
    for n_r in range(0, Number_of_Radii):
        Y = np.sum(np.square(Parameters[n_m,:,n_r,:,:,:,:]),(0,1))/(Number_of_correctly_labeled_images*Number_of_initial_conditions)
        Cmax = np.max(np.reshape(Y, 36))
        Cmin = 10*np.min(np.reshape(Y + ZERO_MATRIX * 1000000, 36))
        Y_temp += [Cmax / Cmin]
        print(Cmax, Cmin)
    Y = np.array(Y_temp)
    if(len(M)==1):
        axs.plot(Radii,Y, label= ' TO DO')
    else:
        axs[n_m].plot(Radii,Y, label= ' TO DO')
    if(len(M)==1):
        axs.set_xlabel(r'TO DO')
        axs.set_ylabel(r'TO DO')
        axs.legend()
        axs.grid()
    else:
        axs[n_m].set_xlabel(r'TO DO')
        axs[n_m].set_ylabel(r'TO DO')
        axs[n_m].legend()
        axs[n_m].grid()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
plt.show()
