import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import Model_definitions_Leo
import Data_treatement
import Transformations
import matplotlib.pyplot as plt
import Useful_functions

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
#
# ### RESNET #####
# PATH = "./model/resnet18_trained.pt"
# Final_PATHS = "./model/resnet18_trained"
# Model1 = Model_definitions_Leo.ResNet18()
# Model2 = Model_definitions_Leo.ResNet18()
# Model3 = Model_definitions_Leo.ResNet18()
# size_of_net = 18
# type_of_net = 'ResNet'

M = [Model1,Model2,Model3]

checkpoint = torch.load(PATH)
index = 1

for c in checkpoint:
    temp = torch.load(Final_PATHS +str(index) +".pt")
    M[index-1].load_state_dict(temp)
    index += 1

#M = [Model1]
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
temperature = 0.01
Parameter_resolution = 10 # in this case we are looking at the step_size
Number_of_correctly_labeled_images = 200 # Number of images for which we do our statistics on
Number_of_initial_conditions = 1
N_likelyhoods = 1

Lambda = 1
max_steps = [1]
#max_steps = np.linspace(201,401,26) ### a fixed step number instead of the while loop - useful for low norm effects
Number_of_max_steps = len(max_steps)

diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape

#Gradient_ascent_step = np.linspace(0.1,1, num=Parameter_resolution, endpoint=True, dtype=None, axis=0)
Gradient_ascent_step = np.logspace(-1,1, num=Parameter_resolution, endpoint=True, dtype=None, axis=0)

NON_ZERO_MATRIX = np.array([[1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,0.],[1.,1.,1.,1.,1.,0.],[1.,1.,1.,1.,0.,0.],[1.,1.,0.,0.,0.,0.]])
delta = np.zeros((6,6))
delta[0][0] = -1/2
ZERO_MATRIX = -1*NON_ZERO_MATRIX+ np.ones(diffeo_shape)
#print(NON_ZERO_MATRIX+delta)

#### Arrays of information ######

Parameters = np.zeros((3, Number_of_correctly_labeled_images, 10)+shape)
Norms = np.zeros((3, Number_of_correctly_labeled_images, 10))
Losses = np.zeros((3, Number_of_correctly_labeled_images, 10))
Number_of_steps = np.zeros((3, Number_of_correctly_labeled_images, 10))
#
#     ### With added noise ###
#     initial_conditions += [(10 ** (-6)*torch.rand(shape)+standard_initialization).to(device)]
# # #
########## Testing ##########
for n_m,m in enumerate(M,0):
    m.eval()
    Number_of_real_images = 0
    for n_d, data in enumerate(trainloader,0):
        inputs, labels = data
        #inputs, labels = inputs.to(device), labels.to(device)
        outputs = m(inputs)
        predicted = torch.max(outputs.data, 1).indices.to(device)
        # p = torch.max(outputs.data, 1)
        # li = criterion(outputs, labels)
        if(predicted==labels):
            Number_of_real_images += 1
            print(n_d,Number_of_real_images)
            index = 0
            parameters = torch.randn(shape)
            inputs_temp = inputs
            Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
            while ((predicted == labels)):
                Transf.parameters = torch.randn(shape)*torch.tensor(np.array([NON_ZERO_MATRIX]), dtype=torch.float32)
                #print(Transf.parameters)
                #inputs_temp = Transformations.deform_OLD(inputs,temperature,cut_off)
                inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                #inputs_temp, Norms_i = Transf.Noise(inputs,[10])
                #print(Norms_i)
                outputs = m(inputs_temp)

                predicted = torch.max(outputs.data, 1).indices.to(device)
                index += 1
            Useful_functions.imshow((inputs[0]))
            Useful_functions.imshow(inputs_temp[0])
            print(index)
            # Parameters[n_m][Number_of_real_images - 1][labels.item()] = ((Transf.parameters).detach().numpy())
            # Norms[n_m][Number_of_real_images - 1][labels.item()] = torch.norm((Transf.parameters)).item()
            Number_of_steps[n_m][Number_of_real_images - 1][labels.item()] = index
            if (Number_of_real_images >= Number_of_correctly_labeled_images):
                break

######### Saving ##########

np.save('./results/'+ type_of_net + str(size_of_net)+'/BasicParameters', Parameters)
np.save('./results/'+ type_of_net + str(size_of_net)+'/BasicNorms', Norms)
np.save('./results/'+ type_of_net + str(size_of_net)+'/BasicLosses', Losses)
np.save('./results/'+ type_of_net + str(size_of_net)+'/BasicNumber_of_steps', Number_of_steps)
############## Data treatement ###################

### Loading variables ####

Parameters = np.load('./results/'+ type_of_net + str(size_of_net)+'/BasicParameters.npy')
Norms = np.load('./results/'+ type_of_net + str(size_of_net)+'/BasicNorms.npy')
Losses =  np.load('./results/'+ type_of_net + str(size_of_net)+'/BasicLosses.npy')
Number_of_steps = np.load('./results/'+ type_of_net + str(size_of_net)+'/BasicNumber_of_steps.npy')

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


SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['text.usetex'] = True

##### Fig 1 - Average number of initializations per image #######


M = [1]
fig, axs = plt.figure() ,plt.axes()
fig.suptitle(r'$<n_{f}>(\alpha)$ for 3 different '+ type_of_net + str(size_of_net)  + r' networks')
fig.set_size_inches( 7.5,6.5)
for n_m in range(0, len(M)):
    Ytemp = np.sum(Number_of_steps[n_m,:,:],(1))
    N_of_bars = np.max(Ytemp)/10
    Ytemp[0] = 1
    print(Ytemp)
    X = np.linspace(0, np.max(Ytemp), N_of_bars)
    Y = np.ones(N_of_bars)
    for n_x in range(0,len(X)):
        print(X[n_x])
        if (n_x > 0):
            bool = (Ytemp <= X[n_x])*(Ytemp > X[n_x-1])
        else:
            bool = (Ytemp <= X[n_x])
        print(bool)
        print()
        Y[n_x] = (len(Ytemp[bool]))
    #print(X)
    axs.bar(X,Y, width = np.max(Ytemp)/(N_of_bars-1))
axs.set_xlabel(r'$i$')
axs.set_ylabel(r'$<n_{f}>$')
axs.legend()
axs.grid()
plt.show()

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
