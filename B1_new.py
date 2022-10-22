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

type_of_net = 'VGG'
type_of_net = 'Resnet'

# #### VGG #####
PATH = "./model/vggbn_trained.pt"
Final_PATHS = "./model/vggbn_trained"
Model1 = Model_definitions_Leo.VGG("VGG11")
Model2 = Model_definitions_Leo.VGG("VGG11")
Model3 = Model_definitions_Leo.VGG("VGG11")
size_of_net = 11
#
# ### RESNET #####
# PATH = "./model/resnet18_trained.pt"
# Final_PATHS = "./model/resnet18_trained"
# Model1 = Model_definitions_Leo.ResNet18()
# Model2 = Model_definitions_Leo.ResNet18()
# Model3 = Model_definitions_Leo.ResNet18()
# size_of_net = 18

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
temperature = 0.001
Parameter_resolution = 9 # in this case we are looking at the step_size
Number_of_correctly_labeled_images = 50 # Number of images for which we do our statistics on
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

#### Arrays of information ######


Likelyhoods = np.linspace(0.5,1,N_likelyhoods)
Likelyhood_rates = np.zeros((3,N_likelyhoods,Parameter_resolution,Number_of_initial_conditions,10))
Parameters = np.zeros((3, Number_of_correctly_labeled_images, Number_of_max_steps,Parameter_resolution,Number_of_initial_conditions, 10)+shape)
Norms = np.zeros((3,Number_of_correctly_labeled_images,Number_of_max_steps,Parameter_resolution,Number_of_initial_conditions,10))
Losses = np.zeros((3,Number_of_correctly_labeled_images,Number_of_max_steps,Parameter_resolution,Number_of_initial_conditions,10))
Number_of_steps = np.zeros((3,Number_of_correctly_labeled_images,Number_of_max_steps,Parameter_resolution,Number_of_initial_conditions,10))

NON_ZERO_MATRIX = np.array([[1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.,0.],[1.,1.,1.,1.,1.,0.],[1.,1.,1.,1.,0.,0.],[1.,1.,0.,0.,0.,0.]])
delta = np.zeros((6,6))
delta[0][0] = -1/2
ZERO_MATRIX = -1*NON_ZERO_MATRIX+ np.ones(diffeo_shape)
print(NON_ZERO_MATRIX+delta)

#### Initializing initial conditions ####
initial_conditions = []
initial_norms = []
#initial_conditions += [(10 ** (-6)*torch.zeros(shape)).to(device)]
for i in range(0,Number_of_initial_conditions):
    #initial_conditions += [(10 ** (-6)*torch.rand(shape)+(10 ** (-6) * torch.ones(shape))).to(device)]
    initial_conditions += [(10 ** (-6) * torch.ones(shape)).to(device)]
    #initial_conditions += [(torch.tensor(np.array([NON_ZERO_MATRIX+delta]), dtype=torch.float32)).to(device)]
    #initial_norms +=[torch.norm(initial_conditions[i]).item()]
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
            print(Number_of_real_images)
            for n_a, a in enumerate(Gradient_ascent_step,0):
                for n_i, i in enumerate(initial_conditions,0):
                    parameters = torch.clone(i)
                    parameters.requires_grad_(requires_grad=True)
                    Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    outputs = m(inputs_temp)
                    l = criterion(outputs, labels)
                    predicted = torch.max(outputs.data, 1).indices.to(device)
                    index = 1
                    delta_L = l-li
                    index_max_step = 0
                    #for iter in range(0,int(max_steps[-1])):
                    while ((predicted == labels)):
                        inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                        outputs = m(inputs_temp)
                        predicted = torch.max(outputs.data, 1).indices.to(device)
                        l = criterion(outputs, labels)
                        l.backward()
                        with torch.no_grad():
                            Transf.parameters += a * parameters.grad
                            #print(Transf.parameters)
                                ##### escape methods

                                # parameters += min(index,10)*a * (parameters.grad)
                                # parameters += np.sqrt(index)*a * (parameters.grad)
                                # parameters += (index) * a * (parameters.grad)
                                # Momentum
                                # Transf.parameters += a*(Lambda * parameters.grad +(1-Lambda)*old_grad)
                                # old_grad = parameters.grad

                        parameters.grad.zero_()
                        inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                        outputs = m(inputs_temp)
                        predicted = torch.max(outputs.data, 1).indices.to(device)
                        l = criterion(outputs, labels)
                        delta_L = l-li
                        index += 1
                            # if (index%20==0):
                            #     print(torch.norm((Transf.parameters)).item())

                        if (torch.norm((parameters)).item() > 1500) or (index > 300/a):  # arbitrary limit...
                            print("faulty image", Number_of_real_images, n_a)
                            break
                        # if (iter == int(max_steps[index_max_step]-1)): # [SPECIFIC TO ITER INVESTIGATIONS]
                        #     (Parameters[n_m][Number_of_real_images - 1][index_max_step][n_a][n_i][labels.item()]) = ((Transf.parameters).detach().numpy())
                        #     index_max_step+= 1
                    (Parameters[n_m][Number_of_real_images - 1][index_max_step][n_a][n_i][labels.item()]) = ((Transf.parameters).detach().numpy())
                    # Norms[n_m, Number_of_real_images - 1,n_N ,n_a,n_i, labels.item()] = torch.norm((Transf.parameters)).item()
                    # Losses[n_m, Number_of_real_images - 1,n_N, n_a,n_i, labels.item()] = delta_L.item()
                    # Number_of_steps[n_m, Number_of_real_images - 1,n_N, n_a,n_i, labels.item()] = index
                    for n_p, p in enumerate(Likelyhoods,0):
                        print(torch.norm((Transf.parameters)).item())
                        if (np.exp(-torch.norm((Transf.parameters)).item()**2) >= p):
                            Likelyhood_rates[n_m][n_p][n_a][n_i][labels.item()] += 1
            if (Number_of_real_images >= Number_of_correctly_labeled_images):
                break

##### Showing and plotting ###########

    # for a in range(0, 10):
    #     for b in range(0, Parameter_resolution):
    #         Average = np.sum(Norms[n, :,b,a])/Number_of_correctly_labeled_images
    #         print('the average norm with resolution ', Gradient_ascent_step[b], 'for class', a, ' is', Average)
    #     print("")

########## Saving ##########

for i in range(0,Number_of_initial_conditions):
    initial_conditions[i] = initial_conditions[i].detach().numpy()

np.save('./results/B1Parameters', Parameters)
np.save('./results/B1Norms', Norms)
np.save('./results/B1Losses', Losses)
np.save('./results/B1Number_of_steps', Number_of_steps)
np.save('./results/B1initial_conditions',np.array(initial_conditions))
np.save('./results/B1initial_norms',np.array(initial_norms))
np.save('./results/B1Likelyhood_rates',Likelyhood_rates)
################# Data treatement ###################

### Loading variables ####

Parameters = np.load('./results/B1Parameters.npy')
Norms = np.load('./results/B1Norms.npy')
Losses =  np.load('./results/B1Losses.npy')
Number_of_steps = np.load('./results/B1Number_of_steps.npy')
initial_conditions =  np.load('./results/B1initial_conditions.npy')
initial_norms = np.load('./results/B1initial_norms.npy')
Likelyhood_rates = np.load('./results/B1Likelyhood_rates.npy')

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

##### Fig 1 - Distribution of C_{ij} averaged on images for different N_iter #######

X = np.linspace(0, 35, 36)
fig, axs = plt.subplots(1, len(M))
fig.suptitle(r'Distribution of $\left[C^{(n_{f})}_{i,j}\right]^{2}$ for '+ str(len(M)) +' '+ type_of_net + str(size_of_net) + r' Net(s) for different $ \alpha $ and $n_{f}$')
fig.set_size_inches( 16.5,5.5)

for n_m in range(0, len(M)):
    for n_a in range(0,Parameter_resolution):
        for n_i in range(0,Number_of_initial_conditions):
            for n_N in range(0,Number_of_max_steps):
                #Y = np.sum(abs(Parameters[n_m, :, n_N, n_a, n_i, :, :, :, :]), (0, 1, 2)) / (10 * Number_of_correctly_labeled_images)
                Y = np.sum(np.square(Parameters[n_m,:,n_N,n_a,n_i,:,:,:,:]),(0,1,2))/(Number_of_correctly_labeled_images)
                #m1 = np.sum(Y)
                #Y = Y/m1
                Y = np.reshape(Y,36)
                if(len(M)==1):
                    print()
                    axs.plot(X, Y, label= 'walk of '+str(max_steps[n_N])+ r' steps and with $\alpha = $'+str(Gradient_ascent_step[n_a]))
                    #axs.plot(X, Y, label=str(n_i + 1) + number_marker(n_i + 1) + ' init. cond. of norm ' + str(float('%.2g' % initial_norms[n_i])) + ' ' + str(n_N))
                else:
                    axs[n_m].plot(X, Y, label ='walk of '+str(max_steps[n_N])+ r' steps and with $\alpha = $'+str(Gradient_ascent_step[n_a]))
                #axs[n_m].plot(X, Y, label=(r'$<|C_{i,j}|> $ averaged on ' + str(Number_of_correctly_labeled_images)+r'with a step size of '+str(Gradient_ascent_step[n_a])))
    if(len(M)==1):
        axs.set_xlabel(r'$ i+c\cdot j$')
        axs.set_ylabel(r'$<\left[C^{(n_{f})}_{i,j}\right]^2>$')
        axs.legend()
        axs.grid()
    else:
        axs[n_m].set_xlabel(r'$ i+c\cdot j$')
        axs[n_m].set_ylabel(r'$<\left[C^{(n_{f})}_{i,j}\right]^2>$')
        axs[n_m].legend()
        axs[n_m].grid()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
plt.show()

#### Fig 2 - delta C ratio as a function of alpha final #######

fig, axs = plt.subplots(1, len(M))
fig.suptitle(r'Distribution of $\delta C^{(n_{f})}$ for '+ str(len(M)) +' '+ type_of_net + str(size_of_net) + r' Net(s) for different $ \alpha $')
fig.set_size_inches( 16.5,5.5)
for n_m in range(0, len(M)):
    for n_i in range(0,Number_of_initial_conditions):
        Y_temp = []
        for n_a in range(0,Parameter_resolution):
            Y = np.sum(np.square(Parameters[n_m, :, 0, n_a, n_i, :, :, :, :]), (0,1,2))/(Number_of_correctly_labeled_images)
            m1 = np.sum(Y)/28
            Y = Y/m1
            Y = Y - np.ones(diffeo_shape)
            Y = np.square(Y)
            Y_temp += [np.sum(Y) / 28]
        Y = np.array(Y_temp)
        if (len(M)==1):
            axs.plot(Gradient_ascent_step, Y,label = r'$\alpha = $' + str(Gradient_ascent_step[n_a]))
        else:
            axs[n_m].plot(np.array(max_steps), Y,label = r'$\alpha = $' + str(Gradient_ascent_step[n_a]))
            Y_temp = []
    if (len(M)==1):
        axs.set_xlabel(r'$n_{f}$')
        axs.set_ylabel(r'$\delta C $')
        axs.legend()
        axs.grid()
    else:
        axs[n_m].set_xlabel(r'$n_{f}$')
        axs[n_m].set_ylabel(r'$\delta C $')
        axs[n_m].legend()
        axs[n_m].grid()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
plt.show()

#### Fig 3 - flatness ratio as a function of alpha final #######

fig, axs = plt.subplots(1, len(M))
fig.suptitle(r'Distribution of $\delta C^{(n_{f})}$ for '+ str(len(M)) +' '+ type_of_net + str(size_of_net) + r' Net(s) for different $ \alpha $')
fig.set_size_inches( 16.5,5.5)
for n_m in range(0, len(M)):
    for n_i in range(0,Number_of_initial_conditions):
        Y_temp = []
        for n_a in range(0,Parameter_resolution):
            Y = np.sum(np.square(Parameters[n_m, :, 0, n_a, n_i, :, :, :, :]), (0,1,2))/(Number_of_correctly_labeled_images)
            Cmax = np.max(np.reshape(Y,36))
            Cmin = np.min(np.reshape(Y+ZERO_MATRIX*1000000, 36))
            Y_temp += [Cmax/Cmin]
            print(Cmax, Cmin)
        Y = np.array(Y_temp)
        if (len(M)==1):
            axs.plot(Gradient_ascent_step, Y,label = r'$\alpha = $' + str(Gradient_ascent_step[n_a]))
        else:
            axs[n_m].plot(np.array(max_steps), Y,label = r'$\alpha = $' + str(Gradient_ascent_step[n_a]))
            Y_temp = []
    if (len(M)==1):
        axs.set_xlabel(r'$n_{f}$')
        axs.set_ylabel(r'$\delta C $')
        axs.legend()
        axs.grid()
    else:
        axs[n_m].set_xlabel(r'$n_{f}$')
        axs[n_m].set_ylabel(r'$\delta C $')
        axs[n_m].legend()
        axs[n_m].grid()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
plt.show()

#
# #### Fig 4 - delta C ratio as a function of N_iter final #######
#
# X = np.linspace(0, 35, 36)
# fig, axs = plt.subplots(1, len(M))
# fig.suptitle(r'Distribution of $\delta C^{(n_{f})}$ for '+ str(len(M)) +' '+ type_of_net + str(size_of_net) + r' Net(s) for different $ \alpha $')
# fig.set_size_inches( 16.5,5.5)
# for n_m in range(0, len(M)):
#     for n_a in range(0,Parameter_resolution):
#         for n_i in range(0,Number_of_initial_conditions):
#             Y_temp = []
#             for i in range(0,Number_of_max_steps):
#                 Y = np.sum(np.square(Parameters[n_m, :, i, n_a, n_i, :, :, :, :]), (0, 1, 2))/(Number_of_correctly_labeled_images)
#                 m1 = np.sum(Y)/28
#                 Y = Y/m1
#                 Y = Y - np.ones(diffeo_shape)
#                 Y = np.square(Y)
#                 Y_temp +=[np.sum(Y)/28]
#             Y = np.array(Y_temp)
#             if (len(M)==1):
#                 axs.plot(np.array(max_steps), Y,label = r'$\alpha = $' + str(Gradient_ascent_step[n_a]))
#             else:
#                 axs[n_m].plot(np.array(max_steps), Y,label = r'$\alpha = $' + str(Gradient_ascent_step[n_a]))
#     if (len(M)==1):
#         axs.set_xlabel(r'$n_{f}$')
#         axs.set_ylabel(r'$\delta C $')
#         axs.legend()
#         axs.grid()
#     else:
#         axs[n_m].set_xlabel(r'$n_{f}$')
#         axs[n_m].set_ylabel(r'$\delta C $')
#         axs[n_m].legend()
#         axs[n_m].grid()
# plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
# plt.show()

# ##### Fig 3 - Final norm as a function of N_iter #######
#
# X = np.linspace(0, 35, 36)
# fig, axs = plt.subplots(1, len(M))
# #fig.suptitle(r'Distribution of $C^{(t_{f})}_{i,j} for 3'+ type_of_net + str(size_of_net) + r' Nets with various initial conditions $C^{(0)}$ and step sizes, $ \alpha $')
# fig.set_size_inches( 16.5,5.5)
# for n_m in range(0, len(M)):
#     for n_a in range(0,Parameter_resolution):
#         for n_i in range(0,Number_of_initial_conditions):
#             Y = np.sum(np.linalg.norm((Parameters[n_m, :, :, n_a, n_i, :, :, :, :]),ord = 'fro', axis = (4,5)),(0,2,3))/(10*Number_of_correctly_labeled_images)
#                 #Y = np.sum(abs(Parameters[n_m,:,:,n_a,n_i,:,:,0,0]),(0,2,3))/np.sum(abs(Parameters[n_m,:,:,n_a,n_i,:,:,5,1]),(0,2,3))
#             if(len(M)==1):
#                 axs.plot(np.array(max_steps), Y*Y,label = r'$\alpha = $' + str(Gradient_ascent_step[n_a]))
#             else:
#                 axs[n_m].plot(np.array(max_steps), Y*Y, label = r'$\alpha = $' + str(Gradient_ascent_step[n_a]))
#                 #axs[n_m].plot(X, Y, label=(r'$<|C_{i,j}|> $ averaged on ' + str(Number_of_correctly_labeled_images)+r'with a step size of '+str(Gradient_ascent_step[n_a])))
#     if (len(M) == 1):
#         axs.set_xlabel(r'$n_{f}$')
#         axs.set_ylabel(r'$<\|C^{(n_{f})}_{i,j}\|_{f}>$')
#         axs.legend()
#         axs.grid()
#     else:
#         axs[n_m].set_xlabel(r'$n_{f}$')
#         axs[n_m].set_ylabel(r'$<\|C^{(n_{f})}_{i,j}\|_{f}>$')
#         axs[n_m].legend()
#         axs[n_m].grid()
# plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
# plt.show()

# else:
#     fig, axs = plt.subplots(1, len(M))
#     # fig.suptitle(r'Distribution of $C^{(t_{f})}_{i,j} for 3'+ type_of_net + str(size_of_net) + r' Nets with various initial conditions $C^{(0)}$ and step sizes, $ \alpha $')
#     fig.set_size_inches(16.5, 5.5)
#     for n_m in range(0, len(M)):
#         for n_a in range(0, Parameter_resolution):
#             for n_i in range(0, Number_of_initial_conditions):
#                 Y = np.sum(np.linalg.norm((Parameters[n_m, :, :, n_a, n_i, :, :, :, :]), ord='fro', axis=(4, 5)),
#                            (0, 2, 3)) / (10 * Number_of_correctly_labeled_images)
#                 # Y = np.sum(abs(Parameters[n_m,:,:,n_a,n_i,:,:,0,0]),(0,2,3))/np.sum(abs(Parameters[n_m,:,:,n_a,n_i,:,:,5,1]),(0,2,3))
#                 axs[n_m].plot(np.array(max_steps), Y,
#                               label=str(n_i + 1) + number_marker(n_i + 1) + ' init. cond. of norm ' + str(
#                                   float('%.2g' % initial_norms[n_i])))
#                 # axs[n_m].plot(X, Y, label=(r'$<|C_{i,j}|> $ averaged on ' + str(Number_of_correctly_labeled_images)+r'with a step size of '+str(Gradient_ascent_step[n_a])))
#         axs[n_m].set_xlabel(r'$ i+c\cdot j$')
#         axs[n_m].set_ylabel(r'$<|C_{i,j}|>$')
#         axs[n_m].legend()
#         axs[n_m].grid()
#     plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.4)
#     plt.show()
#
# data_n = [[],[]]
# for type in range(0,2):
#     print("New type")
#     for i in range(0,10):
#         N = np.count_nonzero(Norms[0, :,  Parameter_resolution - 1, type, i])
#         print(np.sum(Norms[0, :,  Parameter_resolution - 1, type, i])/N)
#         data_n[type] += [np.sum(Norms[0, :,  Parameter_resolution - 1, type, i])/N]
#

##### Fig 4 - Likelyhood plots #######

fig, axs = plt.subplots(1, len(M))
fig.suptitle(r'$f_{\alpha}(p)$} '+ str(len(M)) +' '+ type_of_net + str(size_of_net) + r' Net(s) for different $ \alpha $ and $n_{f}$')
fig.set_size_inches( 16.5,5.5)

for n_m in range(0, len(M)):
    for n_a in range(0,Parameter_resolution):
        for n_i in range(0,Number_of_initial_conditions):
            Y = np.sum(Likelyhood_rates[n_m,:,n_a,n_i,:],1)/(Number_of_correctly_labeled_images)
            print(Y)
            if(len(M)==1):
                print()
                axs.plot(Likelyhoods, Y, label= r'$f_{\alpha}(p)\quad \alpha = $}'+str(Gradient_ascent_step[Parameter_resolution]))
                #axs.plot(X, Y, label=str(n_i + 1) + number_marker(n_i + 1) + ' init. cond. of norm ' + str(float('%.2g' % initial_norms[n_i])) + ' ' + str(n_N))
            else:
                axs[n_m].plot(Likelyhoods, Y, label ='')
            #axs[n_m].plot(X, Y, label=(r'$<|C_{i,j}|> $ averaged on ' + str(Number_of_correctly_labeled_images)+r'with a step size of '+str(Gradient_ascent_step[n_a])))
    if(len(M)==1):
        axs.set_xlabel(r'$ i+c\cdot j$')
        axs.set_ylabel(r'$<\left[C^{(n_{f})}_{i,j}\right]^2>$')
        axs.legend()
        axs.grid()
    else:
        axs[n_m].set_xlabel(r'$ i+c\cdot j$')
        axs[n_m].set_ylabel(r'$<\left[C^{(n_{f})}_{i,j}\right]^2>$')
        axs[n_m].legend()
        axs[n_m].grid()
plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95 ,top=0.9, wspace=0.4, hspace=0.4)
plt.show()



# ##### Fig 3 - #######
#
# X = np.linspace(0,9,10)
# fig, axs = plt.subplots(1, len(M))
# fig.suptitle('Difference between max loss direction and random direction')
# fig.set_size_inches( 16.5,5.5)
#
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
