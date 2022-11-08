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
#
### RESNET #####
PATH = "./model/resnet18_trained.pt"
Final_PATHS = "./model/resnet18_trained"
Model1 = Model_definitions_Leo.ResNet18()
Model2 = Model_definitions_Leo.ResNet18()
Model3 = Model_definitions_Leo.ResNet18()
size_of_net = 18
type_of_net = 'Resnet'

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
print(NON_ZERO_MATRIX+delta)

type_of_net = 'VGG'
size_of_net = 11

ParametersVGG = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Parameters.npy')
NormsVGG = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Norms.npy')
LossesVGG =  np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Losses.npy')
Number_of_stepsVGG = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Number_of_steps.npy')
initial_conditionsVGG =  np.load('./results/'+ type_of_net + str(size_of_net)+'/B1initial_conditions.npy')
initial_normsVGG = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1initial_norms.npy')
Likelyhood_ratesVGG = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Likelyhood_rates.npy')

type_of_net = 'ResNet'
size_of_net = 18

ParametersRes = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Parameters.npy')
NormsRes = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Norms.npy')
LossesRes =  np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Losses.npy')
Number_of_stepsRes = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Number_of_steps.npy')
initial_conditionsRes =  np.load('./results/'+ type_of_net + str(size_of_net)+'/B1initial_conditions.npy')
initial_normsRes = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1initial_norms.npy')
Likelyhood_ratesRes = np.load('./results/'+ type_of_net + str(size_of_net)+'/B1Likelyhood_rates.npy')

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

#################### SUBFIGURE PLOTS ####################

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


################## MULTIPLE MODELS ON ONE FIGURE ############


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


################## MULTIPLE NETWORKS AND AVERAGE OF MODELS ############


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

### Fig 1 - DC ratio as a function of alpha VGG vs Resnet #######

fig, axs = plt.figure() ,plt.axes()
fig.suptitle(r'$<\Delta C(\alpha)>$ for both ResNet18 and VGG11' )
fig.set_size_inches( 7.5,6.5)
for n_i in range(0,Number_of_initial_conditions):
    Y_tempVGG = []
    Y_tempRes = []
    for n_a in range(0,Parameter_resolution):

        YVGG = np.sum(np.square(ParametersVGG[:, :, 0, n_a, n_i, :, :, :, :]), (0,1,2,3))/(Number_of_correctly_labeled_images*3)
        m1VGG = np.sum(YVGG)/28
        YVGG = YVGG - np.ones(diffeo_shape)
        YVGG = np.square(YVGG)
        Y_tempVGG += [np.sum(YVGG) / 28]

        YRes = np.sum(np.square(ParametersRes[:, :, 0, n_a, n_i, :, :, :, :]), (0, 1, 2, 3)) / (Number_of_correctly_labeled_images * 3)
        m1Res = np.sum(YRes) / 28
        YRes = YRes - np.ones(diffeo_shape)
        YRes = np.square(YRes)
        Y_tempRes += [np.sum(YRes) / 28]
    YVGG = np.array(Y_tempVGG)
    YRes = np.array(Y_tempRes)
    axs.plot(Gradient_ascent_step, YVGG, color=(43/255,186/255,186/255),linewidth = 2.5, linestyle='dashed',label='VGG11')
    axs.plot(Gradient_ascent_step,YRes, color=(164/255,101/255,246/255), linewidth = 2.5,linestyle='dashed', label ='ResNet18')
axs.set_xlabel(r'$\alpha$')
axs.set_ylabel(r'$\Delta C $')
axs.legend()
axs.grid()
plt.show()


# fig, axs = plt.figure() ,plt.axes()
# fig.suptitle(r'$\Delta C(\alpha)$ for 3 different '+ type_of_net + str(size_of_net)  + r' networks')
# fig.set_size_inches( 7.5,6.5)
# for n_m in range(0, len(M)):
#     for n_i in range(0,Number_of_initial_conditions):
#         Y_temp = []
#         for n_a in range(0,Parameter_resolution):
#             Y = np.sum(np.square(Parameters[n_m, :, 0, n_a, n_i, :, :, :, :]), (0,1,2))/(Number_of_correctly_labeled_images)
# #             m1 = np.sum(Y)/28
# #             Y = Y/m1
# #             Y = Y - np.ones(diffeo_shape)
# #             Y = np.square(Y)
# #             Y_temp += [np.sum(Y) / 28]
#         Y = np.array(Y_temp)
#         axs.plot(Gradient_ascent_step, Y,label = type_of_net + str(size_of_net) +' networks '+ str(n_m+1))
# axs.set_xlabel(r'$\alpha$')
# axs.set_ylabel(r'$\Delta C $')
# axs.legend()
# axs.grid()
# plt.show()

# ### Fig 3 - dC as a function of alpha final #######
fig, axs = plt.figure() ,plt.axes()
fig.suptitle(r'$<\delta C(\alpha)>$ for both ResNet18 and VGG11' )
fig.set_size_inches( 7.5,6.5)
for n_i in range(0,Number_of_initial_conditions):
    Y_tempVGG = []
    Y_tempRes = []
    for n_a in range(0,Parameter_resolution):
        YVGG = np.sum(np.square(ParametersVGG[:, :, 0, n_a, n_i, :, :, :, :]), (0,1,2,3))/(Number_of_correctly_labeled_images*3)
        CmaxVGG = np.max(np.reshape(YVGG,36))
        CminVGG = np.min(np.reshape(YVGG+ZERO_MATRIX*1000000, 36))
        Y_tempVGG += [CmaxVGG/CminVGG]

        YRes = np.sum(np.square(ParametersRes[:, :, 0, n_a, n_i, :, :, :, :]), (0,1, 2, 3)) / (Number_of_correctly_labeled_images * 3)
        CmaxRes = np.max(np.reshape(YRes, 36))
        CminRes = np.min(np.reshape(YRes + ZERO_MATRIX * 1000000, 36))
        Y_tempRes += [CmaxRes / CminRes]
    YVGG = np.array(Y_tempVGG)
    YRes = np.array(Y_tempRes)
    axs.plot(Gradient_ascent_step, YVGG, color=(43/255,186/255,186/255),linewidth = 2.5, linestyle='dashed',label='VGG11')
    axs.plot(Gradient_ascent_step,YRes, color=(164/255,101/255,246/255), linewidth = 2.5,linestyle='dashed', label ='ResNet18')
axs.set_xlabel(r'$\alpha$')
axs.set_ylabel(r'$\delta C $')
axs.legend()
axs.grid()
plt.show()
#
# #### Fig 4 - Final_step as a function alpha #######
fig, axs = plt.figure() ,plt.axes()
fig.suptitle(r'$<n_{f}>(\alpha)$ for 3 different '+ type_of_net + str(size_of_net)  + r' networks')
fig.set_size_inches( 7.5,6.5)
for n_m in range(0, len(M)):
    for n_i in range(0,Number_of_initial_conditions):
        Y = np.sum(Number_of_steps[n_m, :, 0, :, n_i, :],(0,2))
        axs.plot(Gradient_ascent_step, Y,label = type_of_net + str(size_of_net) +' networks '+ str(n_m+1))
axs.set_xlabel(r'$\alpha$')
axs.set_ylabel(r'$<n_{f}>$')
axs.legend()
axs.grid()
plt.show()

############# NITER PLOTS ##############
