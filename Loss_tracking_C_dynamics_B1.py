import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import Model_definitions_Leo
import Data_treatement
import Transformations

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
Nres = 4
Nimages = 1000

Parameters = np.zeros((3,Nimages,Nres) + shape)
Norms = np.zeros((3,Nimages,Nres,10))
Losses = np.zeros((3,Nimages,Nres,10))
Losses_per_step = [[],[],[]]
average_radius_of_gradient = [[],[],[]]

R = np.zeros((cut_off,cut_off))
for i in range(0,cut_off):
    for j in range(0,cut_off):
        R[i,j] = ((i+1)**2+(j+1)**2)
R = torch.tensor(R).to(device)

#GA = np.linspace(0.005, 1, num=Nres, endpoint=True, dtype=None, axis=0)
GA = np.logspace(-2, 0, num=Nres, endpoint=True,base=10.0, dtype=None, axis=0)
passed = np.zeros(Nres)
Number_of_retries = 1
criterion = nn.CrossEntropyLoss()
diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape
Cs = torch.zeros((3,Nres) + shape).to(device)
for n,m in enumerate(M,0):
    Nrealex = 0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = m(inputs)
        predicted = torch.max(outputs.data, 1).indices.to(device)
        p = torch.max(outputs.data, 1)
        li = criterion(outputs, labels)
        if(predicted==labels):
            Nrealex += 1
            Losses_per_step[n] += [[]]
            average_radius_of_gradient[n] += [[]]
            for n_a, a in enumerate(GA, 0):
                Losses_per_step[n][Nrealex-1] += [[]]
                average_radius_of_gradient[n][Nrealex-1] += [[]]
                parameters = (10 **(-6)*torch.ones(shape)).to(device)
                #parameters = (10**(-6)+9*(10**(-6))*torch.rand(shape)).to(device)
                parameters.requires_grad_(requires_grad= True)
                Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                outputs = m(inputs_temp)
                predicted = torch.max(outputs.data, 1).indices.to(device)
                index = 0
                delta_L = 0
                while ((predicted == labels)):
                    Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    outputs = m(inputs_temp)
                    predicted = torch.max(outputs.data, 1).indices.to(device)
                    l = criterion(outputs, labels)
                    l.backward()
                    delta_L = l-li
                    # if (index % 100 == 0):
                    #     print(delta_L)
                    Losses_per_step[n][Nrealex - 1][n_a] += [delta_L.item()]

                    with torch.no_grad():
                        parameters += a * (parameters.grad)
                        i = int(torch.argmax(abs(parameters.grad)).item()%6+1)
                        j = int(torch.argmax(abs(parameters.grad)).item()/6+1)
                        value = torch.sum(torch.mul(R,torch.mul(parameters.grad,parameters.grad))).item()
                    average_radius_of_gradient[n][Nrealex - 1][n_a] += [value]

                    parameters.grad.zero_()
                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    outputs = m(inputs_temp)
                    predicted = torch.max(outputs.data, 1).indices.to(device)
                    index += 1

                    if ( torch.norm((parameters)).item() > 500)or(index > 1000/a):
                        print("param norm ",torch.norm((parameters)).item(),"number of iterations", index)
                        print("faulty image",Nrealex,n_a)
                        break
                #print(Nrealex)
                with torch.no_grad():
                    Parameters[n,Nrealex-1,n_a,:,:,:] = np.array(parameters)
                    Norms[n,Nrealex-1,n_a,labels.item()] = torch.norm((parameters)).item()
                    Losses[n,Nrealex-1,n_a,labels.item()] = (l-li).item()
            if (Nrealex >= Nimages):
                break


############# Printing and Plotting ##################

    for c in range(0,10):
        for n_a in range(0,Nres):
            Average = np.sum(Norms[n,:,n_a,c])
            print('the average with resolution ', GA[n_a], 'of the ',c,'th class is ',Average)
            #print("the average |C|'s for model ", n+1, """are ",np.sum(abs(Parameters[n,:,n_a,c]))/Nimages)
        print("")

    X = np.linspace(0,35,36)

fig, axs = plt.subplots(1, len(M))
fig.suptitle('Stepsizes for different networks')
for n in range(0,len(M)):
    for n_a in range(0,Nres):
        Y = np.sum((Parameters[n,:,n_a,:,:,:])*(Parameters[n,:,n_a,:,:,:]),0)
        Y = np.reshape(Y / Nimages,(36,1))
        axs[n].plot(X, Y, label = "step size of "+str(GA[n_a]))
    axs[n].legend()
plt.show()

N_IMAGES_ALPHA = 1

fig, axs = plt.subplots(1, len(M))
fig.suptitle('Norm evolution for different step sizes and images')
for n in range(0,len(M)):
    for i in range(0,N_IMAGES_ALPHA):
        Y = np.sum(Norms[n,i,:,:],1)
        axs[n].plot(GA, Y, label = "step size of "+str(GA[n_a])+" and image "+str(i))
        axs[n].legend()
plt.show()
plt.savefig('Damping_effect.png')


np.save('./results/B1AverageCs', Cs.cpu().detach().numpy())
np.save('./results/B1norms', Norms)
np.save('./results/B1Losses', Losses)
np.save('./results/B1Losses_per_step',np.array(Losses_per_step))