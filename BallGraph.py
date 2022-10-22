import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import Model_definitions_Leo
import Data_treatement
import Transformations


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

M = [Model1]
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
gradient_ascent_step_size = 0.1

Nimages = 3
diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape
batch_size = 1

Number_of_retries = 1
criterion = nn.CrossEntropyLoss()
diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
shape = (batch_size,) + diffeo_shape
NCs = 40
Cs = torch.linspace(-0.1, 0.1, NCs)
print(Cs)
outputs = torch.zeros(3,36,36,Nimages,NCs,NCs)
for a in range(2,3):
    for b in range(3, 4):
        cut = ((0,int(a/6),a%6),(0,int(b/6),b%6))
        print(cut)
        for n,m in enumerate(M,0):
            Nrealexample = 0
            for i, data in enumerate(trainloader,0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                if (i < Nimages):
                    for n_c,c in enumerate(Cs,0):
                        for n_d,d in enumerate(Cs,0):
                            parameters = torch.zeros(shape).to(device)
                            parameters[cut[0]] = c
                            parameters[cut[1]] = d
                            parameters.requires_grad_(requires_grad= True)
                            Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                            #print(Transf.parameters)
                            inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                            out =  m(inputs_temp)
                            predicted = torch.max(out.data, 1).indices.item()
                            outputs[n,a,b, i, n_c, n_d] = criterion(out,labels).item()
                           #print(n_c,n_d)
for i in range(0,6):
    for j in range(0,6):
        for k in range(0,6):
            for l in range(0,Nimages):
                plt.imshow(outputs[i,j,k,l], interpolation='none')
                plt.colorbar()
                plt.show()
np.save('./results/ball', outputs)



