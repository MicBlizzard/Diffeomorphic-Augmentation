import torch
import math as mt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Data_treatement
import Transformations
import Useful_functions

def Average(lst):
    if (len(lst)>0):
        return sum(lst) / len(lst)
    else:
        return "No average"
def train(device,Initial_PATH,Final_PATH,batch_size,N_epochs, cut_off = 1 , temperature = 0, N_attack=0, gradient_ascent_step_size=0):
    Norms = []
    Likelyhoods = []
    final_average_loss = 0
    Net = torch.load(Initial_PATH)
    Net.eval()
    Net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)
    trainset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Train_Import_NP()
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=0)
    if (Data_type == "Vectors"):
        cut_off = len(diffeo)
    if (temperature > 0 ):
        diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
        #distr = Transformations.diffeo_distr(batch_size,diffeo_shape,device)
        shape = (batch_size,) + diffeo_shape
    for epoch in range(0, N_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # loop over each batch
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            if( temperature > 0 ):
                parameters = torch.randn(shape,requires_grad = True).to(device)
                #parameters = torch.tensor(Transformations.diffeo_parameters(distr, shape, diffeo_shape, device, temperature), requires_grad=True).to(device)
                # parameters = torch.zeros(shape, requires_grad=True).to(device)
                # parameters = ((10**(-8))*torch.ones(shape)).to(device)
                # parameters.requires_grad_(requires_grad= True)
                Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                for n in range(0, N_attack):
                    inputs_temp, Norms_i = Transf.Diffeomorphism(inputs)
                    inputs_temp.to(device)
                    outputs_temp = Net(inputs_temp)
                    l = criterion(outputs_temp, labels)
                    l.backward()
                    with torch.no_grad():
                        parameters += gradient_ascent_step_size * (parameters.grad)
                        #print("the gradient is ",parameters.grad)

            ### FORWARD & BACKWARD PASSES & UPDATE

            optimizer.zero_grad()  # zero the parameter gradients
            if (temperature > 0):
                Norms += Transf.Diffeomorphism(inputs)[1]
                Likelyhoods += Useful_functions.likelyhood_of_diffeomorphisms(parameters,temperature)
                outputs = Net(Transf.Diffeomorphism(inputs)[0])
            else:
                outputs = Net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # General optimization
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                final_average_loss = running_loss/200
                running_loss = 0.0

    print('Finished Training')
    torch.save(Net, Final_PATH)
    return final_average_loss, Average(Norms), Average(Likelyhoods)