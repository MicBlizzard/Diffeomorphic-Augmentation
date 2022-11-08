import torch
import math as mt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Data_treatement
import Transformations
import Useful_functions

Ndiffeo = 200

# NON_ZERO_MATRIX = np.array(
#         [[1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 0.], [1., 1., 1., 1., 1., 0.],
#          [1., 1., 1., 1., 0., 0.], [1., 1., 0., 0., 0., 0.]])
#     ZERO_MATRIX = -1 * NON_ZERO_MATRIX + np.ones(NON_ZERO_MATRIX.shape)
#     NON_ZERO_TENSOR = torch.tensor(NON_ZERO_MATRIX, dtype=torch.bool)
#     print(NON_ZERO_TENSOR)
#
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
    Image_size = int(np.sqrt(dimension/3).item())
    x = torch.linspace(0, 1, Image_size, device=device)
    k = torch.arange(1, cut_off + 1, device = device)
    i, j = torch.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    NON_ZERO_TENSOR = (r < cut_off + 0.5)

    if (Data_type == "Vectors"):
        cut_off = len(diffeo)
    if (temperature > 0 ):
        diffeo_shape, total_parameters = Transformations.Diffeo_shape(diffeo_shape,cut_off)
        shape = (batch_size,) + diffeo_shape
        parameters = torch.zeros(shape).to(device)
    for epoch in range(0, N_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # loop over each batch
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = Net(inputs)
            predicted = torch.max(outputs.data, 1).indices.to(device)
            if( temperature > 0 ):
                print(parameters.shape,parameters)
                parameters[:,NON_ZERO_TENSOR] = (torch.randn(shape)[:,NON_ZERO_TENSOR]).to(device)
                print(parameters)
                Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension, index_of_discriminating_factor, diffeo)
                BOOL_OF_EQUALITY = torch.eq(predicted, labels).to(device)
                SUM_OF_EQUAL = torch.sum(BOOL_OF_EQUALITY).item()
                index = 1
                while ((SUM_OF_EQUAL > 0)):
                    # Transf.parameters[BOOL_OF_EQUALITY] =  (torch.randn(shape)[BOOL_OF_EQUALITY]).to(device)
                    # outputs = Net(Transf.Diffeomorphism(inputs))
                    # predicted = torch.max(outputs.data, 1).indices.to(device)
                    # BOOL_OF_EQUALITY = torch.eq(predicted, labels).to(device)
                    # SUM_OF_EQUAL = torch.sum(BOOL_OF_EQUALITY).item()
                    # print(index)
                    index += 1
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