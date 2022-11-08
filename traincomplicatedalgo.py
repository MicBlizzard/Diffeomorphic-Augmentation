import torch
import math as mt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Model_definitions_Leo
import Data_treatement
import Transformations
import Useful_functions

def Average(lst):
    if (len(lst)>0):
        return sum(lst) / len(lst)
    else:
        return "No average"

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}
def train(device,Initial_PATH,Final_PATH,batch_size,N_epochs, cut_off = 1 , temperature = 0, N_attack=0, gradient_ascent_step_size=0.1):
    Ndiffeo = 1000
    temperature = 0.001
    gradient_ascent_step_size = 1
    Nwalk = 1000/gradient_ascent_step_size
    Net = Model_definitions_Leo.VGG("VGG11")
    #temp = torch.load(Initial_PATH + str(1) + ".pt")
    temp = torch.load("./model/vggbn_trained2.pt")
    Net.load_state_dict(temp)
    Norms = []
    Likelyhoods = []
    final_average_loss = 0
    # Net = torch.load(Initial_PATH)
    Net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)
    trainset, Data_type, Model, dimension, index_of_discriminating_factor, diffeo, diffeo_shape = Data_treatement.Train_Import_NP()
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
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
        Transf = Transformations.Transformations(temperature, cut_off, parameters, Model, Data_type, dimension,
                                                 index_of_discriminating_factor, diffeo)
    for epoch in range(0, N_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        Number_of_walkers = 0
        Number_of_weak_adversarial = 0
        for i, data in enumerate(trainloader, 0): # loop over each batch
            Walk = False
            Weak = False
            Transf.parameters = torch.zeros(shape).to(device)
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = Net(inputs)
            predicted = torch.max(outputs.data, 1).indices.to(device)
            print(i,predicted,labels)
            if( temperature > 0 ):
                Net.eval()
                BOOL_OF_EQUALITY = torch.eq(predicted, labels).to(device)
                SUM_OF_EQUAL = torch.sum(BOOL_OF_EQUALITY).item()
                index = 0
                while ((SUM_OF_EQUAL > 0)):
                    Transf.parameters.requires_grad = False
                    NEW_TEMP = torch.zeros(shape,dtype = torch.bool)
                    NEW_TEMP[BOOL_OF_EQUALITY] = NON_ZERO_TENSOR
                    Transf.parameters[NEW_TEMP] = (torch.randn(shape)[NEW_TEMP]).to(device)
                    #print(Transf.parameters)
                    outputs = Net(Transf.Diffeomorphism(inputs)[0])
                    #print(Transf.Diffeomorphism(inputs)[1])
                    predicted = torch.max(outputs.data, 1).indices.to(device)
                    BOOL_OF_EQUALITY = torch.eq(predicted, labels).to(device)
                    SUM_OF_EQUAL = torch.sum(BOOL_OF_EQUALITY).item()
                    #print(index)
                    if (index > Ndiffeo):
                        walk_index = 0
                        Transf.parameters.requires_grad = True
                        Walk = True
                        while ((SUM_OF_EQUAL > 0)and(walk_index <= Nwalk)):
                            NEW_TEMP = torch.zeros(shape, dtype=torch.bool)
                            NEW_TEMP[BOOL_OF_EQUALITY] = NON_ZERO_TENSOR
                            outputs = Net(Transf.Diffeomorphism(inputs)[0])
                            l = criterion(outputs, labels)
                            l.backward()
                            with torch.no_grad():
                                #print(Transf.parameters,Transf.parameters.grad)
                                Transf.parameters[NEW_TEMP] += gradient_ascent_step_size * Transf.parameters.grad[NEW_TEMP]
                                #print(Transf.parameters)
                            Transf.parameters.grad.zero_()
                            outputs = Net(Transf.Diffeomorphism(inputs)[0])
                            predicted = torch.max(outputs.data, 1).indices.to(device)
                            BOOL_OF_EQUALITY = torch.eq(predicted, labels).to(device)
                            SUM_OF_EQUAL = torch.sum(BOOL_OF_EQUALITY).item()
                            #print('N walk is',walk_index)
                            walk_index += 1
                    index += 1
                    if (index > int(1.01*Ndiffeo)):
                        Weak = True
                        break
            # Net.train()
            # # FORWARD & BACKWARD PASSES & UPDATE
            if (Walk):
                print("walker")
                Number_of_walkers += 1
            if (Weak):
                print("Weak")
                Number_of_weak_adversarial += 1
            #
            # optimizer.zero_grad()  # zero the parameter gradients
            # if (temperature > 0):
            #     Norms += Transf.Diffeomorphism(inputs)[1]
            #     Likelyhoods += Useful_functions.likelyhood_of_diffeomorphisms(parameters,temperature)
            #     outputs = Net(Transf.Diffeomorphism(inputs)[0])
            # else:
            #     outputs = Net(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()  # General optimization
            # running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f} Walkers: {Number_of_walkers / 200:.3f} Weak samples : {Number_of_weak_adversarial / 200:.3f}')
                Number_of_walkers = 0
                Number_of_weak_adversarial = 0
                final_average_loss = running_loss/200
                running_loss = 0.0

    print('Finished Training')
    torch.save(Net, Final_PATH)
    return final_average_loss, Average(Norms), Average(Likelyhoods)