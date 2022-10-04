import torchvision
from torchvision import datasets, transforms
import Data_models
import numpy as np

# Type 0 - fraction of data set
# type 1 - fixed number of train
# type 2 - fixed number of test

def Train_test_split(Data, Number_or_fraction, Type=0):
    if (Type == 0):
        if (Number_or_fraction > 1):
            Number_or_fraction /= len(Data[0])
            print(Number_or_fraction)
        L = len(Data[0])
        Tr = round(L * Number_or_fraction)
        Data_train = [Data[0][:Tr], Data[1][:Tr],Data[2],Data[3]]
        Data_test = [Data[0][Tr:], Data[1][Tr:],Data[2],Data[3]]
    if (Type == 1):
        Data_train = [Data[0][:Number_or_fraction], Data[1][:Number_or_fraction],Data[2],Data[3]]
        Data_test = [Data[0][Number_or_fraction:], Data[1][Number_or_fraction:],Data[2],Data[3]]
    if (Type == 2):
        Data_train = [Data[0][:-Number_or_fraction], Data[1][:-Number_or_fraction],Data[2],Data[3]]
        Data_test = [Data[0][-Number_or_fraction:], Data[1][-Number_or_fraction:],Data[2],Data[3]]
    return Data_train, Data_test

import csv
##### saving appropreately ###

import csv
##### saving appropreately ###

def DataSave_NP(Data, train = True):
    if (train):
        np.save('./data/train/inputs', Data[0])
        np.save('./data/train/labels', Data[1])
        np.save('./data/train/shifts', Data[2])
        np.save('./data/train/scales', Data[3])
    else:
        np.save('./data/test/inputs', Data[0])
        np.save('./data/test/labels', Data[1])
        np.save('./data/test/shifts', Data[2])
        np.save('./data/test/scales', Data[3])

def Model_save(Model):
    informations = [Model.type,str(Model.dimension_of_space),str(Model.index_of_discriminative_factor)]
    diffeo_basis = []
    if (informations[0] == "NODVSM"):
        diffeo_basis += [Model.diffeo_unit_vector]
    if (informations[0] == "Stripe"):
        for i in range(0,Model.index_of_discriminative_factor):
            vector = np.zeros(Model.dimension_of_space)
            vector[i] = int(1)
            diffeo_basis += [vector]
        for i in range(Model.index_of_discriminative_factor+1,Model.dimension_of_space):
            vector = np.zeros(Model.dimension_of_space)
            vector[i] = int(1)
            diffeo_basis += [vector]
    if (informations[0] == "CIFAR10"):
        for i in range(0,32):
            for j in range(0,32):
                vector = np.zeros((32,32))
                vector[i,j] = int(1)
                diffeo_basis += [vector]
    diffeo_basis = np.array(diffeo_basis)
    np.save("./data/diffeo_basis.npy",diffeo_basis)
    with open(r'./data/information.txt', 'w') as fp:
        for item in informations:
            # write each item on a new line
            fp.write("%s\n" % item)

def DataSave(Data, train = True):
    if (train):
        file_vector = open('./data/Vectors/train/vectors.csv', 'w')
        file_label = open('./data/Vectors/train/labels.csv', 'w')
    else:
        file_vector = open('./data/Vectors/test/vectors.csv', 'w')
        file_label = open('./data/Vectors/test/labels.csv', 'w')
    writer_vector = csv.writer(file_vector)
    writer_label = csv.writer(file_label)
    for i in range(0,len(Data[0])):
        vector = Data[0][i]
        label =  [Data[1][i]]
        writer_vector.writerow(vector)
        writer_label.writerow(label)
    file_vector.close()
    file_label.close()

def MetaSave(Data):
    file = open('./data/Vectors/meta.csv', 'w')
    writer = csv.writer(file)
    for i in range(0,len(Data[2])):
        vector = [Data[2][i],Data[3][i]]
        writer.writerow(vector)
    file.close()

def Train_Import_NP(location = "./data"):
    trainset =  Data_models.CustomDataset(root=location, train=True)
    with open('./data/information.txt') as f:
        lines = f.readlines()
    #print(lines)
    Model = (lines[0])
    dimension = int(lines[1])
    index_of_discriminative_factor = int(lines[2])
    diffeomorphism_unit_vectors = np.load(location+"/diffeo_basis.npy")
    diffeo_shape = diffeomorphism_unit_vectors[0].shape
    return trainset, trainset.Data_type, Model, dimension, index_of_discriminative_factor, diffeomorphism_unit_vectors, diffeo_shape

def Test_Import_NP(location = "./data"):
    testset = Data_models.CustomDataset(root=location, train=False)
    with open('./data/information.txt') as f:
        lines = f.readlines()
    #print(lines)
    Model = (lines[0])
    dimension = int(lines[1])
    index_of_discriminative_factor = int(lines[2])
    diffeomorphism_unit_vectors = np.load(location+"/diffeo_basis.npy")
    diffeo_shape = diffeomorphism_unit_vectors[0].shape
    return testset, testset.Data_type, Model, dimension, index_of_discriminative_factor, diffeomorphism_unit_vectors, diffeo_shape

def Import_NP(location = "./data"):
    with open('information.txt') as f:
        lines = f.readlines()
        print(lines)
    trainset =  Data_models.CustomDataset(root=location, train=True)
    testset = Data_models.CustomDataset(root=location, train=True)

    return trainset, testset, trainset.Data_type

def Import(location):
    x = location.split('/')
    Data_type = x[-1]
    Data = []
    if (Data_type == "Vectors"):
        trainset = Data_models.CustomVectorDataset(root="./" + location, train=True)
        testset = Data_models.CustomVectorDataset(root="./" + location, train=False)
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if (Data_type == "Images"):
            trainset = Data_models.CustomImageDataset(root="./" + location, train=True, transform=transform)
            testset = Data_models.CustomImageDataset(root="./" + location, train=False, transform=transform)

        else:
            Data_type = "CIFAR10"
            trainset = torchvision.datasets.CIFAR10(root="./" + x[0], train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root="./" + x[0], train=False, download=True, transform=transform)
            print("train test split ", trainset.__len__(), testset.__len__())
    return trainset, testset, Data_type