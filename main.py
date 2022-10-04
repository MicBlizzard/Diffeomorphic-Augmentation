import Data_generator
import Data_models
import Data_treatement
import Model_definitions
import train
import numpy as np
import torch as tr

######## GENERAL INFORMATION ##############

device = tr.device('cuda' if tr.cuda.is_available() else 'cpu')
Initial_PATH = "./model/net.pth"
Final_PATH = "./model/net_trained.pth"

############# DATA GENERATION ################

#Type_of_System = ["Vectors","Stripe"]
#Type_of_System = ["Vectors","NODVSM"]
Type_of_System = ["Images","CIFAR10"]

if (Type_of_System[0] == "Vectors"):

    dimension_of_space = 2  # for now this is the only acceptable answer
    index_of_discriminative_factor = 0  # direction of discrimination

    beginnings = []
    ends = []
    for i in range(0, dimension_of_space):
        beginnings += [0]
        ends += [10]
        # end = round(100*Number_of_clusters*mt.pi*(((number_of_frequencies+3)/10)**2)) # ensures a roughly 1% chance of intersection of two balls

    # the boundaries that define the different classes (usually 2)
    boundary_per_class = [[[beginnings[index_of_discriminative_factor], (ends[index_of_discriminative_factor] - beginnings[index_of_discriminative_factor]) / 2]], [[(ends[index_of_discriminative_factor] - beginnings[index_of_discriminative_factor]) / 2, ends[index_of_discriminative_factor]]]]  # typical binary classifier

    Number_of_classes = len(boundary_per_class)
    strength_of_similarity = 1


    #### Stripe ####

    if(Type_of_System[1] == "Stripe"):
        Number_of_points = 2000  # = Number of total points
        N_points = Number_of_points
        Model = Data_models.Stripe(boundary_per_class,beginnings,ends, index_of_discriminative_factor, strength_of_similarity, dimension_of_space)
    #### NODVSM ####

    if (Type_of_System[1] == "NODVSM"):
        Number_of_clusters = 2000  # = 1 if stripe model
        Number_of_points_per_cluster = 1  # = Number of total points if stripe model
        N_points = Number_of_points_per_cluster
        Number_of_points = Number_of_points_per_cluster * Number_of_clusters
        number_of_frequencies = 10  # set to 0 when dealing with Stripe

        ### this is one usual initialization of the diffeo vector
        diffeo_unit_vector = np.ones(dimension_of_space)
        norm = np.linalg.norm(diffeo_unit_vector)
        diffeo_unit_vector = diffeo_unit_vector / norm

        strength_of_similarity = 1

        number_of_classes = len(boundary_per_class)
        mean = (ends[index_of_discriminative_factor] - beginnings[index_of_discriminative_factor]) / 2
        var = 1
        Data = []
        points = []
        labels = []

        shifts = []
        scales = []
        for i in range(0, dimension_of_space):
            shifts += [0]
            scales += [1]

        Mean = mean * np.ones(dimension_of_space)
        Cov = var * np.identity(dimension_of_space)
        Model = Data_models.nodvsm(boundary_per_class, index_of_discriminative_factor, strength_of_similarity,beginnings,ends, Mean, Cov, diffeo_unit_vector, Number_of_clusters, number_of_frequencies)

    Normalize = True

else:
    #### CIFAR10 ####
    if (Type_of_System[1] == "CIFAR10"):
        Normalize = False
        N_points = 60000
        Model = Data_models.CIFAR10()
        points = Model.points
        labels = Model.labels
        shifts = Model.shifts
        scales = Model.scales
        # Model = Data_models.CIFAR10()

Data, Data_train, Data_test = Data_generator.data_genetor(Model,N_points, Normalize,train_test_split = 0.8)

Data_treatement.DataSave_NP(Data_train, True)
Data_treatement.DataSave_NP(Data_test, False)
Data_treatement.Model_save(Model)
Data_treatement.MetaSave(Data)

################# Neural Network Initialization ###################

if (Type_of_System[0] =="Images"):
    convolution_channels = [64,128,256,512,512]
    convolution_depths = [1,1,1,2,2]
    types_of_pooling = ['M','M','M','M','M']
    pool_factors = [2,2,2,2,2]
    net = Model_definitions.VGG(Model_definitions.configuration_builder(convolution_channels,convolution_depths,types_of_pooling))
    net.to(device)
else:
    depth = 100
    net = Model_definitions.SimpleFCC(dimension_of_space,depth,Number_of_classes)
    net.to(device)
tr.save(net, Initial_PATH)

################# TRAINING #######################

batch_size = 4
N_epochs = 1

Type_of_training = "Standard"
#Type_of_training = "Adversarial"

if (Type_of_training== "Standard"):

    final_loss, Norms, Likelyhood = train.train(device, Initial_PATH, Final_PATH, batch_size, N_epochs)

if (Type_of_training == "Adversarial"):
    cut_off = 2
    temperature = 0.003
    if (Type_of_System[0] == "Vectors"):
        temperature = 0
    N_attack = 2
    gradient_ascent_step_size = 0.01

    final_loss, Norms, Likelyhood = train.train(device, Initial_PATH, Final_PATH, batch_size, N_epochs, cut_off, temperature, N_attack, gradient_ascent_step_size)

print('The final loss is ',final_loss,'the average deformation norm is ', Norms,"the diffeomorphism likelyhood is ", Likelyhood)