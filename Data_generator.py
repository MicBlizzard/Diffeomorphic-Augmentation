import numpy as np
import Data_models
import Data_treatement
#
# Number_of_clusters = 2000  # = Number of total points
# Number_of_points_per_cluster = 1  # =1 if Stripe model
# Number_of_points = Number_of_points_per_cluster * Number_of_clusters
#
# dimension_of_space = 2  # for now this is the only acceptable answer
# index_of_discriminative_factor = 0  # for now this is the only acceptable answer
#
# # parameter for the NODVSM model
# number_of_frequencies = 10  # set to 0 when dealing with Stripe
#
# #### diffeo_unit vector - to do ####
#
# diffeo_unit_vector = np.ones(dimension_of_space)
# norm = np.linalg.norm(diffeo_unit_vector)
# diffeo_unit_vector = diffeo_unit_vector / norm
#
# # parameters defining the stripe model (included in NODVSM)
# beginnings = []
# ends = []
# for i in range(0, dimension_of_space):
#     beginnings += [0]
#     ends += [10]
#     # end = round(100*Number_of_clusters*mt.pi*(((number_of_frequencies+3)/10)**2)) # ensures a roughly 1% chance of intersection of two balls
#
# boundary_per_class = [[[beginnings[index_of_discriminative_factor],
#                         (ends[index_of_discriminative_factor] - beginnings[index_of_discriminative_factor]) / 2]], [
#                           [(ends[index_of_discriminative_factor] - beginnings[index_of_discriminative_factor]) / 2,
#                            ends[index_of_discriminative_factor]]]]  # typical binary classifier
#
# strength_of_similarity = 1
#
# number_of_classes = len(boundary_per_class)
# mean = (ends[index_of_discriminative_factor] - beginnings[index_of_discriminative_factor]) / 2
# var = 1
#
# Mean = mean * np.ones(dimension_of_space)
# Cov = var * np.identity(dimension_of_space)
#
# ### initialize the model
#
# Model = Data_models.Stripe(boundary_per_class, index_of_discriminative_factor, strength_of_similarity, dimension_of_space)
# #Model = Data_models.nodvsm(boundary_per_class, index_of_discriminative_factor, strength_of_similarity,beginnings,ends, Mean, Cov, diffeo_unit_vector, Number_of_clusters, number_of_frequencies)
# #Model = Data_models.CIFAR10()
#
# Normalize = True

def data_genetor(Model , Number_of_points_per_cluster, Normalize,train_test_split = 0.8):
    Data = []
    points = []
    labels = []

    shifts = []
    scales = []
    for i in range(0, Model.dimension_of_space):
        shifts += [0]
        scales += [1]

    if (Model.type == "Stripe"):
        for i in range(0, Number_of_points_per_cluster):
            x = -1 * np.ones(Model.dimension_of_space)
            lower = np.sign(x - np.array(Model.beginnings))
            upper = np.sign(np.array(Model.ends) - x)
            lower_test = np.amin(lower)
            upper_test = np.amin(upper)
            while (lower_test < 0) or (upper_test < 0):
                x = np.random.uniform(Model.beginnings[Model.index_of_discriminative_factor], Model.ends[Model.index_of_discriminative_factor],
                                      Model.dimension_of_space)
                # x = nprd.multivariate_normal(Mean, Cov, size=None, check_valid='warn', tol=1e-8)
                lower = np.sign(x - np.array(Model.beginnings))
                upper = np.sign(np.array(Model.ends) - x)
                lower_test = np.amin(lower)
                upper_test = np.amin(upper)
            points += [x.tolist()]
            labels += [(Model.f(points[i]))]

    if (Model.type == "NODVSM"):
        for i in range(0, Model.number_of_clusters):
            for j in range(0, Number_of_points_per_cluster):
                temp_x_rel = np.random.uniform(0, 1, Model.dimension_of_space)
                temp_unit_x_rel = temp_x_rel / np.linalg.norm(temp_x_rel)
                R = Model.Radius(i, temp_unit_x_rel)
                r_ij = np.random.uniform(0, R)
                x_rel = r_ij * temp_unit_x_rel
                x = np.array(Model.class_points[i]) + x_rel
                points += [x.tolist()]
                labels += [(Model.labels[i])]

    if (Model.type == "CIFAR10"):
        points = Model.points
        labels = Model.labels
        shifts = Model.shifts
        scales = Model.scales

    if (Normalize):
        shifts = Model.beginnings
        scales = []
        for i in range(0, len(Model.ends)):
            scales += [Model.ends[i] - Model.beginnings[i]]
        for i in range(0, Number_of_points_per_cluster):
            for j in range(0, Model.dimension_of_space):
                points[i][j] = (points[i][j] - shifts[j]) / scales[j]

    points = np.array(points)
    labels = np.array(labels)
    shifts = np.array(shifts)
    scales = np.array(scales)
    #
    Data = [points, labels, shifts, scales]
    #
    # #### IMPORT DATA TREATEMENT FOR THE REST

    Data_train, Data_test = Data_treatement.Train_test_split(Data,train_test_split, Type=0)

    print("the length of the train set is ", len(Data_train[0]))
    print("the length of the train set is ", len(Data_test[0]))

    return Data, Data_train, Data_test
#
# if (Model.type == "Stripe"):
#     for i in range(0, Number_of_points):
#         x = -1 * np.ones(dimension_of_space)
#         lower = np.sign(x - np.array(beginnings))
#         upper = np.sign(np.array(ends) - x)
#         lower_test = np.amin(lower)
#         upper_test = np.amin(upper)
#         while (lower_test < 0) or (upper_test < 0):
#             x = np.random.uniform(beginnings[index_of_discriminative_factor], ends[index_of_discriminative_factor],
#                                   dimension_of_space)
#             # x = nprd.multivariate_normal(Mean, Cov, size=None, check_valid='warn', tol=1e-8)
#             lower = np.sign(x - np.array(beginnings))
#             upper = np.sign(np.array(ends) - x)
#             lower_test = np.amin(lower)
#             upper_test = np.amin(upper)
#         points += [x.tolist()]
#         labels += [(Model.f(points[i]))]
#
# if(Model.type == "NODVSM"):
#     for i in range(0, Number_of_clusters):
#         for j in range(0, Number_of_points_per_cluster):
#             temp_x_rel = np.random.uniform(0, 1, dimension_of_space)
#             temp_unit_x_rel = temp_x_rel / np.linalg.norm(temp_x_rel)
#             R = Model.Radius(i, temp_unit_x_rel)
#             r_ij = np.random.uniform(0, R)
#             x_rel = r_ij * temp_unit_x_rel
#             x = np.array(Model.class_points[i]) + x_rel
#             points += [x.tolist()]
#             labels += [(Model.labels[i])]
#
# if(Model.type == "CIFAR10"):
#     Normalize = False
#     points = Model.points
#     labels = Model.labels
#     shifts = Model.shifts
#     scales = Model.scales
# if (Normalize):
#     shifts = beginnings
#     scales = []
#     for i in range(0, len(ends)):
#         scales += [ends[i] - beginnings[i]]
#     for i in range(0, Number_of_points):
#         for j in range(0, dimension_of_space):
#             points[i][j] = (points[i][j] - shifts[j]) / scales[j]
#
# points = np.array(points)
# labels = np.array(labels)
# shifts = np.array(shifts)
# scales = np.array(scales)
# #
# Data = [points, labels, shifts, scales]
# #
# # #### IMPORT DATA TREATEMENT FOR THE REST
#
# Data_train, Data_test = Data_treatement.Train_test_split(Data, 0.800, Type=0)
#
# print("the length of the train set is ", len(Data_train[0]))
# print("the length of the train set is ", len(Data_test[0]))
#
# Data_treatement.DataSave_NP(Data_train, True)
# Data_treatement.DataSave_NP(Data_test, False)
# Data_treatement.Model_save(Model)
# Data_treatement.MetaSave(Data)