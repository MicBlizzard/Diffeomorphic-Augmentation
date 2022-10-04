import random as rd
rd.seed(42)
import math as mt
import numpy as np
import numpy.random as nprd
import pandas as pd
import os
from skimage import io
nprd.seed(42)
import matplotlib.pyplot as plt

import torch as tr
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class Model():
    def f(self,x):
        return x

class Stripe(Model):
    def __init__(self, boundary_per_class,beginnings,ends, index_of_discriminative_factor, strength_of_similarity=1, dimension_of_space = 2):
        self.type = "Stripe"
        self.beginnings = beginnings
        self.ends = ends
        self.boundary_per_class = boundary_per_class
        self.number_of_classes = len(self.boundary_per_class)
        self.index_of_discriminative_factor = index_of_discriminative_factor
        self.strength_of_similarity = strength_of_similarity
        self.cluster = []
        self.dimension_of_space = dimension_of_space
    def f(self, x):
        lab = -1
        for Class in self.boundary_per_class:
            lab += 1
            for interv in Class:
                if (interv[0] <= x[self.index_of_discriminative_factor] < interv[1]):
                    #                     u = nprd(0,1)
                    #                     if (u > Similarty_factor_strength): # this breaks the perfect influence of x on the type of class
                    #                         shift = rd.randint(1, self.number_of_classes-1)
                    #                         lab = (lab_class + shift) % self.number_of_classes
                    break
            else:
                continue

            break
        return lab

    # def diffeo(self, x, magnitude):
    #     d = len(x)
    #     out = np.array(x)
    #     delta = np.rand(d - 1)
    #     delta *= magnitude / (np.linalg.norm(delta))
    #     for i in range(0, self.index_of_discriminative_factor):
    #         out[i] += delta[i]
    #     for j in range(self.index_of_discriminative_factor, d):
    #         out[j + 1] += delta[j]
    #     out = out.tolist()
    #     return out


class nodvsm(Model):
    def __init__(self, boundary_per_class, index_of_discriminative_factor, strength_of_similarity,beginnings,ends, Mean, Cov,
                 diffeo_unit_vector, number_of_clusters, number_of_frequencies):
        # def __init__(self,boundary_per_class,strength_of_similarity,Mean,Cov,diffeo_angle,number_of_clusters,number_of_frequencies):
        self.type = "NODVSM"
        self.dimension_of_space = len(diffeo_unit_vector)
        self.boundary_per_class = boundary_per_class
        self.strength_of_similarity = strength_of_similarity
        self.index_of_discriminative_factor = index_of_discriminative_factor
        self.Stripe = Stripe(boundary_per_class, index_of_discriminative_factor, strength_of_similarity)
        self.beginnings = beginnings
        self.ends = ends

        self.Mean = Mean
        self.Cov = Cov

        self.number_of_clusters = number_of_clusters
        self.diffeo_unit_vector = diffeo_unit_vector
        self.class_points = []
        self.labels = []
        self.constants_per_point = []
        self.number_of_frequencies = number_of_frequencies

        for n in range(0, self.number_of_clusters):
            x = -1 * np.ones(self.dimension_of_space)

            lower = np.sign(x - np.array(beginnings))
            upper = np.sign(np.array(ends) - x)
            lower_test = np.amin(lower)
            upper_test = np.amin(upper)
            constants = []
            while (lower_test < 0) or (upper_test < 0):
                x = np.random.uniform( self.beginnings[index_of_discriminative_factor], self.ends[index_of_discriminative_factor],
                                      self.dimension_of_space)
                # x = nprd.multivariate_normal(Mean, Cov, size=None, check_valid='warn', tol=1e-8)
                lower = np.sign(x - np.array(self.beginnings))
                upper = np.sign(np.array(self.ends) - x)
                lower_test = np.amin(lower)
                upper_test = np.amin(upper)
                x = x.tolist()
            self.class_points += [x]
            self.labels += [(self.Stripe.f(self.class_points[n]))]

            #             #for k in range(0,self.number_of_frequencies):
            #                 #Ak = rd.uniform(0,1/self.number_of_frequencies)
            #                 #Bk = rd.uniform(0,2*mt.pi)
            #                 #constants += [Ak,Bk]
            #             constants += [np.random.uniform(1,1.5)] # the +3 comes from here
            #             L = np.random.uniform(35,155)
            #             constants += [L]
            #             constants += [np.random.uniform(1/(1.2*L),1/(0.8*L))]
            #             self.constants_per_point += [constants]
            constants += [np.random.uniform(2.5, 3)]  # radius of ball
            constants += [np.random.uniform(2.5, 3)]  # additional diffeo
            constants += [np.random.uniform(0.01, 0.05)]  # width of diffeo
            self.constants_per_point += [constants]

    def Radius(self, index, position):
        np_position = np.array(position)
        unit_vector = np_position / np.linalg.norm(np_position)
        # print(unit_vector)
        dot = np.dot(unit_vector, self.diffeo_unit_vector)
        # print(dot)
        # theta = np.arccos(dot)
        # print(theta)
        result = 0
        small_radius = 3
        # N_freq = int((len(self.constants_per_point[index])-3)/2)
        # theta_ = self.diffeo_angle
        # for n in range(0,N_freq):
        # result += self.constants_per_point[index][2*n]*(mt.sin(2*n*(polar_angle-self.constants_per_point[index][2*n+1]))+1)
        result += small_radius
        result += self.constants_per_point[index][0] * mt.exp(
            (-(dot - 1) ** 2) / (2 * (self.constants_per_point[index][1]) ** 2))
        # result += self.constants_per_point[index][0]*mt.exp(-(dot-1)**2/2*(self.constants_per_point[index][1]**2))
        # result += self.constants_per_point[index][2*N_freq]*(np.sinc((polar_angle-theta_)*self.constants_per_point[index][2*N_freq+1])+1)*mt.exp(-(polar_angle-theta_)**2/self.constants_per_point[index][2*N_freq+2])
        # result += self.constants_per_point[index][2*N_freq]*(np.sinc((polar_angle-theta_-2*mt.pi)*self.constants_per_point[index][2*N_freq+1])+1)*mt.exp(-(polar_angle-theta_-2*mt.pi)**2/self.constants_per_point[index][2*N_freq+2])
        return result / 10

    def f(self, x):
        lab = -1
        index = -1
        # ex = [1,0]
        for x_class in self.class_points:
            index += 1
            X_class = np.array(x_class)
            X = np.array(x)
            Delta = X - X_class
            Delta = Delta.tolist()

            #             s = np.sign(Delta)

            #             theta_relative = -1/2*(s[1]-1)*2*mt.pi + s[1]*np.arccos(np.dot(Delta, ex)/(np.linalg.norm(Delta)))
            #             #print("the class point is ",X_class, "the point is ",X, " and the Delta is ", Delta)
            #             #print("the projection is ",np.dot(Delta, ex))
            #             #print("the normalized projection of the delta is ",np.dot(Delta, ex)/(np.linalg.norm(Delta)))

            #             #print("and the reported angle is ",round(theta_relative,3), " rad")
            #             #print(theta_relative)
            Rad = np.linalg.norm(X - X_class)
            if (Rad <= self.Radius(index, Delta)):
                lab = self.labels[index]
                return lab
        return lab

    # def diffeo(self, x, magnitude):
    #     return [x[0] + magnitude * mt.cos(self.diffeo_angle), x[1] + magnitude * mt.sin(self.diffeo_angle)]


class CIFAR10(Model):
    def __init__(self,shifts = [0.5,0.5,0.5],scales = [0.5,0.5,0.5]):
        self.type = "CIFAR10"
        self.dimension_of_space = 3*32*32
        self.index_of_discriminative_factor = -1
        self.shifts = shifts
        self.scales = scales
        batch_size = 1

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(shifts, scales)])

        trainset = torchvision.datasets.CIFAR10(root="./data", transform=transform, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root="./data", transform=transform, train=False, download=True)

        trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=0)
        testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)
        points = []
        labels = []
        for i, data in enumerate(trainloader, 0):
            # print(data[0].numpy())
            points += [data[0].numpy()[0]]
            labels += [data[1].numpy()[0]]

        for i, data in enumerate(testloader, 0):
            # print(data[0].numpy())
            points += [data[0].numpy()[0]]
            labels += [data[1].numpy()[0]]
        self.points = points
        self.labels = labels

def fdisc(f, x1, x2):
    grid = np.zeros((len(x1), len(x2)))
    for i in range(0, len(x1)):
        for j in range(0, len(x2)):
            point = [x2[j, j], x1[i, i]]
            grid[i, j] = (f(point))
    return grid


def Modelplot(f, intx, inty):
    x1_min = intx[0]
    x1_max = intx[1]
    x2_min = inty[0]
    x2_max = inty[1]

    x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2), np.arange(x2_min, x2_max, 0.2))

    fig = plt.figure(figsize=(10, 7))

    y = fdisc(f, x1, x2)
    plt.imshow(y, extent=[x1_min, x1_max, x2_min, x2_max], cmap=cm.jet, origin='lower')
    plt.colorbar()
    plt.title("First dimension of the output", fontsize=8)
    plt.show()

class CustomDataset(Dataset):
    def __init__(self, root='./data', train=True, transform=None, target_transform=None):
        if (train):
            self.directory = root + '/train'
        else:
            self.directory = root + '/test'

        self.inputs = (np.load(self.directory+'/inputs.npy')).astype(np.float32)
        self.labels = (np.load(self.directory+'/labels.npy'))

        if (len(self.inputs[0].shape) == 1):
            self.Data_type = "Vectors"
        else:
            self.Data_type = "Images"
        self.transform = transform
        self.target_transform = target_transform
        # # if (self.Data_type == "Images"):
        # #     #self.Transf =
        # # else:
        # #     #self.Transf =

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class CustomImageDataset(Dataset):
    def __init__(self, root='./data/Images', train=True, transform=None, target_transform=None):
        if (train):
            self.img_labels = pd.read_csv(root + '/train/labels.csv')
            self.img_dir = root + '/train/images'
        else:
            self.img_labels = pd.read_csv(root + '/test/labels.csv')
            self.img_dir = root + '/test/images'

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = io.imread(img_path)
        if (image.shape[2] == 4):
            # remove transparence_for_png
            image = (image[:, :, :3])
        label = (int(self.img_labels.iloc[idx, 1]))
        # label = torch.tensor(int(self.img_labels.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class CustomVectorDataset(Dataset):
    def __init__(self, root='./data/Vectors', train=True, target_transform=None):

        if (train):
            self.vect_labels = ((np.loadtxt(root + '/train/labels.csv', delimiter=",", dtype=np.int64)))
            # self.vect_labels =  torch.from_numpy((np.loadtxt(root + '/train/labels.csv',delimiter = ",", dtype = np.int64)))
            self.vect = tr.from_numpy(np.loadtxt(root + '/train/vectors.csv', delimiter=",", dtype=np.float32))

        else:
            self.vect_labels = ((np.loadtxt(root + '/test/labels.csv', delimiter=",", dtype=np.int64)))
            self.vect = tr.from_numpy(np.loadtxt(root + '/test/vectors.csv', delimiter=",", dtype=np.float32))

        self.target_transform = target_transform

    def __len__(self):
        return len(self.vect_labels)

    # def Len(self):
    #   return print(self.img_labels)

    def __getitem__(self, idx):
        return self.vect[idx], self.vect_labels[idx].item()