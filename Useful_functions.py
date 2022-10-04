import math as mt
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

def python_index(i):
    return i-1

def intuitive_index(i):
    return i+1

def likelyhood_of_diffeomorphisms(c_matrix,T):
    ps = []
    for i in range(0,len(c_matrix)):
        ij_max = len(c_matrix[i])
        p = 1
        if (len(c_matrix.shape) == 3):
            for i in range(1,ij_max+1):
                for j in range(1,ij_max+1):
                    p*= mt.exp(-1/2*((c_matrix[i][python_index(i),python_index(j)])**2)*T/((i)**2+(j)**2))
                    #p *= mt.exp(-1/2*((c_matrix[python_index(i),python_index(j)])**2)*T/((i)**2+(j)**2))
        ps += [p]
    return ps


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
