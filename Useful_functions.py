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
    #print((c_matrix).shape,T)
    for i in range(0,len(c_matrix)):
        ij_max = len(c_matrix[i])
        p = 1
        if (len(c_matrix.shape) == 3):
            for j in range(1,ij_max+1):
                for k in range(1,ij_max+1):
                    #print(c_matrix[i][python_index(j),python_index(k)])
                    #print(mt.exp(-1/2*((c_matrix[i][python_index(j),python_index(k)])**2)))
                    p*= mt.exp(-1/2*((c_matrix[i][python_index(j),python_index(k)])**2))
                    #p *= mt.exp(-1/2*((c_matrix[python_index(i),python_index(j)])**2)*T/((i)**2+(j)**2))
        ps += [p]
    return ps


def imshow(img1,img2):
    img1 = img1 / 2 + 0.5     # unnormalize
    npimg1 = img1.numpy()
    img2 = img2 / 2 + 0.5     # unnormalize
    npimg2 = img2.numpy()
    #plt.imshow(npimg)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.transpose(npimg1[:,:,:], ( 2,1,0)))
    axs[1].imshow(np.transpose(npimg2[:,:,:], ( 2,1,0)))
    plt.show()
