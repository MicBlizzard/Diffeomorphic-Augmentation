import math
import random
#from PIL import Image

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import matplotlib
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

# pylint: disable=no-member, invalid-name, line-too-long
"""
Computes diffeomorphism of 2D images in pytorch
"""
@functools.lru_cache()
def scalar_field_modes(n, m, dtype=torch.float64, device='cpu'):
    """
    sqrt(1 / Energy per mode) and the modes
    """
    x = torch.linspace(0, 1, n, dtype=dtype, device=device)
    k = torch.arange(1, m + 1, dtype=dtype, device=device)
    i, j = torch.meshgrid(k, k)
    r = (i.pow(2) + j.pow(2)).sqrt()
    e = (r < m + 0.5) / r
    s = torch.sin(math.pi * x[:, None] * k[None, :])
    return e, s


def scalar_field(n, m,C, device='cpu'):
    """
    random scalar field of size nxn made of the first m modes
    """
    e, s = scalar_field_modes(n, m, dtype=torch.get_default_dtype(), device=device)
    c = C * e
    #print("the c field is ", c)
    return torch.einsum('ij,xi,yj->yx', c, s, s)

def deform(image, T, cut,C, interp='linear'):
    """
    1. Sample a displacement field tau: R2 -> R2, using tempertature `T` and cutoff `cut`
    2. Apply tau to `image`
    :param img Tensor: square image(s) [..., y, x]
    :param T float: temperature
    :param cut int: high frequency cutoff
    """
    n = image.shape[-1]
    assert image.shape[-2] == n, 'Image(s) should be square.'
    device = image.device.type

    # Sample dx, dy
    # u, v are defined in [0, 1]^2
    # dx, dx are defined in [0, n]^2
    u = scalar_field(n, cut,C, device)  # [n,n]
    v = scalar_field(n, cut,C, device)  # [n,n]
    dx = T ** 0.5 * u * n
    dy = T ** 0.5 * v * n

    # Apply tau
    return remap(image, dx, dy, interp)


def remap(a, dx, dy, interp):
    """
    :param a: Tensor of shape [..., y, x]
    :param dx: Tensor of shape [y, x]
    :param dy: Tensor of shape [y, x]
    :param interp: interpolation method
    """
    n, m = a.shape[-2:]
    assert dx.shape == (n, m) and dy.shape == (n, m), 'Image(s) and displacement fields shapes should match.'

    y, x = torch.meshgrid(torch.arange(n, dtype=dx.dtype), torch.arange(m, dtype=dx.dtype))

    xn = (x - dx).clamp(0, m-1)
    yn = (y - dy).clamp(0, n-1)

    if interp == 'linear':
        xf = xn.floor().long()
        yf = yn.floor().long()
        xc = xn.ceil().long()
        yc = yn.ceil().long()

        xv = xn - xf
        yv = yn - yf

        return (1-yv)*(1-xv)*a[..., yf, xf] + (1-yv)*xv*a[..., yf, xc] + yv*(1-xv)*a[..., yc, xf] + yv*xv*a[..., yc, xc]

    if interp == 'gaussian':
        # can be implemented more efficiently by adding a cutoff to the Gaussian
        sigma = 0.4715

        dx = (xn[:, :, None, None] - x)
        dy = (yn[:, :, None, None] - y)

        c = (-dx**2 - dy**2).div(2 * sigma**2).exp()
        c = c / c.sum([2, 3], keepdim=True)

        return (c * a[..., None, None, :, :]).sum([-1, -2])

def temperature_range(n, cut):
    """
    Define the range of allowed temperature
    for given image size and cut.
    """
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    T1 = 1 / (math.pi * n ** 2 * log)
    T2 = 4 / (math.pi ** 3 * cut ** 2 * log)
    return T1, T2

def typical_displacement(T, cut, n):
    if isinstance(cut, (float, int)):
        log = math.log(cut)
    else:
        log = cut.log()
    return n * (math.pi * T * log) ** .5 / 2

def TransVectors(X,temperature,cut_off, parameters, diffeo):
    diffeo = torch.tensor(diffeo, dtype = torch.float32)
    X_new = torch.clone(X)
    norms = []
    for i in range(0,len(X_new)):
        for j in range(0,len(parameters[i])):
            X_new[i] += parameters[i,j]*diffeo[j]
        norms += [torch.norm(X[i] - X_new[i]).item()]
    return X_new, norms

def TransImages(X,T,cut,C_list):
    X_new = torch.clone(X)
    norms = []
    for i in range(0,X.shape[0]):
        X_new[i] =  deform(X[i], T, cut ,C_list[i], interp='linear')
        norms +=[torch.norm(X[i]-X_new[i]).item()]
    return X_new, norms

class Transformations:
    def __init__(self, temperature, cut_off, parameters ,Model,Data_type,dimension,index_of_discriminating_factor, diffeo):

        self.Model = Model
        self.Data_type = Data_type
        self.dimension = dimension
        self.index_of_discriminating_factor = index_of_discriminating_factor
        self.diffeo = diffeo

        self.temperature = temperature
        self.cut_off = cut_off
        self.parameters = parameters

    def Diffeomorphism(self, input):
        if (self.Data_type == "Images"):
            return TransImages(input, self.temperature, self.cut_off, self.parameters)
        else:
            return TransVectors(input, self.temperature,self.cut_off, self.parameters, self.diffeo)

    def Noise(self, input, norms):
        X_new = torch.clone(input)
        delta = torch.rand(input.shape)
        for i in range(0,len(input)):
            delta[i] *= norms[i]/torch.norm(delta[i])
            X_new[i] += delta[i]
        return X_new, norms

def Diffeo_shape(diffeo_shape,cut_off):
    diffeo_shape = np.array(diffeo_shape)
    total_parameters = 0
    for i in range(0, len(diffeo_shape)):
        diffeo_shape[i] *= (cut_off / diffeo_shape[i])
        total_parameters += diffeo_shape[i]
    diffeo_shape = diffeo_shape.tolist()
    diffeo_shape = tuple(diffeo_shape)
    return diffeo_shape, total_parameters

def diffeo_distr(batch_size,diffeo_shape,device):
    total = 1
    for p in diffeo_shape:
        total *= p
    loc = torch.zeros(int(batch_size*total)).to(device)
    covariance_matrix = torch.eye(int(batch_size*total)).to(device)
    distr = torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix, precision_matrix=None, scale_tril=None, validate_args=None)
    return distr
#
def diffeo_parameters(distr,shape,diffeo_shape,device, temperature):
    para = distr.sample().to(device)
    para = torch.reshape(para, shape)
    #print("the para in the beginning of the loop are ",para)
    if(len(diffeo_shape)==2):
        for p in para:
            for i in range(0, diffeo_shape[0]):
                for j in range(0, diffeo_shape[1]):
                    p[i, j] *= mt.sqrt(temperature / ((i + 1) ** 2 + (j + 1) ** 2))
    else:
        para = torch.zeros(shape)
        #print("the para in the end of the loop are ",para)
    return para
# class Vector_Transformations(Transformations):
#
#     def __init__(self, Temperature, cut_off, parameters, unit_vectors):
#         self.unit_vectors = unit_vectors
#         super(Vector_Transformations, self).__init__(Temperature, cut_off, parameters)
#
#     def diffeo(self, vectors):
#         dimension = len(self.unit_vectors)
#         print("the dimension of the space is ", dimension)
#         print(self.unit_vectors)
#         sub_space_dimension = int(np.sum(self.unit_vectors))
#         print("the dimension of the diffeo subspace is ",sub_space_dimension)
#         initial_diffeo_vector = torch.zeros(dimension)
#         initial_temp_diffeo_vector = torch.rand(sub_space_dimension)
#         n = torch.norm(initial_temp_diffeo_vector,p='fro')
#         print("initial vectors ",initial_diffeo_vector,initial_temp_diffeo_vector)
#         print("the norm is ", n)
#         index = 0
#         for i in range(0,dimension):
#             if (self.unit_vectors[i] > 0):
#                 initial_diffeo_vector[i] = initial_temp_diffeo_vector[index]
#                 index += 1
#         print("final diffeo ", initial_diffeo_vector/n.detach())
#         out = torch.clone(vectors)
#         norms = []
#         # for i in range(len(vectors)):
#         #     out[i] = torch.add(vectors[i], torch.mul(self.parameters[i], self.unit_vector))
#         #     norms += [torch.norm(torch.mul(self.parameters[i], self.unit_vector), p='fro').item()]
#         # # out = torch.add(vectors, torch.mul(self.parameters[i],unit_vector))
#         return out, norms
#
#     def Noise(self, vectors, norms): ### add a cut-off
#         out = torch.clone(vectors)
#         Norms = []
#         for i in range(len(vectors)):
#             temp_v = torch.rand(vectors[i].size(0))
#             delta = torch.mul(temp_v,
#                               norms[i] / torch.norm(temp_v, p='fro', dim=None, keepdim=False, out=None, dtype=None))
#             out[i] = vectors[i] + delta
#             Norms += [torch.norm(delta, p='fro').item()]
#         return out, Norms
#
# class Image_Transformations(Transformations):
#     def __init__(self, Temperature, cut_off, parameters):
#         super(Vector_Transformations, self).__init__(Temperature, cut_off, parameters)
#
#     def diffeo(self, image):
#         return Diffeomorphisms.Trans(image, self.Temperature, self.cut_off_frequency, self.parameters), 0
#
#     def Noise(self, images, norms):
#         return image
#
#
#
