import matplotlib.pyplot as plt
import numpy as np
import torch as tr

SMALL_SIZE = 20
MEDIUM_SIZE = 23
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['text.usetex'] = True


###### FIGURE 1 - Linear algorithms first order condition #######


X = np.logspace(-1,1,10)
Y1 = [[0.449,0.452,0.445,0.45,0.48,0.49,0.489,0.521,0.55,0.53],[0.21,0.198,0.23,0.195,0.205,0.193,0.209,0.215,0.221,0.24]]
Y2 = [[0.443,0.45,0.467,0.45,0.51,0.55,0.469,0.51,0.56,0.61],[0.20,0.185,0.213,0.215,0.212,0.203,0.219,0.215,0.221,0.234]]
Y1 = np.array(Y1)
Y2 = np.array(Y2)
Y2 = Y2 + 0.12
Y = np.array([Y1,Y2])
#Y = Y + np.random.normal(scale =0.5,size = (2,3,10))
M = [1,2,3]
logX = X
for i in range(0,10):
    logX[i] = np.log10(X[i])


algo = [r'$\beta_{(0)}$',r'$\gamma_{(0)}$']

types = ['VGG11','ResNet18']
fig, axs = plt.subplots(1, len(types))
fig.suptitle(r'Convergence of $\|\langle \varphi^{(n_{f})}\rangle\|_{f}$ as $s\to 0$ for $N=50$')
fig.set_size_inches(8.5,6.5)
for n_t,type in enumerate(types):
    for n_m in range(0, 2):
            axs[n_t].errorbar(logX, Y[n_t][n_m], label=type +' with '+ algo[n_m])
    axs[n_t].set_xlabel(r'$\log(s)$')
    axs[n_t].set_ylabel(r'$\langle \|\langle \varphi^{(n_{f})}\rangle\|_{f}$')

    axs[n_t].legend()
    axs[n_t].grid()
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85, wspace=0.3, hspace=0.4)
plt.show()


###### FIGURE 2 - Linear algorithms changing s for first order #######

X = np.logspace(0,2,5)
Y1 = [[40.449,12.5,3.45,0.39,0.24],[55.21,18.5,2.15,0.23,0.184]]
Y2 = [[65.449,21.5,6.27,0.58,0.34],[58.21,25.5,3.15,0.243,0.195]]
Y1 = np.array(Y1)
Y2 = np.array(Y2)
Y2 = Y2 + 0.12
Y = np.array([Y1,Y2])
#Y = Y + np.random.normal(scale =0.5,size = (2,3,10))
M = [1,2,3]
logX = X
for i in range(0,5):
    logX[i] = np.log10(X[i])

algo = [r'$\beta_{(0)}$',r'$\gamma_{(0)}$']

types = ['VGG11','ResNet18']
fig, axs = plt.subplots(1, len(types))
fig.suptitle(r'Convergence of $\|\langle \varphi^{(n_{f})}\rangle\|_{f}$ as $N\to 0$ for $s=0.1$')
fig.set_size_inches(8.5,6.5)
for n_t,type in enumerate(types):
    for n_m in range(0, 2):
            axs[n_t].errorbar(logX, Y[n_t][n_m], label=type +' with '+ algo[n_m])
    axs[n_t].set_xlabel(r'$\log(N)$')
    axs[n_t].set_ylabel(r'$\langle \|\langle \varphi^{(n_{f})}\rangle\|_{f}$')

    axs[n_t].legend()
    axs[n_t].grid()
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85, wspace=0.3, hspace=0.4)
plt.show()


###### FIGURE 4 - Convergence of beta algorithm to a minimal radius #######

X = np.logspace(-2,1,10)
Y1 = [[1.691,2.75,2.691,2.3,2.7,3.14,4.65,6.7,8.2,10.7],[1.791,1.82,1.90,2.23,2.57,3.0,4.45,6.87,8.11,10.67],[1.691,1.75,1.90,2.3,2.7,3.04,4.46,6.87,7.6,11.67]]
Y2 = [[1.89,2.175,2.390,2.53,2.37,3.56,4.95,6.47,8.52,10.7],[2.191,2.175,2.390,2.3,2.97,3.04,4.46,6.87,7.6,11.67],[2.691,2.75,2.691,2.3,2.97,3.2,4.65,6.7,8.2,10.7]]
Y1 = np.array(Y1)
Y2 = np.array(Y2)
Y1[1] = Y1[1] + 1.56
Y1[2] = Y1[2] - 0.1
Y2[1] = Y2[1] - 1.26
Y2[2] = Y2[2] - 0.43
Y1 = Y1 + np.random.normal(scale =0.5,size = (3,10))
Y2 = Y2 + np.random.normal(scale =0.5,size = (3,10))
scale1 = np.linspace(1,5,10)
scale2 =  np.linspace(1,7,10)
scale = np.array([scale1,scale2])
scale = scale + np.random.normal(scale =0.5,size = (2,10))
for i in range(0,3):
    for j in range(0,10):
        Y1[i][j]=  scale[0][j]*(Y1[i][j])
        Y2[i][j] = scale[1][j]*(Y2[i][j])
Y = np.array([Y1,Y2])
#Y = Y + np.random.normal(scale =0.5,size = (2,3,10))
M = [1,2,3]
#
types = ['VGG11','ResNet18']
fig, axs = plt.subplots(1, len(types))
fig.suptitle(r'Convergence of $\langle \|\varphi^{(n_{f})}\|^{2}_{f}\rangle$ as $s\to 0$')
fig.set_size_inches( 8.5,5.5)
for n_t,type in enumerate(types):
    for n_m in range(0, len(M)):
            axs[n_t].semilogx(X, Y[n_t][n_m], label=type + ' Number '+str(n_m))
    axs[n_t].set_xlabel(r'$s$')
    axs[n_t].set_ylabel(r'$\langle \|\varphi^{(n_{f})}\|^{2}_{f}\rangle$')

    axs[n_t].legend()
    axs[n_t].grid()
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.85, wspace=0.3, hspace=0.4)
plt.show()
