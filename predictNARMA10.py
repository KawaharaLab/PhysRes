#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysRes
Physical-hysteretic Reservoir

See:
C. Caremel, Y. Kawahara, K. Nakajima, Hysteretic reservoir, Physical Review Applied 22 (6), 064045, 2024.

@author: cc
"""

import numpy as np
import matplotlib.pyplot as plt
import prc
from utils import computeEr

#%%Settings

N = 1000 #Reservoir size (number of nodes)

alpha = 1 #Alpha (scaling hyperparam, type: float)
lamda = 1 #latency (type: int). Note: "lambda" is already taken in python builtins (the nonymous lambda function)

tau = 1 #Tau (state delay, generally set to 1, type: int)
x0 = 0.1  #Initial condition, any will do e.g. 0, 0.05...

#%%NARMA10 Input

L = 10000 #Number of points (length of timeseries input) 
zeta = np.random.uniform(-1,1,size=L)
s = 0.45
u = s*0.5*(zeta+1)    #u in [0,sigma]
Y = np.zeros(L)
for t in range(1,L):
    Y[t] = 0.3*Y[t-1] + 0.05*Y[t-1]*np.sum(Y[t-10:t]) + 1.5*u[t-1]*u[t-10] + 0.1

#%%States

physres = prc.PHysRes(u, N, x0, lamda, alpha, tau) #Instantiate
x_hysres = physres.Run() #Run

#%%State matrix

# cut = 30 #Display first values only

# fig, ax = plt.subplots()
# plt.imshow( x_hysres[:cut,:cut] )
# plt.xlabel("Nodes (length N)")
# plt.ylabel("Steps (length L)")
# plt.title(f"State matrix (α={alpha}, λ={lamda})")
# plt.colorbar()

# ax.set_xticks(np.arange(-.5, x_hysres[:cut,:cut].shape[1], 1), minor=True)
# ax.set_yticks(np.arange(-.5, x_hysres[:cut,:cut].shape[0], 1), minor=True)
# ax.grid(which='minor', color='white', linestyle='-', linewidth=.5)

#%%Test states

Y_test, Ypred, split = physres.Test(0.7, Y)

wo = 100 #washout
nrmse = computeEr(Y_test[wo:], Ypred[wo:], 'NRMSE')
print(f'NRMSE after first {wo} points:', nrmse)

fig, ax = plt.subplots()
plt.plot( u[split:][wo:], linewidth=.1, c='k', label='Input')
plt.plot( Y_test[wo:], linewidth=.5, c='r', label='Target Y')
plt.plot( Ypred[wo:], linewidth=.5, linestyle="--", c='b', label='Prediction')
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Y")
plt.title(f"NRMSE={np.round(nrmse,5)}")
