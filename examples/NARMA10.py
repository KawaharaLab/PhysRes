#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysRes
Physical-hysteretic Reservoir

See:
C. Caremel, Y. Kawahara, K. Nakajima, Hysteretic reservoir, Physical Review Applied 22 (6), 064045, 2024.

"""

import numpy as np

np.random.seed(42)

import matplotlib.pyplot as plt

from prc import model
from scripts import utils

#%% Settings

N = 1000  #Reservoir size (number of nodes)

alpha = 1  #Alpha (scaling hyperparam, type: float)
lamda = 1  #latency (type: int). Note: "lambda" is already taken in python builtins (the nonymous lambda function)

tau = 1  #Tau (state delay, generally set to 1, type: int)
x0 = 0.1  #Initial condition, any will do e.g. 0, 0.05...

#%% NARMA10 Input

L = 10000  #Number of points (length of timeseries input)
zeta = np.random.uniform(-1, 1, size=L)
s = 0.45
u = s * 0.5 * (zeta + 1)  #u in [0,sigma]
u = u.reshape(-1, 1)
Y = np.zeros(L)
for t in range(1, L):
    Y[t] = 0.3 * Y[t - 1] + 0.05 * Y[t - 1] * np.sum(Y[t - 10:t]) + 1.5 * u[t - 1] * u[t - 10] + 0.1

#%% Init

physres = model.PhysRes(u, N, x0, lamda, alpha, tau)  #Instantiate

#%% States
X = physres.Run(normalization="minmax", save_operand=False)  #Run

#%% Test states

Y_test, Ypred, split = physres.TrainTest(0.7, Y)

wo = 100  #washout
nrmse = utils.computeEr(Y_test[wo:], Ypred[wo:], 'NRMSE')
print(f'NRMSE after first {wo} points:', nrmse)

fig, ax = plt.subplots()
plt.plot(physres.X_test[wo], linewidth=.1, c='k', label='Input')
plt.plot(Y_test[wo:], linewidth=.5, c='r', label='Target Y')
plt.plot(Ypred[wo:], linewidth=.5, linestyle="--", c='b', label='Prediction')
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Y")
plt.title(f"NRMSE={np.round(nrmse,5)}")

#%% Alternative approach.
#Perhaps our dataset is already split, or we first want a trained model, save it, and test the model separately.
#Then do:

split = int(L * 0.7)  #desired ratio training/total
u_train, u_test = u[:split], u[split:]
Y_train, Y_test = Y[:split], Y[split:]

#Here we define the same Win matrix in optional argument for both the training and test phase
Win = np.random.uniform(0, 0.5, size=(u.shape[1], N))

physres_collect_train = model.PhysRes(u_train, N, x0, lamda, alpha, tau, Win=Win)
X_train = physres_collect_train.Run()
Ypred_train, Wout = physres_collect_train.Train(Y_train)
print(utils.computeEr(Y_train[wo:], Ypred_train[wo:], 'NRMSE'))

physres_collect_test = model.PhysRes(u_test, N, x0, lamda, alpha, tau, Win=Win)
X_test = physres_collect_test.Run()
Ypred_test = physres_collect_test.Test(Wout)
print(utils.computeEr(Y_test[wo:], Ypred_test[wo:], 'NRMSE'))
