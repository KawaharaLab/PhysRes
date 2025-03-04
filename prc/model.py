#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysRes
Physical-hysteretic Reservoir

See:
C. Caremel, Y. Kawahara, K. Nakajima, Hysteretic reservoir, Physical Review Applied 22 (6), 064045, 2024.

"""

import numpy as np
from scripts.utils import normalize, update_progress
from tqdm import tqdm

class PhysRes(object):

    def __init__(self, u, N, x0, lamda, alpha, tau, Win=None, X=None):
        """
    Parameters:
        u (np.ndarray): Input series, represented as a vector of dimensions LxM, 
                        where L is the series length and M is the input dimension.
        N (int): Size of the reservoir, determining the number of internal units.
        x0 (np.ndarray): Initial conditions, given as a vector of dimension 1xL.
        lamda (int): Latency hyperparameter.
        alpha (float): Scaling hyperparameter.
        tau (int): State delay.
    """

        self.u, self.N, self.x0, self.lamda, self.alpha, self.tau = u, N, x0, lamda, alpha, tau
      
        self.L, self.M = u.shape[0], u.shape[1]
        
        #initialize the arrays
        if X is None: 
            self.X = np.zeros((self.L, self.N))
            self.X[:self.tau] = self.x0
        else: 
            self.X=X
        
        if Win is None: self.Win = np.random.uniform(0, 0.5, size=(self.M, self.N))
        else: self.Win=Win

    def Run(self, normalization="minmax", verbose=True, save_operand=False):
        """
        Description: 
            Here we collect the sub-states at each time step t for the input u, following our hysteretic encoding.
        Returns:
            The final state matrix X.
        """

        self.normalization = normalization
        self.verbose = verbose
        
        if save_operand: operand_arr = []
        
        if self.normalization == "sinewave": self.phi = np.linspace(0, 2 * np.pi, self.N)

        for t in range(self.tau, self.L):

            if self.normalization == "sinewave":  #Option 1. Normalization via sine wave (for physical implementation)
                operand = np.sin(self.phi) + np.asarray(np.dot(self.u[t], self.Win) + self.X[t - self.tau])

            if self.normalization == "minmax":  #Option 2. Normalization via min-max (better)
                operand = normalize(np.asarray(np.dot(self.u[t], self.Win) + self.X[t - self.tau]), -1, 1)

            self.X[t] = sigma(operand.reshape(-1), self.lamda, self.alpha)

            if save_operand: operand_arr.append(operand)

            if self.verbose:
                update_progress(t / (self.L - self.tau))

        if save_operand: self.operand_arr = operand_arr

        return self.X
        

    def TrainTest(self, split_r, Y):
        """
        Description: 
            Train and test together for the target Y.
            split_r is defined as the desired ratio training/total.
        Returns:
            Test target, prediction for the target, split (indices).
        """

        split = int(self.L * split_r)

        self.X_train, self.X_test = self.X[:split], self.X[split:]
        self.Y_train, self.Y_test = Y[:split], Y[split:]

        self.Wout, self.Ypred_test = linear_regression(self.X_train, self.Y_train, self.X_test)
        
        return self.Y_test, self.Ypred_test, split
    
    
    def Train(self,  Y_train):
        """
        Description: 
            Train only, for some target Y_train.
        Returns:
            Prediction for the training set, linear regressor Wout.
        """
        
        self.Y_train = Y_train

        self.Wout = np.dot(np.linalg.pinv(self.X), self.Y_train)  # Pseudo-inverse
        self.Ypred_train = np.dot(self.X, self.Wout)
        
        return self.Ypred_train, self.Wout
    
    
    def Test(self,  Wout):
        """
        Description: 
            Test only.
        Returns:
            Prediction based on the linear regressor Wout.
        """
                
        self.Wout = Wout

        self.Ypred_test = np.dot(self.X, self.Wout)

        return self.Ypred_test


def sigma(zeta, latency, scaling):  #where the dynamics happen
    if latency == 0:
        return fun(zeta, scaling)
    else:
        return np.hstack((np.zeros(latency), fun(zeta, scaling)[:-latency]))


def fun(x, A):  #Originally 1/(1+np.exp(-A*x)) but other elementary functions such as np.tanh(x) can be used
    return 1 / (1 + np.exp(-A * x))


def linear_regression(X_train, Y_train, X_test):

    # Initialize progress bar
    with tqdm(total=100, desc="Linear Regression", leave=True) as pbar:
        
        X_inv = np.linalg.pinv(X_train)
        pbar.update(40)  

        Wout = np.dot(X_inv, Y_train) 
        pbar.update(40)

        Ypred_test = np.dot(X_test, Wout)
        pbar.update(20)

    return Wout, Ypred_test
