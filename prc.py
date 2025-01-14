#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cc
"""

import numpy as np
from utils import normalize, update_progress

class PHysRes(object):
        
    def __init__(self, u, N, x0, lamda, alpha, tau):
        
        self.u, self.N, self.x0, self.lamda, self.alpha, self.tau = u, N, x0, lamda, alpha, tau
        self.L = len(u)
        self.Win = np.random.uniform(0, 0.5, size=(1, N))
        self.phi = np.linspace(0, 2 * np.pi, N)

    def Run(self, normalization="minmax", verbose=True ):
        
        x_hysres = np.zeros((self.L, self.N)) #Array initialization
        x_hysres[:self.tau] = self.x0
        self.normalization=normalization
        self.verbose=verbose
           
        for t in range(self.tau, self.L):
            
            if self.normalization=="sinewave": #Option 1. Normalization via sine wave
                operand = np.sin(self.phi) + np.asarray( np.dot(self.u[t], self.Win) + x_hysres[t-self.tau] )
            
            if self.normalization=="minmax": #Option 2. Normalization via min-max
                operand = normalize( np.asarray( np.dot(self.u[t], self.Win) + x_hysres[t-self.tau]), -1,1)
            
            x_hysres[t] = sigma(operand.reshape(-1), self.lamda, self.alpha)
            
            if self.verbose: update_progress(t/(self.L-self.tau))
        
        self.States = x_hysres

        return self.States
    
    def Test(self, split_r, Y):
        
        split = int(self.L*split_r) #desired ratio training/total
        
        self.States_train, self.States_test = self.States[:split], self.States[split:]
        self.Y_train, self.Y_test = Y[:split], Y[split:]
        
        self.Wout = np.dot(np.linalg.pinv( self.States_train), self.Y_train)  # Pseudo-inverse
        
        self.Ypred = np.dot(self.States_test, self.Wout)

        return self.Y_test, self.Ypred, split

def sigma(zeta, latency, scaling): #where the dynamics happen
    if latency==0: return fun(zeta, scaling) 
    else: return np.hstack((np.zeros(latency),fun(zeta,scaling)[:-latency]))
    
def fun(x, A): #Originally 1/(1+np.exp(-A*x)) but other elementary functions such as np.tanh(x) can be used
   return 1/(1+np.exp(-A*x))