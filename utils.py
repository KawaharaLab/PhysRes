#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cc
"""

import numpy as np
import sys

def normalize(a, min_=0,max_=1):
    xmax, xmin = np.nanmax(a), np.nanmin(a)
    return (max_-min_)*(a - xmin)/(xmax - xmin) + min_

def computeEr(target, pred, er_type='NRMSE'):
    if er_type == 'NRMSE':
        return np.sqrt( np.mean(np.square(target-pred))/np.var(target) ) # sd squared

    if er_type == 'NMSE':
        return np.square(np.linalg.norm(pred-target)) / np.square(np.linalg.norm(target)) # Squared L2 norm

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    block = int(round(barLength*progress))
    text = "\rCompleted: [{0}] {1:.2f}%".format( "="*block + " "*(barLength-block), progress*100)
    sys.stdout.write(text)
    sys.stdout.flush()
    if progress*100>=100: print("\n")
