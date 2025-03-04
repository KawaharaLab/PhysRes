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
from matplotlib.font_manager import FontProperties

font_prop = FontProperties(size=10)

from prc import model
from scripts import utils

import os

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
data_path = parent_directory + "/data/MNIST-dataset/"

#%% Settings
N = 10000  #Reservoir size (number of nodes)

alpha = 1  #Alpha (scaling hyperparam, type: float)
lamda = 1  #latency (type: int). Note: "lambda" is already taken in python builtins (the nonymous lambda function)

tau = 1  #Tau (state delay, generally set to 1, type: int)
x0 = 0.1  #Initial condition, any will do e.g. 0, 0.05...
# %%MNIST data

df_train = pd.read_csv(os.path.join(data_path, 'mnist_train.csv'))
df_test = pd.read_csv(os.path.join(data_path, 'mnist_test.csv'))

u_train = df_train.drop(columns='label')
u_test = df_test.drop(columns='label')
u_train = utils.normalize(u_train.to_numpy(), -1, 1)
u_test = utils.normalize(u_test.to_numpy(), -1, 1)

Y_train = np.asarray(df_train['label'])
Y_test = np.asarray(df_test['label'])

labels = np.unique(Y_test)
#Output dimension (length=10)
P = len(labels)
# array of one-hot label encodings
labels_one_hot = np.eye(P)

Y_train_multi = labels_one_hot[Y_train]
Y_test_multi = labels_one_hot[Y_test]

#%% Network

L = u_train.shape[0]
M = u_train.shape[1]  # length of the features (pixels) dimension (input size)

#Here we define the same Win matrix in optional argument for both the training and test phase
Win = np.random.uniform(0, 0.5, size=(M, N))

physres_collect_train = model.PhysRes(u_train, N, x0, lamda, alpha, tau, Win=Win)

X_train = physres_collect_train.Run()
Ypred_train_proba, Wout = physres_collect_train.Train(Y_train_multi)
#Not evaluated but possible:
#Ypred_train = np.argmax(Ypred_train_proba, axis=1)

physres_collect_test = model.PhysRes(u_test, N, x0, lamda, alpha, tau, Win=Win)
X_test = physres_collect_test.Run()
Ypred_test_proba = physres_collect_test.Test(Wout)
Ypred_test = np.argmax(Ypred_test_proba, axis=1)

#%% Accuracy

report = classification_report(Y_test, Ypred_test, target_names=labels.astype(str), digits=4)
print(report)

w_avg = utils.get_report(report)

cm = confusion_matrix(Y_test, Ypred_test)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    fmt='d',
    annot=True,
    square=True,
    cmap='Blues',
    vmin=0,
    vmax=np.max(cm),
    linewidths=0.5,
    linecolor='k',  # draw black grid lines
    cbar=True)  # disable colorbar

# re-enable outer spines
sns.despine(left=False, right=False, top=False, bottom=False)

plt.title(f'N={N}, test size={len(Y_test)}, weighted average accuracy = {w_avg}', fontproperties=font_prop)
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.tight_layout()

# plt.savefig('results_MNIST.png', bbox_inches='tight', dpi=300)

#%%Inspect

invalid_indices = np.where(Ypred_test != Y_test)[0]

if sum(invalid_indices) != 0:

    example_index = invalid_indices[9]  # Change this to plot different images
    example = df_test.iloc[example_index]

    # The label (first column) indicates the digit
    label = example[0]
    # The image data (remaining columns) represents the pixel intensity values
    image_data = example[1:].values

    # Reshape the image data to a 28x28 grid
    image = image_data.reshape(28, 28)

    # Plot the image
    plt.figure(figsize=(6, 5))
    plt.imshow(image, cmap='gray')
    plt.title(f'MNIST Digit: {label} / Predicted: {Ypred_test[example_index]}')
    plt.axis('off')
    plt.show()

    # plt.savefig('results_MNIST_invalid_example.png', bbox_inches='tight', dpi=300)
