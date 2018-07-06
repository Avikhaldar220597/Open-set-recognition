from config_modified import qtas_numbers
from config_modified import qtas_numbers_start
from config_modified import sigma
from config_modified import tau

from lib_modified import base_original
from lib_modified import calculate_delta
from lib_modified import calculate_F1_score
from lib_modified import plot_PCA
from lib_modified import next_samples
from lib_modified import VR

from evm_modified import test_EVM
from evm_modified import train_EVM

import numpy as np
import matplotlib.pyplot as plt

# obtaining and training the model with 30 random numbers betwen 1 and 50
(X_train, y_train, X_test, y_test, X_remaining, y_remaining, classes_known) = base_original("C:/Users/Avik/Desktop/projects/JU summers 2018/BasicCharacterTrain/dataset.csv", qtas_numbers, qtas_numbers_start)
(EVs_psi, EVs_X, EVs_y) = train_EVM(X_train, y_train, tau, sigma)

plot_PCA(X_train, y_train, EVs_X, EVs_y)
VR(y_train, EVs_y)

# all test classes participated in the training
Cr = qtas_numbers_start
Ct = qtas_numbers_start
Ce = qtas_numbers_start
delta = calculate_delta(Ct, Cr, Ce)

# classifies the test set samples 
y_hat = test_EVM(EVs_psi, EVs_X, EVs_y, X_test, delta)

print(calculate_F1_score(y_test, y_hat))

classes_available = set(range(0, qtas_numbers)).difference(classes_known)

# tests the models for the samples that are increamentally inserted
for number in range(0, qtas_numbers - qtas_numbers_start):
    # obtain the samples of the untrained classes to be tested
    (_, _, X_test_new, _, X_remaining, y_remaining, letter_drawn) = next_samples(X_remaining, y_remaining, classes_available)

    # remove the letter selected from the set of letters
    classes_available = classes_available.difference(letter_drawn)

    # samples that are not used in the trainig should not be classified as unknown(-1)
    y_test_new = - np.ones(X_test_new.shape[0])

    # tests all the test samples from the drawn classes to generate a F1 score value
    X_test = np.concatenate((X_test, X_test_new), axis=0)
    y_test = np.concatenate((y_test, y_test_new), axis=0)  

    # update variables as one more class inserted
    Cr += 1
    Ce +=1  
    delta = calculate_delta(Ct, Cr, Ce)

    # classifies the test sample
    y_hat = test_EVM(EVs_psi, EVs_X, EVs_y, X_test, delta)

    print(calculate_F1_score(y_test, y_hat))
    
    plt.scatter(y_test,y_hat)
    plt.scatter(EVs_X,EVs_y)

