from collections import Counter #for counting hashable classes
from collections import OrderedDict #for keeping order entry

from config_modified import qtas_numbers # total 50 numbers from 1 to 50 as class labels

from random import sample
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

import csv
import matplotlib.pyplot as plt
import numpy as np

def base_original(file_name, full_size, start_size):
    # reads the database
    reader = csv.reader(open(file_name, "r"), delimiter=",")
    x = list(reader)
    base = np.array(x)

    # separates feature set from X and output
    X = base[:, 1:]
    y = base[:, 0] #contains the random class labels from 1 to 50 

 #we have class labels from 1 to 50 or a vector of integers 
    y = y.astype(int)

    # draws the classes labels 1 to 50 that will be used for training
    training_number = sample(range(0, full_size), k=start_size)

    return separates_base(X, y, training_number)

def calculate_delta(Ct, Cr, Ce):
    return 1/2 * (1 - np.sqrt(2 * Ct / (Cr + Ce)))

def calculate_F1_score(y, y_hat):
    return f1_score(y, y_hat, average='micro')  

def plot_PCA(X, y, EVs_X, EVs_y):
    X_reduced = PCA(n_components=2).fit_transform(X) #reducing the feature set into lower dimensionality space using single value decomposition
    EVs_X_reduced = PCA(n_components=2).fit_transform(EVs_X) #reducing the extreme vectors into lower dimensionality space using single value decomposition

    x_min, x_max = X_reduced[:, 0].min() - .5, X_reduced[:, 0].max() + .5
    y_min, y_max = X_reduced[:, 1].min() - .5, X_reduced[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf() #clear the current figure

    # Plot the training points
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k')
    plt.scatter(EVs_X_reduced[:, 0], EVs_X_reduced[:, 1], c=EVs_y, edgecolor='k', 
        marker="s")    
    plt.xlabel('1a component')
    plt.ylabel('2a component')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    plt.show()

def next_samples(X, y, classes_available):
    class_drawn = sample(classes_available, 1)

    return separates_base(X, y, class_drawn)

def separates_base(X, y, training_number):
    # draws 80% of the sample whose class labels(1 to 50) are drawn for
    # training stage
    samples_sorted_classes = np.argwhere(np.isin(y, training_number).reshape(-1)).reshape(-1)#checking whether the training_numbers are present in y or not
    training_examples = sample(samples_sorted_classes.tolist(), k=(round(samples_sorted_classes.size * 0.8)))#coverting the above array into list and drawing 80% as training set

    # obtaining the training database
    X_train = X[training_examples]
    y_train = y[training_examples]

    # obtain the test samples of the classes drawn
    #making a set difference and then converting it into list (string type)  then converting it into integer array
    test_samples = np.array( list(set(samples_sorted_classes).difference(set(training_examples))),dtype=int)
    # obtaing the training database
    X_test = X[test_samples]
    y_test = y[test_samples]

    # remove from original base the samples of the classes drawn
    np.delete(X, samples_sorted_classes, 0)
    np.delete(y, samples_sorted_classes, 0) 

    return (X_train, y_train, X_test, y_test, X, y, set(y_train))        

def VR(y, EVs_y):
    # the frequency of occurrence of each class of the original sample and the
    # extreme values
    frequency_original = Counter(y)
    frequency_EV = Counter(EVs_y)

    VRs = {}

    # for each number(1 to 50) calculate VR
    for number in range(0, qtas_numbers):
        chave = number + 1 # the class label, loop running from 0 to 49 but class labels are from 1 to 50
       # so as class labels are from 1 to 50 and y was a vector of integers from 1 to 50 so consider chave
        if frequency_original[chave] == 0:
            VRs[chave] = '-'
        else:
            VRs[chave] = round(frequency_EV[chave] / frequency_original[chave], 2)
    
    # sort by class labels(1 to 50)
    VRs = OrderedDict(sorted(VRs.items()))

    print('VRs per number:')
    print(VRs)
    print('VR general: ', round(len(EVs_y) / len(y), 2))