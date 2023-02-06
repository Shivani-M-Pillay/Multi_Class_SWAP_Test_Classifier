from os import X_OK
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

from copy import deepcopy
import json
from sklearn.model_selection import train_test_split

import pennylane as qml
from pennylane.templates import AngleEmbedding, IQPEmbedding

from qiskit.quantum_info import Statevector

import random

from circuit_execution import *

# FUNCTIONS TO VERIFY THE CLASSIFICATION

# defines linear_kernel (amplitude encoding)
def linear_kernel(x,z,d=1):
    # normalize the two vectors
    x = x/np.linalg.norm(x)
    z = z/np.linalg.norm(z)
    return np.abs(np.inner(x,z))**(2*d)

# defines the cosine_kernel (rotation encoding)
def cosine_kernel(x,z):
    kernel = 1
    for i in range(len(x)):
        kernel *= np.abs(np.cos(x[i]-z[i]))**2
    return kernel

# classifies test point using R^3 vector, verify simulated results
def analytical_classification(test_input, X_train, y_train, y_test, label_vectors, kernel_func, P=0):
    classification = {}
    classification['Test Input'] = test_input.tolist()
    pred_vector = np.zeros([3,])

    for i in range(0,3):
        pred_vector[i] = 0 
        for m in range(len(X_train)):
            pred_vector[i] += (1/len(X_train))*kernel_func(test_input,X_train[m])*label_vectors[y_train[m]][i]
    if P==0: 
        classification["Predicted Vector"] = pred_vector.tolist()
        pred_class = max(np.unique(y_train),key = lambda x : np.dot(pred_vector,label_vectors[x]))
        print("Predicted Vector:",pred_vector)
    else:
        noisy_pred_vector = (1-P)*pred_vector
        classification["Noisy Predicted Vector"] = noisy_pred_vector.tolist()
        pred_class = max(np.unique(y_train),key = lambda x : np.dot(noisy_pred_vector,label_vectors[x]))
        print("Noisy Predicted Vector:",noisy_pred_vector)

    classification["True Class"] = int(y_test)
    classification["Predicted Class"] = int(pred_class)

    print("Predicted Class:",pred_class)
    print("True Class:",y_test)
    # returns the class where the inner product between the class' label state and the normalized predicted state is the highest
    return pred_class, classification

# for a given split of the training and test set, return the test accuracy 
def test_accuracy(X_train, X_test, y_train, y_test, label_vectors, classification_func, kernel_func=linear_kernel, encoding="amplitude", P=0, fold_num=0):
    correct = 0 
    classifications = []

    for i in range(X_test.shape[0]):
        print("Test Point:",X_test[i])
        if classification_func == analytical_classification:
            pred, classification = analytical_classification(X_test[i], X_train, y_train, y_test[i], label_vectors, kernel_func, P=P)
        elif classification_func == statevector_classification:
            training_circ = construct_fold_circuit(X_train,y_train,label_vectors,encoding,simulator="statevector",P=P)
            training_sv = Statevector.from_label("0"*training_circ.num_qubits)
            training_sv = execute_circuit_for_sv_sim(training_sv, training_circ)
            # the number of training points 
            num_training_points = len(X_train)
            # the number of index qubits needed in the index register to store each unique training point
            num_index_qubits = int(np.ceil((np.log2(num_training_points))))
            pred, classification = statevector_classification(X_test[i], y_test[i], training_sv, num_index_qubits, y_train, label_vectors, encoding, i, P=P)
        """
        else:
           pred = qasm_classification(X_test[i], X_train, y_train, y_test[i], label_vectors, encoding, P=P)
        """

        true = y_test[i]

        if(pred==true):
            correct += 1
            
        classifications.append(classification)

        # store all experiment data 
        experiment_data = [{"description":"Analytical Classification of Small XOR Dataset (3 features, 4 classes) using Linear Kernel, Fold "+str(fold_num),
                            "accuracy":float(correct/X_test.shape[0]),
                            "classifications":classifications}]
        
        
        filename = "xor_3d_linear_numerical_fold_test" + str(fold_num) + ".json"

        with open(filename,'w') as file:
            json.dump(experiment_data, file)

    return correct/X_test.shape[0], classifications

# performs 5-fold cross validation given a dataset and its labels
def cross_validation(X, y, label_vectors, classification_func, kernel_func=linear_kernel, encoding="amplitude", P=0):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    accuracies = 0 
    accuracies_list = []
    all_classifications = []
    
    fold_num = 1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        accuracy, fold_classifications = test_accuracy(X_train, X_test, y_train, y_test, label_vectors, classification_func, kernel_func, encoding, P, fold_num)
        all_classifications.append(fold_classifications)
        accuracies_list.append(accuracy)
        accuracies += accuracy
        fold_num += 1
    
    # store all experiment data 
    experiment_data = [{"description":"Numerical Classification of XOR Dataset (3 features, 4 classes) using Linear Kernel (All Folds)",
                        "accuracies":accuracies_list,
                        "classifications":all_classifications}]

    filename = 'xor_3d_linear_numerical_all_folds.json'

    with open(filename,'w') as file:
       json.dump(experiment_data, file)
    
    return accuracies/5
    
def leave_one_out_cross_validation(X, y, label_vectors, classification_func, kernel_func=linear_kernel, encoding="amplitude", P=0):
    correct = 0
    classifications = []

    for i in range(X.shape[0]):
        # pick test input 
        test_input = X[i]
        y_test = y[i]
        
        # remove test input and test input class from training set 
        X_train = np.delete(X,i,axis=0)
        y_train = np.delete(y,i,axis=0)

        if classification_func == analytical_classification:
            pred, classification = analytical_classification(test_input, X_train, y_train, y_test, label_vectors, kernel_func, P=P)
        elif classification_func == statevector_classification:
            training_circ = construct_fold_circuit(X_train,y_train,label_vectors,encoding,simulator="statevector",P=P)
            training_sv = Statevector.from_label("0"*training_circ.num_qubits)
            training_sv = execute_circuit_for_sv_sim(training_sv, training_circ)
            # the number of training points 
            num_training_points = len(X_train)
            # the number of index qubits needed in the index register to store each unique training point
            num_index_qubits = int(np.ceil((np.log2(num_training_points))))
            pred, classification = statevector_classification(test_input, y_test, training_sv, num_index_qubits, y_train, label_vectors, encoding, i, P=P)

        classifications.append(classification)

        if pred == y_test:
            correct += 1

        # store all experiment data 
        experiment_data = [{"description":"Analytical Classification of Small XOR Dataset (3 features, 4 classes) using Linear Kernel",
                            "accuracy":float(correct/X.shape[0]),
                            "classifications":classifications}]
        
        filename = "small_xor_3d_linear_analytical_p=0.09.json"

        with open(filename,'w') as file:
            json.dump(experiment_data, file)
        
    return correct/X.shape[0]


label_vectors = {0:np.array([0., 0., 1.]), 1:np.array([ 0.4714045, -0.8164966, -0.3333333]), 2:np.array([ 0.4714045,  0.8164966, -0.3333333]), 3:np.array([-0.942809 ,  0.       , -0.3333333])}

data = read_csv("small_3d_xor.csv")

X = np.array(list(zip(data['X'],data['Y'],data['Z'])))
y = np.array(data['Class'])

leave_one_out_cross_validation(X, y, label_vectors, analytical_classification, kernel_func=linear_kernel, encoding="amplitude", P=0.09)

""""
kf = KFold(n_splits=5, shuffle=True, random_state=0)

train_index = list(kf.split(X))[2][0]
test_index = list(kf.split(X))[2][1]

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

test_accuracy( label_vectors, statevector_classification, linear_kernel, encoding='amplitude', P=0, fold_num=3)
"""

"""
test_input = np.array([[-1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],[-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)]])
X_train = np.array([[-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)],[-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)], [1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],[0.5009694990086683, 0.8430950105267833, 0.19550029229606564],[0.59169366080681, 0.5475374174630182, 0.59169366080681],[0.59169366080681, 0.5475374174630182, 0.59169366080681]])
y_train = [0,1,2,3,0,0,0]
y_test = [0,1] # [0,1,2,3]
"""

"""
# XOR PROBLEM 

class_0 = np.array([1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)])
class_1 = np.array([-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)])

label_vectors = {0:class_0, 1:class_1}

accuracy = cross_validation(X, y, label_vectors, analytical_classification, kernel_func=linear_kernel, encoding="amplitude", P=0)

"""

"""
label_vectors = {0:np.array([0., 0., 1.]),1:np.array([ 0.4714045, -0.8164966, -0.3333333]),2:np.array([ 0.4714045,  0.8164966, -0.3333333]),3:np.array([-0.942809 ,  0.       , -0.3333333])}

test_input = np.array([[1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)],[-1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],[-1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)]])
X_train = np.array([[-1/np.sqrt(3),-1/np.sqrt(3),-1/np.sqrt(3)],[-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],[1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)], [1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)]])
y_train = [0,1,2,3]
y_test = [0,1,2,3]

print("ANALYTICAL CLASSIFICATION")
test_accuracy(X_train, test_input, y_train, y_test, label_vectors, analytical_classification, kernel_func=linear_kernel)
print("-"*20)

print("ANALYTICAL CLASSIFICATION WITH NOISE")
test_accuracy(X_train, test_input, y_train, y_test, label_vectors, analytical_classification, kernel_func=linear_kernel, P=0.1)
print("-"*20)

print("STATEVECTOR CLASSIFICATION")
test_accuracy(X_train, test_input, y_train, y_test, label_vectors, statevector_classification, encoding="amplitude")
print("-"*20)

print("STATEVECTOR CLASSIFICATION WITH NOISE")
test_accuracy(X_train, test_input, y_train, y_test, label_vectors, statevector_classification, encoding="amplitude", P=0.1)
print("-"*20)
"""

"""
# PIE DATASET
label_vectors = {0:[1,0,0], 1:[-0.5,0.866,0], 2:[-0.5,-0.866,0]}
#test_input =  np.array([[ 0.86,  0.84]])
#y_test =[0]
# y_train = [0,1,2]
# X_train =  np.array([[ 0.96,  0.74],[-0.78,  0.23], [-0.22,  0.87]])

# 0 2 1 2
# class 0
y_train = [0,2,1,2]

# class 0 
# test_input =  [ 0.2 ,  0.85]

# class 2
test_input = np.array([[-0.72,  0.89],[ 0.2 ,  0.85]])

# label_vectors = {0:np.array([1,0,0]), 1:np.array([-0.500011  ,  0.86601905,  0.        ]), 2:np.array([-0.500011  ,  -0.86601905,  0.        ])}

X_train =  np.array([[ 0.81,  0.75],[ 0.69, -0.1 ],[-0.55,  0.55],[-0.8 ,  0.77]])

y_test = [2,0]

print("ANALYTICAL CLASSIFICATION")
test_accuracy(X_train, test_input, y_train, y_test, label_vectors, analytical_classification, kernel_func=linear_kernel)
print("-"*20)

print("ANALYTICAL CLASSIFICATION WITH NOISE")
test_accuracy(X_train, test_input, y_train, y_test, label_vectors, analytical_classification, kernel_func=linear_kernel, P=0.1)
print("-"*20)

print("STATEVECTOR CLASSIFICATION")
test_accuracy(X_train, test_input, y_train, y_test, label_vectors, statevector_classification, encoding="amplitude")
print("-"*20)

print("STATEVECTOR CLASSIFICATION WITH NOISE")
test_accuracy(X_train, test_input, y_train, y_test, label_vectors, statevector_classification, encoding="amplitude", P=0.1)
print("-"*20)
"""