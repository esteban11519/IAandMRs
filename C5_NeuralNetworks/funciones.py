from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def softmax1(A):
    expA = np.exp(A)
    return expA / expA.sum()

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()



def cost(T, Y):
    return -(T*np.log(Y)).sum()


def cost2(T, Y):
    # Lo mismo que la función cost() pero optimizada
    # Entropia cruzada media.
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

#Error
def error_rate(targets, predictions):
    return np.mean(targets != predictions)

#One - hot matriz
def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def getData():
    # El tamaño es de  28x28 = 784 (Vectores)
    tam_imag  = 28
    num_labels = 10 #Numero de variables de salida 
    pixels = tam_imag * tam_imag 
    train_data = np.loadtxt("mnist_train.csv", 
                            delimiter=",")
    test_data = np.loadtxt("mnist_test.csv", 
                           delimiter=",") 
    #Se usa esto para evitar valores de 0 y que la red no aprenda 
    fac = 0.99 / 255
    Xtrain   = np.asfarray(train_data[:, 1:]) * fac + 0.01 #Se deja los valores entre 0.01 y 0.99
    Ytrain = np.array(train_data[:, :1], dtype= int ).flatten()

    Xvalid   = np.asfarray(test_data[:, 1:]) * fac + 0.01
    Yvalid   = np.array(test_data[:, :1] , dtype= int).flatten()

    return Xtrain, Ytrain, Xvalid, Yvalid


