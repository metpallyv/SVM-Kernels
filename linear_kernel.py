__author__ = 'Vardhaman'
import sys, getopt
import math
import csv
import math
import copy
import time
import numpy as np
from collections import Counter
from numpy import *
import matplotlib.pyplot as plt

def normalize(matrix,sd,me):
    with np.errstate(divide='ignore'):
        a = matrix
        sd_list = []
        mean_list = []
        if me == 0 and sd == 0:
            b = np.apply_along_axis(lambda x: (x-np.mean(x))/float(np.std(x)),0,a)
            tmp = a.shape[1]
            for i in range(tmp):
                sd_list.append(np.std(a[:,i]))
                mean_list.append(np.mean(a[:,i]))
            return b,sd_list,mean_list
        else:
            res = np.empty(shape=[a.shape[0],0])

            for i in range(a.shape[1]):
                col = matrix[:, i]
                mean_val = me[i]
                std_val = sd[i]
                b = np.apply_along_axis(lambda x: (x-mean_val)/float(std_val),0,col)
                res = np.concatenate((res, b), axis=1)
        res = np.nan_to_num(res)
    return res,sd,me

def load_csv(file):
    X = genfromtxt(file, delimiter=",",dtype=str)
    np.random.shuffle(X)
    return (X)

def random_numpy_array(ar):
    np.random.shuffle(ar)
    arr = ar
    return arr

def generate_set(X):
    X = X.astype(np.float)
    num_test = round(0.1*(X.shape[0]))
    start = 0
    end = num_test
    test_attri_list =[]
    test_class_names_list =[]
    training_attri_list = []
    training_class_names_list = []
    for i in range(10):
        X_test = X[start:end , :]
        tmp1 = X[:start, :]
        tmp2 = X[end:, :]
        X_training = np.concatenate((tmp1, tmp2), axis=0)
        y_training = X_training[:, -1]
        y_test = X_test[:, -1]
        X_training = X_training[:,:-1]
        X_test = X_test[:,:-1]
        X_training = np.matrix( X_training )
        X_test = np.matrix(X_test)
        X_training_normalized,sd,mean = normalize(X_training,0,0)
        X_training_normalized = np.nan_to_num(X_training_normalized)
        X_test_normalized,sd,mean = normalize(X_test,sd,mean)
        len1 = X_training_normalized.shape[0]
        len2 = X_test_normalized.shape[0]
        x_training_ones = np.ones(len1)
        x_training_ones = x_training_ones.reshape([x_training_ones.shape[0],1])
        x_test_ones = np.ones(len2)
        x_test_ones = x_test_ones.reshape([x_test_ones.shape[0],1])
        X_training_normalized = np.concatenate((x_training_ones,X_training_normalized),axis=1)
        X_test_normalized = np.concatenate((x_test_ones,X_test_normalized),axis=1)
        y_test = y_test.flatten()
        y_training = y_training.flatten()
        test_attri_list.append(X_test_normalized)
        test_class_names_list.append(y_test)
        training_attri_list.append(X_training_normalized)
        training_class_names_list.append(y_training)
        start = end
        end = end+num_test
    return test_attri_list,test_class_names_list,training_attri_list,training_class_names_list

def linear_kernel(X1,X2):
    return np.dot(X1,X2)

def train(X,Y):
    X_count = X.shape[0]
    alpha = np.zeros(X_count)
    flag = 1
    max_iterations = 1000
    for ite in range(max_iterations):

        for i in range(X.shape[0]):
            sum = 0
            for j in range(X.shape[0]):
                val= alpha[j] * Y[j] * linear_kernel(X[i],X[j])
                sum = sum + val
            if sum <= 0:
                sum = -1
            elif sum >0:
                sum = 1
            if Y[i] != sum:
                alpha[i] = alpha[i] + 1
    return alpha

def compute_efficiency(train_X,train_Y,test_X,test_Y,alpha):
    m = test_Y.size
    right = 0
    for i in range(m):
        s = 0
        for a, x_train,y_train  in zip(alpha, train_X,train_Y):
            s += a * y_train * linear_kernel(test_X[i],x_train)
        if s >0:
            s = 1
        elif s <=0:
            s = -1
        if test_Y[i] == s:
            right +=1

    print " Correct : ",right," Accuracy : ",right*100/test_X.shape[0]


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"f:t:e:")
    except getopt.GetoptError as error:
        print "Unknown input argument provided : "," Error : ",str(error)
        sys.exit(2)
    newfile = ""
    for opt,value in opts:
        if opt == "-f":
            newfile = value
    num_arr = load_csv(newfile)
    test_x,test_y,training_x,training_y = generate_set(num_arr)
    for i in range(len(training_x)):
        theta = train(training_x[i],training_y[i])
        compute_efficiency(training_x[i],training_y[i],test_x[i],test_y[i],theta)

if __name__ == "__main__":
    main(sys.argv[1:])
