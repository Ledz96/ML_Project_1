# -*- coding: utf-8 -*-
"""Utils functions for KNN"""

import numpy as np
from utils_predictions_manipulation import*

def get_KNN(Xtest,Xtrain,Ytrain,dist_func, k_n):
    """ input are extended X matrices, returns ypredictions array for Xtest using k closest distances
    calculated with given dist_func Given X's have been OHE and scales and y predictions are given as 0/1"""
    ypredarray=np.zeros(Xtest.shape[0])
    
    ret = np.empty((Xtest.shape[0],k_n))
    
    for j in range(Xtest.shape[0]): #prediction for j'th row of Xtest
        rowdistances=np.zeros(Xtrain.shape[0])
        #for i in range(Xtrain.shape[0]):   #i denotes index of row of Xtrain
        
        rowdistances=dist_manhattan_broadcast(Xtrain, Xtest[j])  #computing distance between test point and each Xtrain row
        
        sortindex=np.argsort(rowdistances) #orders neighbor rows for closest distances 
        indexn=sortindex[:k_n]
        ret[j] = indexn.reshape(1,-1)
            
    return ret

def dist_euclidean(row1, row2):
    """Euclidean distance between two vectors"""
    r1=row1[1:] #takes features after column of 1
    r2=row2[1:]
    dist=np.linalg.norm(r1-rb)
    return dist

def dist_manhattan(row1, row2):
    """Manhattan distance between two vectors"""
    r1=row1[1:] #takes features after column of 1
    r2=row2[1:]
    dist=np.sum(np.abs(r1-r2))
    return dist

def dist_manhattan_broadcast(matrix, row):
    """Manhattan distance a vector and a matrix (N_rows vectors)"""
    matrix = matrix[:,1:]
    row = row[1:]
    
    dist=np.sum(np.abs(matrix-row), axis=1)
    return dist

def get_prediction_from_knn(knn, k, y):
    """Given a k, y, and a list of nn, returns a prediction"""
    
    k_neighbors_vals = y[knn[:,:k].astype(int)]
    k_neighbors_mean = k_neighbors_vals.mean(axis=1)
    return probability_to_prediction(k_neighbors_mean)