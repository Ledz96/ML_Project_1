# -*- coding: utf-8 -*-
"""Utils functions for standardizing data and build poly"""

import numpy as np
import statistics
import itertools

def standardize(x):
    """Standardize the original data set ignoring column of 1."""
    xnew=x
    subx=x[:,1:]
    mean_x = np.mean(subx, axis=0)
    subx = subx - mean_x
    substd_x = np.std(subx, axis=0)
    subx = subx / substd_x
    xnew[:,1:]=subx
    return xnew, mean_x, substd_x

def build_poly_multi(tx, degree):
    """polynomial basis functions for input data tx, for all features expanded to d degree.Extended matrix is inputed np.array"""
    # ***************************************************
    # augmented x matrix with column of 1 input
    #list of indices for features modified
    #degrees of polynomial to apply to all features irrespectively
    # ***************************************************
    xmat=np.empty((tx.shape[0],degree*(tx.shape[1]-1)+1))
    xmat[:,0] = np.ones((tx.shape[0]))
    index = 1
    for i in range(1,tx.shape[1]):
        for d in range(1,degree+1):
            coltmp=tx[:,i]**d
            xmat[:,index] = coltmp
            index += 1
        
    return xmat

def build_poly_index(tx, index_list, degree):
    """polynomial basis functions for input data tx, for all features expanded to d degree.Extended matrix is inputed np.array. Index list includes all variables that will be expanded, mostly useful to avoid expanding categorical variables"""
    # ***************************************************
    # augmented x matrix with column of 1 input
    #list of indices for features modified
    #degrees of polynomial to apply to all features irrespectively
    # ***************************************************
    added_cols = sum([(len(degree[i])-1) for i in index_list])
    xmat=np.empty((tx.shape[0], tx.shape[1] + sum([(len(degree[i])-1) for i in index_list])))
    ind = 0
    for i in range(tx.shape[1]):
        if i in index_list:
            #print("BP:", i, degree[i])
            for d in degree[i]:
                if d<1:
                    coltmp=(np.abs(tx[:,i])**d)*np.sign(tx[:,i])
                    xmat[:,ind] = coltmp
                    ind+=1
                else:
                    coltmp=tx[:,i]**d
                    xmat[:,ind] = coltmp
                    ind+=1
        else:
            coltmp=tx[:,i]
            xmat[:,ind] = coltmp
            ind+=1
    return xmat

def min_max_scale(x):
    """Compute min-max scaling on x and return the scaled matrix"""
    
    x_min = (x.min(axis=0))
    x_max = (x.max(axis=0))

    return (x - x_min) / (x_max - x_min)

def split_data_set(X_total, Y_total, thresh):
    '''Splits data into all combinations of thersholds defined in thresh'''
    #Generates all combinations of thresholds
    combinations = []
    for i in range(len(thresh)):
        out = list(itertools.combinations(thresh, i+1))
        combinations.append([i for i in out])
    flat_list = [item for sublist in combinations for item in sublist]

    X_data_sets = []
    Y_data_sets = []
    for sublist in flat_list:
        #Iterates over sublist and creates split X
        in_Xs = [X_total]
        in_Ys = [Y_total]
        out_Xs = [] # Temporary variable used for holding splits
        out_Ys = []

        for split in sublist:
            #Applies all splits in sublist
            for i in range(len(in_Xs)):
                X = in_Xs[i]
                Y = in_Ys[i]
                out_Xs.append(X[X[:,split[0]]>=split[1]])
                out_Xs.append(X[X[:,split[0]]<split[1]])
                out_Xs.append(X[np.isnan(X[:,split[0]])])
                
                out_Ys.append(Y[X[:,split[0]]>=split[1]])
                out_Ys.append(Y[X[:,split[0]]<split[1]])
                out_Ys.append(Y[np.isnan(X[:,split[0]])])
            in_Xs = out_Xs.copy()
            in_Ys = out_Ys.copy()
            out_Xs = []
            out_Ys = []

        #Verifies if points are missing or duplicated 
        if not sum([arr.shape[0] for arr in in_Xs]) == X_total.shape[0]:
            print("Error in splitting Xs", sum([arr.shape[0] for arr in in_Xs]), [arr.shape[0] for arr in in_Xs])

        if not sum([arr.shape[0] for arr in in_Ys]) == Y_total.shape[0]:
            print("Error in splitting Ys")

        #Stores each dataset (for each sublist of splits)
        X_data_sets.append(in_Xs)
        Y_data_sets.append(in_Ys)

    # Removes empty split sets 
    X_sets_clean = []
    for dset in X_data_sets:
        X_sets_clean.append([split for split in dset if split.shape[0]>0])

    Y_sets_clean = []
    for dset in Y_data_sets:
        Y_sets_clean.append([split for split in dset if split.shape[0]>0])
    
    return X_sets_clean, Y_sets_clean, flat_list

def all_combinations_list(thresh):
    '''Generates all combinations for items in list (thresh)'''
    combinations = []
    for i in range(len(thresh)):
        out = list(itertools.combinations(thresh, i+1))
        combinations.append([i for i in out])
    flat_list = [item for sublist in combinations for item in sublist]
    return flat_list
    