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
    """polynomial basis functions for input data tx, for all features expanded to degrees in list.Extended matrix is inputed np.array"""
    # ***************************************************
    # augmented x matrix with column of 1 input
    #list of indices for features modified
    #degrees of polynomial to apply to all features irrespectively
    # ***************************************************
    xmat=np.ones(tx.shape[0]).reshape(-1,1)
    for i in range(1,tx.shape[1]):
        for d in degree:
            coltmp=tx[:,i]**d
            xmat = np.append(xmat, coltmp.reshape(-1,1), axis=1)

        
    return np.array(xmat)

def build_poly_index(tx, index_list, degree):
    """polynomial basis functions for input data tx, for all features expanded to d degree.Extended matrix is inputed np.array. Index list includes all variables that will be expanded, mostly useful to avoid expanding categorical variables"""
    # ***************************************************
    # augmented x matrix with column of 1 input
    #list of indices for features modified
    #degrees of polynomial to apply to all features irrespectively
    # ***************************************************
    xmat=np.ones(tx.shape[0]).reshape(-1,1)
    for i in range(1,tx.shape[1]):
        if i in index_list:
            for d in degree:
                coltmp=tx[:,i]**d
                xmat = np.append(xmat, coltmp.reshape(-1,1), axis=1)
        if i not in index_list:
            coltmp=tx[:,i]
            xmat = np.append(xmat, coltmp.reshape(-1,1), axis=1)
    return np.array(xmat)

def min_max_scale(x):
    """Compute min-max scaling on x and return the scaled matrix"""
    
    x_min = (x.min(axis=0))
    x_max = (x.max(axis=0))

    return (x - x_min) / (x_max - x_min)

def standardize_by_distribution(data, st_type):
    '''Allows for standardization by different distributions'''
    
    out = data[[not np.isnan(i) for i in data]]

    # No standardization
    if st_type==0:
        return data
    
    # Gaussian standardization
    if st_type==1:
        data = data - out.mean()
        data = data/out.std()
        return data
    
    # Poisson standardization (mean == std)
    if st_type==2:
        data = data/out.std()
        return data
    
    # Mean stadardization (mean = 2x std)
    if st_type==3:
        data = data/out.mean()
        return data
    
    # Skewed standardization
    if st_type==4:
        try:
            data = data - statistics.mode(out)
        except:
            data = data - out.mean()
        data = data/out.std()
        return data
    
    # Max standardization
    if st_type==5:
        data = data/out.max()
        return data

def select_standardization(data):
    """Returns the type of standardization to apply to each feature (data)"""
    
    out = data[[not np.isnan(i) for i in data]]
    
    if len(np.unique(out))<10:          #No standardization if categorical
        st_type = 0
        
    elif ((out.max()-out.min())<2) & (out.max()<10): #No standardization if small span
        st_type = 0
        
    elif out.min()>=0:
        if out.mean()/out.std()>=2.8:   #Mode standardization for spread out
            st_type = 4
        elif out.mean()/out.std()>=1.5: #Mean standardization
            st_type = 3
        elif out.mean()/out.std()>=0.5: #Std Standardization
            st_type = 2
        else:                           #Max standardixation
            st_type = 5
            
    elif abs(out.min())==out.max():     #Max standardization
        st_type = 5
        
    else:                               #Gaussian standardization
        st_type = 1
    
    return st_type

def standardize_data(X_array):
    '''Returns the standardized data array as internally defined in st_types'''
    
    X_out = X_array.copy()
    for i in range(X_array.shape[1]):
        data = X_array[:,i]
        st_type = select_standardization(data)
        X_out[:,i] = standardize_by_distribution(data, st_type)
        
    return X_out

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
                out_Ys.append(Y[X[:,split[0]]>=split[1]])
                out_Ys.append(Y[X[:,split[0]]<split[1]])
            in_Xs = out_Xs.copy()
            in_Ys = out_Ys.copy()
            out_Xs = []
            out_Ys = []

        #Verifies if points are missing or duplicated 
        if not sum([arr.shape[0] for arr in in_Xs]) == X_total.shape[0]:
            print("Error in splitting Xs")

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