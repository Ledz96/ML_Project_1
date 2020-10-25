# -*- coding: utf-8 -*-
"""standardization functions"""

from enum import Enum
import multiprocessing
from multiprocessing.pool import Pool
import numpy as np
import statistics

class StdType(Enum):
    NO_STD = 0
    GAUSSIAN = 1
    POISSON = 2
    MEAN = 3
    SKEWED = 4
    MAX = 5
    
def standardize_switch(argument):
    switcher = {
        StdType.GAUSSIAN: standardize_gaussian,
        StdType.POISSON: standardize_poisson,
        StdType.MEAN: standardize_mean,
        StdType.SKEWED: standardize_skewed,
        StdType.MAX: standardize_max,
    }
    
    f = switcher.get(argument, lambda: "Incorrect standardization type")
    return f

def standardize_gaussian(data):
    data = data - data.mean()
    data = data/data.std()
    return data

def standardize_poisson(data):
    data = data/data.std()
    return data

def standardize_mean(data):
    data = data/data.mean()
    return data

def standardize_skewed(data):
    try:
        data = data - statistics.mode(data)
    except:
        data = data - data.mean()
    data = data/data.std()
    return data

def standardize_max(data):
    data = data/data.max()
    return data

def select_standardization(data):
    """Returns the type of standardization to apply to each feature (data)"""
        
    if len(np.unique(data))<10:          #No standardization if categorical
        st_type = StdType.NO_STD
        
    elif ((data.max()-data.min())<2) & (data.max()<10): #No standardization if small span
        st_type = StdType.NO_STD
        
    elif data.min()>=0:
        if data.mean()/data.std()>=2.8:   #Mode standardization for spread data
            st_type = StdType.SKEWED
        elif data.mean()/data.std()>=1.5: #Mean standardization
            st_type = StdType.MEAN
        elif data.mean()/data.std()>=0.5: #Std Standardization
            st_type = StdType.POISSON
        else:                           #Max standardixation
            st_type = StdType.MAX
            
    elif abs(data.min())==data.max():     #Max standardization
        st_type = StdType.MAX
        
    else:                               #Gaussian standardization
        st_type = StdType.GAUSSIAN
    
    return st_type

def standardize_column(data, col, return_dict):
    st_type = select_standardization(data)
    if st_type != StdType.NO_STD:
        return_dict[col] = standardize_switch(st_type)(data)
        
def standardize_data_parallel(x):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = {}

    ret = x.copy()
    
    for i in range(1,x.shape[1]):
        ps = multiprocessing.Process(target=standardize_column, args=(x[:,i], i, return_dict))
        jobs[i] = ps
        ps.start()
        
    for i in range(1,x.shape[1]):
        jobs[i].join()
        if i in return_dict.keys():
            ret[:,i] = return_dict[i]
        
    return ret

def standardize_data(x):
    ret = x.copy()
    
    for i in range(1, x.shape[1]):
        st_type = select_standardization(x[:,i])
        if st_type != StdType.NO_STD:
            ret[:,i] = standardize_switch(st_type)(x[:,i])
        
    return ret