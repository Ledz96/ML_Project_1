# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    datasamples=int(ratio*len(y))
    indices= np.random.choice(len(y), datasamples, replace=False)
    Xtrain=[x[i] for i in indices]
    Xtest=[item for index, item in enumerate(x) if index not in indices]
    Ytrain=[y[i] for i in indices]
    Ytest=[item for index, item in enumerate(y) if index not in indices]
    
    return Xtest, Xtrain, Ytest, Ytrain