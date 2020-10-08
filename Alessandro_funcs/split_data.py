# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    # split the data based on the given ratio: TODO    
    p = np.random.permutation(len(y))
    y = y[p]
    x = x[p]
    
    limit = int(len(y)*ratio)
        
    return x[:limit],y[:limit],x[limit:],y[limit:]
