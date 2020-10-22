# -*- coding: utf-8 -*-
"""data indexing in order to keep split data order"""

import numpy as np

def prepend_indices(x):
    """prepend indices to x"""
    
    indices = np.array(range(0,len(x)))
    return np.c_[indices, x]

def append_indices(y, indices):
    """append indices to predictions"""
    
    return np.c_[indices, y]

def extract_indices(x):
    """takes away indices and return them"""
    
    indices = x[:,1]
    return indices, np.delete(x, 0, axis=1)