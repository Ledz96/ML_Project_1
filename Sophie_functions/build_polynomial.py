# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    xmat=[]
    for i in range(len(x)):
        xaug=[]
        xi=x[i]
        for j in range(degree+1):
            xaug.append(xi**j)
        xmat.append(xaug)
        
    return np.array(xmat)
    
