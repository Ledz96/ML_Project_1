# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    coefficients = np.empty((x.shape[0], degree + 1))

    exps = np.linspace(0, degree, degree + 1)

    for i, xn in enumerate(x):
        coefficients[i] = np.power(xn, exps)

    return coefficients