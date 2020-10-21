# -*- coding: utf-8 -*-
"""functions to compute OHE of categorical columns"""

import numpy as np

def get_jet_OHE(jets):
    """returns OHE columns given a column of jet values"""

    n_jets = 4
    
    jets_OHE = np.zeros((len(jets), n_jets))
    jets_OHE[np.arange(len(jets_OHE)), jets] = 1
    
    return jets_OHE

def get_centrality_split_OHE(centrality, threshold):
    """returns 2 OHE columns given a column of centrality values and a splitting threshold"""
    
    centrality_OHE = np.zeros((len(centrality), 2))

    centrality_OHE[:,0] = (centrality < threshold).astype(int)
    centrality_OHE[:,1] = 1 - centrality_OHE[:,0]
    
    return centrality_OHE