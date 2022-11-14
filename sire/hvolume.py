"""
-------------------------
cre: Apr 2022

web: github.com/marcodeangelis
org: Univerity of Liverpool

MIT License
-------------------------

"""

import numpy as np
from numpy import (asarray)

from itertools import product

def hvolume(mvcdf, box):
    Ndims = box.shape[0]    # Dimension of box (must match mvcdf dims)
    Nverts = 2**Ndims       # Number of vertices
    Js = box.val            # Matrix of end-points
    verts = vertices(box)   # Get all endpoint coordinates
    if verts.shape[0] != Nverts: 
        raise ValueError("Number of returned vertices does not match needed number")
    cdfVals = np.zeros(Nverts)
    for i in range(Nverts):
        cdfVals[i] = mvcdf.cdf(verts[i,:])     # Evaluate cdf on endpoints
        Ns = 0
        for j in range(Ndims):
            this = verts[i,j] == Js[j,0]    # find how many verts are lower bounds
            Ns = Ns + this
        sign = 1
        if np.mod(Ns, 2) == 1:              # If number of lower bounds are odd
            sign = -1                       # cdf value will be subtracted
        cdfVals[i] = sign * cdfVals[i]
    return np.sum(cdfVals)

def vertices(box):
    X_ = []
    for xi in box:
        X_.append(xi.val)
    return asarray(list(product(*X_)), dtype=float)
