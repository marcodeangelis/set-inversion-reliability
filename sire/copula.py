import numpy as np
from numpy.linalg import inv, det

from scipy.stats import norm, uniform, multivariate_normal, beta

from intervals.methods import width

class MvDist():
    def __init__(self, *args):
        self.__copula = args[0]
        self.__marginals = args[1]
    def marginals(self):
        return self.__marginals
    def length(self):
        return len(self.__marginals)
    def cdf(self, x):
        return self.__copula.cdf( [self.__marginals[i].cdf(x[i]) for i in range(len(x)) ])

class Copula():
    def __init__(self, *args):
        if len(args)==1: self.__func = args[0]
        else:
            cormat = args[1]
            self.__func = args[0](cormat)
    def cdf(self, x):
        return self.__func(x)

Pi = Copula(np.prod)            # ...
Perfect = Copula(np.min)        # make some copulas 

def gaucop(cormat):                  # NUMERICALLY UNSTABLE, NOT RIGOROUS
    def f(x):
        if np.any(np.asarray(x) == 0): return 0.0
        return multivariate_normal(np.zeros(cormat.shape[0]), cormat).cdf(norm(0,1).ppf(x))
    return f

# GauCopula = Copula(gaucop())      # NUMERICALLY UNSTABLE, NOT RIGOROUS

def joint_pdf(marginals:list):
    def f(*x):
        return np.prod([mi.pdf(xi) for mi,xi in zip(marginals,x)],axis=0)
    return f

def c_density(u, cormat):
    SND = norm()
    R_inv = inv(cormat)
    xs = SND.ppf(u)
    return 1/np.sqrt(det(cormat)) * np.exp(-np.matmul(np.matmul(xs.T, R_inv), xs/2)) / np.exp(-np.matmul(np.matmul(xs.T, np.eye(2)), xs/2))

# ASSUME GAUSSIAN COPULA WITH COR, AND 2D
def MV_density(marginals, cormat, x, bounding_box):
    x_new = (x - bounding_box.lo)/width(bounding_box)
    us = [marginals[i].cdf(x_new[i]) for i in range(len(marginals))]
    pdfs = [marginals[i].pdf(x_new[i]) for i in range(len(marginals))]
    return c_density(us, cormat) * np.prod(pdfs)