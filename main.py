from intervals.number import Interval

from sire.copula import Copula,MvDist,gaucop
from sire.sire import (SIVIA,failure_probability_copula,failure_probability,montecarlo_pf,montecarlo_pf_correlated)

from sire.sire import correlated_sampling, sample

import numpy as np

from scipy.stats import norm

def performance_function(): # target failure probability 0.00286 with x1,2 = N(loc=10,scale=3) and product copula.
    def f(*args):
        return -(2.5 - 0.2357 * (args[0]-args[1]) + 0.00463 * (args[0]+args[1]-20)**4)
    return f


if __name__ == '__main__':

    g_fun = performance_function()
    marginals = [norm(loc=10.,scale=3.), norm(loc=10.,scale=3.)]

    Cor = 0.4
    CorrMat = np.array([[1, Cor],[Cor, 1]])

    Cop = Copula(gaucop,CorrMat)
    MVdis = MvDist(Cop, marginals)

    Y0 = Interval(0,np.inf) # failure interval. Set to [-∞, 0] for failure g(x)<0
    X0 = Interval(lo=[-2,-2],hi=[22,22])
    t = 0.01
    c = 10_000

    # Run the sivia algorithm
    S,N,E,eval = SIVIA(g_fun,Y0,X0,e=t,limit=c)

    # Compute the failure probability
    pF_indep = failure_probability(S,E, marginals)
    print('----')
    print(f'SIRE Independent failure probability: {pF_indep}')
    pF = failure_probability_copula(S,N,E, MVdis)
    print(f'SIRE with copula: {pF}')

    k,n, rs = montecarlo_pf(10_000_000, marginals, g_fun)
    print('----')
    print('Independent Monte Carlo:')
    print(f"k={k}, n={n}, pf={k/n}")

    # samples_indep = np.asarray(sample(marginals,1000))
    # print(samples_indep.shape)
    # fig,ax = pyplot.subplots(figsize=(12,12))
    # ax.scatter(samples_indep[0,:],samples_indep[1,:])
    # pyplot.show()

    k,n, rs = montecarlo_pf_correlated(10_000_000, marginals, g_fun, CorrMat, X0)
    print('---')
    print('Correlated Monte Carlo')
    print(f"k={k}, n={n}, pf={k/n}")

    # corsamples = correlated_sampling(n,marginals,CorrMat,X0)
    # from matplotlib import pyplot
    # fig,ax = pyplot.subplots(figsize=(12,12))
    # ax.scatter(corsamples[:,0],corsamples[:,1])
    # pyplot.show()