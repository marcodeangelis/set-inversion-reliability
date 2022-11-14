import numpy

# from sire.build_report import get_user_data,get_performance_functions,get_marginals,get_challenge_results
# from sire import SIVIA
from numpy import prod,asarray,linspace,meshgrid,reshape

from intervals.number import Interval
from intervals.methods import (width,contain,intersect,subtile,intervalise)

from .hvolume import hvolume
from .copula import Pi, Perfect#, GauCopula

from numpy.linalg import inv, det
from scipy.stats import norm, uniform, multivariate_normal, beta
from itertools import product

from typing import Callable

def SIVIA(fun:Callable,Y:Interval,X0:Interval,e:float=0.001,limit:int=100_000,E_continued:list=None):
    #   Y: interval IR of the target
    #  X0: interval IR^n input domain
    # fun: python function f(x1,x2,...,xn)
    X0=intervalise(X0)
    Y =intervalise(Y)
    def Width(x,x0):
        '''
        This subfunction computes the width ratio, and selects the dimension to be bisected.
        :input
        x: single box with shape (n,)
        x0: initial box with shape (n,)
        :output
        j: index i \in {0,1,...,n} of dimension to bisect
        r: width ratio used by SIVIA as a stopping criterion
        '''
        r=width(x)/width(x0)
        j=numpy.argmax(r)
        r_cand = r[j]
        return r_cand, j
    def Bisect(x,i=0):
        '''
        This function performs the actual bisection.
        x: a n-box or n-interval with shape (n,) to be bisected
        i: dimension to be bisected
        
        :output
        x_bisect[:,0]: Interval (d,)-array, first column, first half of the bisection
        x_bisect[:,1]: Interval (d,)-array, second column, second half of the bisection
        '''
        d=x.shape[0] 
        n=[0]*d
        n[i]=2 # ex: (0,0,2,0,0,0) if interval of dim 6 has third dimension bisected
        x_bisect = subtile(x,n=tuple(n))
        return x_bisect[:,0], x_bisect[:,1]
    if E_continued is not None:
        L = E_continued
    else:
        L = [X0]
    S = [] # Sì
    N = [] # No
    E = [] # Eh?
    success = False # not quite yet
    for it in range(limit):
        try:
            X = L.pop(0)
        except IndexError as error:
            if str(error) == 'pop from empty list':
                success = True
                print(f'Algorithm reached genuine end at iteration {it+1}')
                break # for loop the algorithm reached successful completion
        yy = fun(*X)
        c,i = Width(X,X0)
        if contain(Y,yy): # Y contains yy
            S.append(X)
        elif not(intersect(Y,yy)): # Y and yy do not intersect
            N.append(X)
        elif c <= e: # max box size sufficiently small
            E.append(X)
        else:
            x1,x2 = Bisect(X,i)
            L.append(x1)
            L.append(x2)
    if not(success):
        E += L
        print(f'Algorithm stopped at limit iteration: {it+1}. Total number of boxes: f{len(S)+len(N)+len(E)}')
    print(f'S: {len(S)}')
    print(f'N: {len(N)}')
    print(f'E: {len(E)}')
    return S,N,E,it+1

def failure_probability(S,E,marginals,preci=1e8):
    Pe=aggregate_measure(E,marginals)
    Ps=aggregate_measure(S,marginals)
    print(f'Failure probability = [{round(Ps*preci)/preci}, {round((Ps+Pe)*preci)/preci}]')
    return Interval(round(Ps*preci)/preci,round((Ps+Pe)*preci)/preci)

def box_volume(list_of_boxes):
    return sum([prod(width(box)) for box in list_of_boxes])
    
def failure_probability_copula_indep(S, N, E):
    Ps = box_volume(S)
    Pe = box_volume(E)
    return Interval(Ps, Ps+Pe)

def aggregate_measure(list_of_boxes, marginals):
    if len(list_of_boxes)==0: return 0
    x = boxeslist_to_interval(list_of_boxes)
    P = 1
    for i,m in enumerate(marginals):
        P *= m.cdf(x.hi[:,i]) - m.cdf(x.lo[:,i])
    return sum(P)

def boxeslist_to_interval(boxes):
    return Interval([b.lo for b in boxes],[b.hi for b in boxes])

def failure_probability_copula(S, N, E, C = Pi , preci=1e8):
    if C == Pi:
        return failure_probability_copula_indep(S, N, E)
    Ps = sum([hvolume(C, Si) for Si in S])
    Pe = sum([hvolume(C, Ei) for Ei in E])
    return Interval(round(Ps*preci)/preci,round((Ps+Pe)*preci)/preci) #     return Interval(Ps,Ps+Pe)

def failure_probability(S,E,marginals,preci=1e8):
    Pe=aggregate_measure(E,marginals)
    Ps=aggregate_measure(S,marginals)
    return Interval(round(Ps*preci)/preci,round((Ps+Pe)*preci)/preci)

def montecarlo_pf(N:int, marginals:list, gfun):
    samples = numpy.asarray(sample(marginals,N=N))
    g = gfun(*samples)
    i=0
    for gi in g:
        if gi>=0:
            i+=1
    return i,N, samples

def montecarlo_pf_correlated(n:int,marginals:list,gfun,cormat,bounding_box):
    corr_samples  = correlated_sampling(n,marginals,cormat,bounding_box)
    g = gfun(*corr_samples.T)
    i=0
    for gi in g:
        if gi>=0:
            i+=1
    return i, n, corr_samples

def correlated_sampling(n,marginals,cormat,bounding_box):
    d=len(marginals)
    X = multivariate_normal.rvs(numpy.zeros(d),cormat,size=n)
    U = norm.cdf(X)
    samples = numpy.asarray([marginals[i].ppf(U[:,i]) for i in range(len(marginals))]).T
    return samples

def sample(marginals, N=1):
    X = []
    for i,d in enumerate(marginals):
        X.append(d.rvs(N))
    return X

def rescale(samples, bounding_box): return ((samples * width(bounding_box)) + bounding_box.lo).T ### plot MV density
    