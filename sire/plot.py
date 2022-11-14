from matplotlib import pyplot

import numpy as np

from copula import MV_density

FONTSIZE=18
FIGSIZE =(10,10)
def plotbox_(x,y,ax=None,c=None,label=None):
    if ax is None:
        fig = pyplot.figure(figsize=FIGSIZE)
        ax = fig.subplots()
    if c is None:
        ax.fill_between([x.lo,x.hi],[y.lo,y.lo],[y.hi,y.hi], edgecolor='gray',label=label)
    else:
        ax.fill_between([x.lo,x.hi],[y.lo,y.lo],[y.hi,y.hi], facecolor=c, edgecolor='gray',label=label)
    return ax
def standard_plot(S,N,E,c=['red','orange','blue'],legend=True,loc=None):
    xlabel = r'$x_1$'
    ylabel = r'$x_2$'
    fig, ax = pyplot.subplots(figsize=FIGSIZE)
    for s in S:
        plotbox_(s[0],s[1],ax=ax,c=c[0])
    for n in N:
        plotbox_(n[0],n[1],ax=ax,c=c[2])
    for e in E:
        plotbox_(e[0],e[1],ax=ax,c=c[1])
    plotbox_(s[0],s[1],ax=ax,c=c[0],label='Fail zone')
    plotbox_(e[0],e[1],ax=ax,c=c[1],label='Undecided')
    plotbox_(n[0],n[1],ax=ax,c=c[2],label='Safe zone')
    ax.set_xlabel(xlabel,fontsize=FONTSIZE)
    ax.set_ylabel(ylabel,fontsize=FONTSIZE)
    ax.tick_params(direction='out', length=6, width=2, colors='#5a5a64', grid_color='gray', grid_alpha=0.5, labelsize='x-large')
    if legend: ax.legend(fontsize=FONTSIZE-2,loc=loc)
    return fig,ax


def plotbox(x,y,ax=None,c=None,alpha=None,grid=False,label=None):
    if ax is None:
        fig = pyplot.figure(figsize=(20,6))
        ax = fig.subplots()
    if c is None:
        ax.fill_between([x.lo,x.hi],[y.lo,y.lo],[y.hi,y.hi], edgecolor='gray', alpha=alpha, label=label)
    else:
        ax.fill_between([x.lo,x.hi],[y.lo,y.lo],[y.hi,y.hi], facecolor=c, edgecolor='gray', alpha=alpha, label=label)
    if grid: ax.grid()
    return ax

def plotall(S,N,E,ax=None):
    if ax is None:
        fig = pyplot.figure(figsize=(20,9))
        ax = fig.subplots()
    for s in S:
        plotbox(*s,ax=ax,c='red')
    for n in N:
        plotbox(*n,ax=ax,c='blue')
    for e in E:
        plotbox(*e,ax=ax,c='orange')
    ax.set_xlabel('x',fontsize=18)
    ax.set_ylabel('y',fontsize=18)
    pyplot.show

def plot_iso_densities(n,marginals,cormat,ax=None,colormap='rainbow',levels=None):
    us = np.linspace(X0.val[0][0], X0.val[0][1], n)
    vs = np.linspace(X0.val[1][0], X0.val[1][1], n)
    X,Y = np.meshgrid(us,vs)
    XY = np.asarray([X.flatten(), Y.flatten()])
    dens_ = [MV_density(marginals, cormat, XY[:,i]) for i in range(XY.shape[1])]
    dens = np.asarray(dens_).reshape((n,n))
    if ax is None:
        fig,ax=pyplot.subplots(figsize=(12,12))
    # ax = plotall(s,n,e)
    ax.contour(X, Y, dens, colormap=colormap, levels=levels)
#     ax.scatter(rs[:,0],rs[:,1])
    ax.set_aspect('equal')
    pass