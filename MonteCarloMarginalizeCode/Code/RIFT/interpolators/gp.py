# gp.py: Very primitive gaussian process regression module.  Returns an object
#
#  - Prediction:   K(x,x').inv(Kmtx).y.  We store and precompute inv(Kmtx).y

import numpy as np  
from sklearn import gaussian_process

def squared_exponential_kernel(dx,h,sigma0,sigmab):
    ret  = np.power(sigma0,2) * np.exp(- np.power(dx,2)/(2.* np.power(h,2)))
    ret += np.where(np.abs(dx) < 1e-9, np.ones(dx.shape)*np.power(sigmab,2), np.zeros(dx.shape) )
    return ret


class GaussianProcess1d:
    """
    Very simple gaussian process model. Does not take into account errors that vary with position.
    """
    def __init__(self, dat1d,sigma0,sigmab=1e-5,h=None):
        self.h = h
        self.sigma0 = sigma0
        self.sigmab = sigmab
        x= self.x = dat1d[:,0]
        self.y = dat1d[:,1]
        if self.h == None:
            hmin = np.min(x - np.roll(x,1)) # minimum spacing, roughly
            self.h = np.max([hmin, 3*(np.max(x) - np.min(x))/len(x)])
        x = dat1d[:,0]
        xv,xpv = np.meshgrid(x,x)
        xMinusXp = xv - xpv
        Kmtx = squared_exponential_kernel(xMinusXp,self.h,self.sigma0,self.sigmab)
#        print "Det", np.linalg.det(Kmtx)
        KmtxInv = np.linalg.inv(Kmtx)
        KmtxInvY = np.matrix(KmtxInv)*np.matrix( np.reshape(dat1d[:,1], (len(x),1)))
        self.KmtxInvY = np.matrix(np.reshape(KmtxInvY, (len(x),1)))
    def predict(self, x_new):
        xv,xpv = np.meshgrid(x_new,self.x)
        xMinusXp = xv - xpv
        Kmtx_new = np.matrix(squared_exponential_kernel(xMinusXp,self.h,self.sigma0,self.sigmab).T)
#        print Kmtx_new.shape, self.KmtxInvY.shape
        return Kmtx_new* self.KmtxInvY
