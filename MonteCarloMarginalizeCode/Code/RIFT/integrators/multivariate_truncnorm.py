# -*- coding: utf-8 -*-
'''
Multivariate Truncnorm
----------------------
Draw samples from a truncated multivariate Gaussian distribution.
'''
from __future__ import print_function
import itertools
import numpy as np
from scipy.stats import truncnorm
from time import time

# Code adapted from http://www.aishack.in/tutorials/generating-multivariate-gaussian-random/

def _get_corner_coords(bounds):
    '''
    Gets coordinates of corner points given bounds for each dimension
    '''
    d = len(bounds)
    points = np.empty((2**d, d))
    bounds = np.rot90(bounds)
    i = np.matrix(list(itertools.product([0,1], repeat=d)))
    index = 0
    for slice in i:
        t = np.diag(bounds[slice][0])
        points[index] = t
        index += 1
    return points


def _get_new_bounds(bounds, q):
    '''
    Finds smallest rectangular region that, when transformed, will contain the
    desired rectangular sampling region
    '''
    r = q.I # inverse of transformation
    d = len(bounds)
    old_points = _get_corner_coords(bounds)
    new_points = np.empty((2**d, d))
    index = 0
    for point in old_points:
        new = np.dot(r, point)
        new_points[index] = new
        index += 1
    new_bounds = np.empty((d, 2))
    for dim in range(d):
        new_bounds[dim][0] = min(new_points[:,[dim]])
        new_bounds[dim][1] = max(new_points[:,[dim]])
    return new_bounds


def _get_multipliers(cov):
    '''
    Gets the linear transformation to shift samples
    '''
    [lam, sigma] = np.linalg.eig(cov)
    lam = np.matrix(np.diag(np.sqrt(lam)))
    q = np.matrix(sigma) * lam
    return q


def sample(mean, cov, bounds, n):
    '''
    Generate samples

    Parameters
    ----------
    mean : np.ndarray
        Mean of the distribution
    cov : np.ndarray
        Covariance matrix of the distribution
    bounds : np.ndarray
        Bounds for samples
    n : int
        Number of samples to draw

    Returns
    -------
    np.ndarray
        Array of samples
    '''
    mean = np.matrix(mean)
    d = len(bounds)
    q = _get_multipliers(cov)
    new_bounds = _get_new_bounds(bounds - np.rot90(mean, -1), q)
    llim_new = new_bounds[:,[0]]
    rlim_new = new_bounds[:,[1]]
    ret = np.empty((0, d))
    iter = 0
    while len(ret) < n:
        samples = np.rot90(truncnorm.rvs(llim_new, rlim_new, loc=0, scale=1, size=(d, n)))
        samples = np.rot90(np.inner(q, samples))
        samples += mean
        llim = np.rot90(bounds[:,[0]])
        rlim = np.rot90(bounds[:,[1]])
        replace1 = np.greater(samples, llim).all(axis=1)
        replace2 = np.less(samples, rlim).all(axis=1)
        replace = np.array(np.logical_and(replace1, replace2)).flatten()
        to_append = samples[replace]
        ret = np.append(ret, to_append, axis=0)
        if iter > n:
            print('Error sampling')
            return False
        iter += 1
    return ret[:n]
