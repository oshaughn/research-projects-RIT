# Copyright (C) 2013  Evan Ochsner
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
Module of routines to compute an effective Fisher matrix and related utilities,
such as finding a region of interest and laying out a grid over it
"""

from lalsimutils import *
from scipy.optimize import leastsq, brentq
from scipy.linalg import eig, inv

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>"


def effectiveFisher(residual_func, *flat_grids):
    """
    Fit a quadratic to the ambiguity function tabulated on a grid.
    Inputs:
        - a pointer to a function to compute residuals, e.g.
          z(x1, ..., xN) - fit
          for N-dimensions, this is called 'residualsNd'
        - N+1 flat arrays of length K. N arrays for each on N parameters,
          plus 1 array of values of the overlap
    Returns:
        - flat array of upper-triangular elements of the effective Fisher matrix

    Example:
    x1s = [x1_1, ..., x1_K]
    x2s = [x2_1, ..., x2_K]
    ...
    xNs = [xN_1, ..., xN_K]

    gamma = effectiveFisher(residualsNd, x1s, x2s, ..., xNs, rhos)

    gamma = [g_11, g_12, ..., g_1N, g_22, ..., g_2N, g_33, ..., g_3N, ..., g_NN]
    """
    x0 = np.ones(len(flat_grids))
    fitgamma = leastsq(residual_func, x0=x0, args=tuple(flat_grids))
    return fitgamma[0]

def evaluate_ip_on_grid(hfSIG, P, IP, param_names, grid):
    """
    Evaluate IP.ip(hsig, htmplt) everywhere on a multidimensional grid
    """
    Nparams = len(param_names)
    Npts = len(grid)
    assert len(grid[0])==Nparams
    return [update_params_ip(hfSIG, P, IP, param_names, grid[i])
            for i in xrange(Npts)]


def update_params_ip(hfSIG, P, IP, param_names, vals):
    """
    Update the values of 1 or more member of P, recompute norm_hoff(P),
    and return IP.ip(hfSIG, norm_hoff(P))

    Inputs:
        - hfSIG: A COMPLEX16FrequencySeries of a fixed, unchanging signal
        - P: A ChooseWaveformParams object describing a varying template
        - IP: An InnerProduct object
        - param_names: An array of strings of parameters to be updated.
            e.g. [ 'm1', 'm2', 'incl' ]
        - vals: update P to have these parameter values. Must have as many
            vals as length of param_names, ordered the same way

    Outputs:
        - A COMPLEX16FrequencySeries, same as norm_hoff(P, IP)
    """
    hfTMPLT = update_params_norm_hoff(P, IP, param_names, vals)
    if IP.full_output == True:
        rho, rhoSeries, rhoIdx, rhoArg = IP.ip(hfSIG, hfTMPLT)
        return rho
    else:
        return IP.ip(hfSIG, hfTMPLT)


def update_params_norm_hoff(P, IP, param_names, vals, verbose=False):
    """
    Update the values of 1 or more member of P and recompute norm_hoff(P).

    Inputs:
        - P: A ChooseWaveformParams object
        - IP: An InnerProduct object
        - param_names: An array of strings of parameters to be updated.
            e.g. [ 'm1', 'm2', 'incl' ]
        - vals: update P to have these parameter values. Must be array-like
            with same length as param_names, ordered the same way
    Outputs:
        - A COMPLEX16FrequencySeries, same as norm_hoff(P, IP)
    """
    special_params = []
    special_vals = []
    assert len(param_names)==len(vals)
    for i, val in enumerate(vals):
        if hasattr(P, param_names[i]): # Update an attribute of P...
            setattr(P, param_names[i], val)
        else: # Either an incorrect param name, or a special case...
            special_params.append(param_names[i])
            special_vals.append(val)

    # Check allowed special cases of params not in P, e.g. Mc and eta
    if special_params==['Mc','eta']:
        m1, m2 = m1m2(special_vals[0], special_vals[1]) # m1,m2 = m1m2(Mc,eta)
        setattr(P, 'm1', m1)
        setattr(P, 'm2', m2)
    elif special_params==['eta','Mc']:
        m1, m2 = m1m2(special_vals[1], special_vals[0])
        setattr(P, 'm1', m1)
        setattr(P, 'm2', m2)
    elif special_params==['Mc']:
        eta = symRatio(P.m1, P.m2)
        m1, m2 = m1m2(special_vals[0], eta)
        setattr(P, 'm1', m1)
        setattr(P, 'm2', m2)
    elif special_params==['eta']:
        Mc = mchirp(P.m1, P.m2)
        m1, m2 = m1m2(Mc, special_vals[0])
        setattr(P, 'm1', m1)
        setattr(P, 'm2', m2)
    elif special_params != []:
        print special_params
        raise Exception

    if verbose==True: # for debugging - make sure params change properly
        P.print_params()
    return norm_hoff(P, IP)



def find_effective_Fisher_region(P, IP, target_match, param_names,param_bounds):
    """
    Example Usage:
        find_effective_Fisher_region(P, IP, 0.9, ['Mc', 'eta'], [[mchirp(P.m1,P.m2)-lal.LAL_MSUN_SI,mchirp(P.m1,P.m2)+lal.LAL_MSUN_SI], [0.05, 0.25]])
    Arguments:
        - P: a ChooseWaveformParams object describing a target signal
        - IP: An inner product class to compute overlaps.
                Should have deltaF and length consistent with P
        - target_match: find parameter variation where overlap is target_match.
                Should be a real number in [0,1]
        - param_names: array of string names for members of P to vary.
                Should have length N for N params to be varied
                e.g. ['Mc', 'eta']
        - param_bounds: array of max variations of each param in param_names
                Should be an Nx2 array for N params to be varied
    
    Returns:
        Array of boundaries of a hypercube meant to encompass region where
                match is >= target_match.
                e.g. [ [3.12,3.16] , [0.12, 0.18] ]

    N.B. Only checks variations along parameter axes. If params are correlated,
    may get better matches off-axis, and the hypercube won't fully encompass
    region where target_match is achieved. Therefore, allow a generous
    safety factor in your value of 'target_match'.
    """
    # FIXME: Use a root-finder to bound a region of interest
    Nparams = len(param_names)
    assert len(param_bounds) == Nparams
    param_cube = []
    hfSIG = norm_hoff(P, IP)
    for i, param in enumerate(param_names):
        PT = P.copy()
        if param=='Mc':
            param_peak = mchirp(P.m1, P.m2)
        elif param=='eta':
            param_peak = symRatio(P.m1, P.m2)
        else:
            param_peak = getattr(P, param)
        func = lambda x: update_params_ip(hfSIG, PT, IP, [param], [x]) - target_match
        max_param = brentq(func, param_peak, param_bounds[i][1])
        min_param = brentq(func, param_bounds[i][0], param_peak)
        param_cube.append( [min_param, max_param] )

    return param_cube

#
# Routines to make various types of grids for arbitrary dimension
#
def make_regular_1d_grids(param_ranges, pts_per_dim):
    """
    Inputs: 
        - param_ranges is an array of parameter bounds, e.g.:
        [ [p1_min, p1_max], [p2_min, p2_max], ..., [pN_min, pN_max] ]
        - pts_per_dim is either:
            a) an integer - use that many pts for every parameter
            b) an array of integers of same length as param_ranges, e.g.
                [ N1, N2, ..., NN ]
                the n-th entry is the number of pts for the n-th parameter

    Outputs:
        outputs N separate 1d arrays of evenly spaced values of that parameter,
        where N = len(param_ranges)
    """
    Nparams = len(param_ranges)
    assert len(pts_per_dim)
    grid1d = []
    for i in range(Nparams):
        MIN = param_ranges[i][0]
        MAX = param_ranges[i][1]
        STEP = (MAX-MIN)/(pts_per_dim[i]-1)
        EPS = STEP/100.
        grid1d.append( np.arange(MIN,MAX+EPS,STEP) )

    return tuple(grid1d)

def multi_dim_meshgrid(*arrs):
    """
    Version of np.meshgrid generalized to arbitrary number of dimensions.
    Taken from: http://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d
    """
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    #return tuple(ans)
    return tuple(ans[::-1])

def multi_dim_flatgrid(*arrs):
    """
    Creates flattened versions of meshgrids.
    Returns a tuple of arrays of values of individual parameters
    at each point in a grid, returned in a flat array structure.

    e.g.
    x = [1,3,5]
    y = [2,4,6]
    X, Y = multi_dim_flatgrid(x, y)
    returns:
    X = [1,1,1,3,3,3,5,5,5]
    Y = [2,4,6,2,4,6,2,4,6]
    """
    outarrs = multi_dim_meshgrid(*arrs)
    return tuple([ outarrs[i].flatten() for i in xrange(len(outarrs)) ])

def multi_dim_grid(*arrs):
    """
    Creates an array of values of all pts on a multidimensional grid.

    e.g.
    x = [1,3,5]
    y = [2,4,6]
    multi_dim_grid(x, y)
    returns:
    [[1,2], [1,4], [1,6],
     [3,2], [3,4], [3,6],
     [5,2], [5,4], [5,6]]
    """
    temp = multi_dim_flatgrid(*arrs)
    return np.transpose( np.array(temp) )


#
# Routines for least-squares fit
#
def residuals2d(gamma, y, x1, x2):
    g11 = gamma[0]
    g12 = gamma[1]
    g22 = gamma[2]
    return y - (1. - g11*x1*x1/2. - g12*x1*x2 - g22*x2*x2/2.)

def evalfit2d(x1, x2, gamma):
    g11 = gamma[0]
    g12 = gamma[1]
    g22 = gamma[2]
    return 1. - g11*x1*x1/2. - g12*x1*x2 - g22*x2*x2/2.

def residuals3d(gamma, y, x1, x2, x3):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g22 = gamma[3]
    g23 = gamma[4]
    g33 = gamma[5]
    return y - (1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3
            - g22*x2*x2/2. - g23*x2*x3 - g33*x3*x3/2.)

def evalfit3d(x1, x2, x3, gamma):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g22 = gamma[3]
    g23 = gamma[4]
    g33 = gamma[5]
    return 1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3\
            - g22*x2*x2/2. - g23*x2*x3 - g33*x3*x3/2.

def residuals4d(gamma, y, x1, x2, x3, x4):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g14 = gamma[3]
    g22 = gamma[4]
    g23 = gamma[5]
    g24 = gamma[6]
    g33 = gamma[7]
    g34 = gamma[8]
    g44 = gamma[9]
    return y - (1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3 - g14*x1*x4
            - g22*x2*x2/2. - g23*x2*x3 - g24*x2*x4 - g33*x3*x3/2. - g34*x3*x4
            - g44*x4*x4/2.)

def evalfit4d(x1, x2, x3, x4, gamma):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g14 = gamma[3]
    g22 = gamma[4]
    g23 = gamma[5]
    g24 = gamma[6]
    g33 = gamma[7]
    g34 = gamma[8]
    g44 = gamma[9]
    return 1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3 - g14*x1*x4\
            - g22*x2*x2/2. - g23*x2*x3 - g24*x2*x4 - g33*x3*x3/2. - g34*x3*x4\
            - g44*x4*x4/2.

def residuals5d(gamma, y, x1, x2, x3, x4, x5):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g14 = gamma[3]
    g15 = gamma[4]
    g22 = gamma[5]
    g23 = gamma[6]
    g24 = gamma[7]
    g25 = gamma[8]
    g33 = gamma[9]
    g34 = gamma[10]
    g35 = gamma[11]
    g44 = gamma[12]
    g45 = gamma[13]
    g55 = gamma[14]
    return y - (1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3 - g14*x1*x4
            - g15*x1*x5 - g22*x2*x2/2. - g23*x2*x3 - g24*x2*x4 - g25*x2*x5
            - g33*x3*x3/2. - g34*x3*x4 - g35*x3*x5 - g44*x4*x4/2. - g45*x4*x5
            - g55*x5*x5/2.)

def evalfit5d(x1, x2, x3, x4, x5, gamma):
    g11 = gamma[0]
    g12 = gamma[1]
    g13 = gamma[2]
    g14 = gamma[3]
    g15 = gamma[4]
    g22 = gamma[5]
    g23 = gamma[6]
    g24 = gamma[7]
    g25 = gamma[8]
    g33 = gamma[9]
    g34 = gamma[10]
    g35 = gamma[11]
    g44 = gamma[12]
    g45 = gamma[13]
    g55 = gamma[14]
    return 1. - g11*x1*x1/2. - g12*x1*x2 - g13*x1*x3 - g14*x1*x4\
            - g15*x1*x5 - g22*x2*x2/2. - g23*x2*x3 - g24*x2*x4 - g25*x2*x5\
            - g33*x3*x3/2. - g34*x3*x4 - g35*x3*x5 - g44*x4*x4/2. - g45*x4*x5\
            - g55*x5*x5/2.


# Convenience function to return eigenvalues and eigenvectors of a matrix
def eigensystem(matrix):
    """
    Given an array-like 'matrix', returns:
        - An array of eigenvalues
        - An array of eigenvectors
        - A rotation matrix that rotates the eigenbasis
            into the original basis

    Example:
        mat = [[1,2,3], [4,5,6], [7,8,10]]
        evals, evecs, rot = eigensystem(mat)
        evals = array([ 16.70749332+0.j,  -0.90574018+0.j,   0.19824686+0.j])
        evecs = array([[-0.22351336, -0.50394563, -0.83431444],
                       [-0.86584578,  0.0856512 ,  0.4929249 ],
                       [ 0.27829649, -0.8318468 ,  0.48018951]])
        rot = array([[-0.46970759, -0.57567288, -0.7250339 ],
                     [-0.97447675, -0.12998907,  0.3395794 ],
                     [ 0.18421899, -0.86677726,  0.47420155]])

        rot.dot(evecs[0]) = [1,0,0]
        rot.dot(evecs[1]) = [0,1,0]
        rot.dot(evecs[2]) = [0,0,1]

        inv(rot).dot([1,0,0]) = evecs[0]
        inv(rot).dot([0,1,0]) = evecs[1]
        inv(rot).dot([0,0,1]) = evecs[2]

        rot.dot(mat).dot(inv(rot)) = [[evals[0], 0,        0]
                                      [0,        evals[1], 0]
                                      [0,        0,        evals[2]]]
    """
    evals, emat = eig(matrix)
    return evals, np.transpose(emat), inv(emat)

#
# Functions to return points distributed randomly, uniformly inside
# an ellipsoid of arbitrary dimension
#
def uniform_random_ellipsoid(Npts, *radii):
    """
    Return an array of pts distributed randomly and uniformly inside an
    D-dimensional ellipsoid. D is determined by the number of radii args given.

    Returned array has shape like:
    [[x1_1,x2_1, ..., xD_1],
     [x1_2,x2_2, ..., xD_2],
      ...,
    [x1_Npts,x2_Npts, ..., xD_Npts]]
    """
    D = len(radii)
    if D==2:
        return uniform_random_ellipsoid2d(Npts, *radii)
    elif D==3:
        return uniform_random_ellipsoid3d(Npts, *radii)
    elif D==4:
        return uniform_random_ellipsoid4d(Npts, *radii)
    elif D==5:
        return uniform_random_ellipsoid5d(Npts, *radii)
    else:
        raise ValueError('Not implemented for that many dimensions')

def uniform_random_ellipsoid2d(Npts, r1, r2):
    """
    2D case of uniform_random_ellipsoid
    """
    r = np.random.rand(Npts)
    th = np.random.rand(Npts) * 2.*np.pi
    rrt = np.sqrt(r)
    x1 = r1 * rrt * np.cos(th)
    x2 = r2 * rrt * np.sin(th)
    return np.transpose((x1,x2))

def uniform_random_ellipsoid3d(Npts, r1, r2, r3):
    """
    3D case of uniform_random_ellipsoid
    """
    r = np.random.rand(Npts)
    ph = np.random.rand(Npts) * 2.*np.pi
    costh = np.random.rand(Npts)*2.-1.
    sinth = np.sqrt(1.-costh*costh)
    rrt = r**(1./3.)
    x1 = r1 * rrt * sinth * np.cos(ph)
    x2 = r2 * rrt * sinth * np.sin(ph)
    x3 = r3 * rrt * costh
    return np.transpose((x1,x2,x3))

def uniform_random_ellipsoid4d(Npts, r1, r2, r3, r4):
    """
    4D case of uniform_random_ellipsoid
    """
    r = np.random.rand(Npts)
    ph = np.random.rand(Npts) * 2.*np.pi
    costh1 = np.random.rand(Npts)*2.-1.
    costh2 = np.random.rand(Npts)*2.-1.
    sinth1 = np.sqrt(1.-costh1*costh1)
    sinth2 = np.sqrt(1.-costh2*costh2)
    rrt = r**(1./4.)
    x1 = r1 * rrt * sinth1 * sinth2 * np.cos(ph)
    x2 = r2 * rrt * sinth1 * sinth2 * np.sin(ph)
    x3 = r3 * rrt * sinth1 * costh2
    x4 = r4 * rrt * costh1
    return np.transpose((x1,x2,x3,x4))

def uniform_random_ellipsoid5d(Npts, r1, r2, r3, r4, r5):
    """
    4D case of uniform_random_ellipsoid
    """
    r = np.random.rand(Npts)
    ph = np.random.rand(Npts) * 2.*np.pi
    costh1 = np.random.rand(Npts)*2.-1.
    costh2 = np.random.rand(Npts)*2.-1.
    costh3 = np.random.rand(Npts)*2.-1.
    sinth1 = np.sqrt(1.-costh1*costh1)
    sinth2 = np.sqrt(1.-costh2*costh2)
    sinth3 = np.sqrt(1.-costh3*costh3)
    rrt = r**(1./5.)
    x1 = r1 * rrt * sinth1 * sinth2 * sinth3 * np.cos(ph)
    x2 = r2 * rrt * sinth1 * sinth2 * sinth3 * np.sin(ph)
    x3 = r3 * rrt * sinth1 * sinth2 * costh3
    x4 = r4 * rrt * sinth1 * costh2
    x5 = r5 * rrt * costh1
    return np.transpose((x1,x2,x3,x4,x5))
