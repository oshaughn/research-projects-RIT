"""
regions.py
     Provides 'Region' class
      - general purpose 'set' of points  (intrinsic space)
      - on creation
            - hardcodes variable names (list of dimensions)
      - takes a bounding region (e.g, ellipsoid)

     Relies on 'mcsampler.py' to provide integration
"""

import numpy as np
import functools
import RIFT.integrators.mcsampler as mcsampler
import random

default_sampling_neff = 2000


class Region(object):
    """
    Base class for region object
    """

    def __init__(self, plist):
        """
        Create with *ordered* list of parameter names and sanity/valid region limits
        """
        # Parameter names
        self.params = plist[:,0]
        self.llim = plist[:,1]
        self.rlim = plist[:,2]

    def clear(self):
        """
        Clear out the parameters
        """
        self.params = set()

    def inside_q(self,pt):
        """
        Test if point inside region
        """
        raise Exception("This is the base Region class! Use a subclass")

    def draw(self,npts):
        """
        Draw npts random elements from the region"
        """
        raise Exception("This is the base Region class! Use a subclass")
        
    def grid(self,npts):
        """
        Return a grid of parameters for the region
             [x1,x2,xn]
        where each xn is a d-dimensional point in the coordinate system
        """
        raise Exception("This is the base Region class! Use a subclass")


    def grid_native(self,npts):
        """
        Returns the same points as 'grid', but in  "native" coordinates for the region
             [z1, ... zn]
        where each zn is a d-dimenional vector.
        """
        raise Exception("This is the base Region class! Use a subclass")


    def integrate(self,f,measure):
        """
        Returns \int f measure dx.  Functions 'f' and 'measure' must be defined using the global coordinate system.
        Implemented via mcsampler
        """
        raise Exception("This is the base Region class! Use a subclass")

    def sample_and_marginalize(self,fnX, f,measure):
        """
        Returns array of posterior samples of 'fnX', where fnX is any function of the coordinates.
        Useful to produce marginalized 'intrinsic' distributions
        """
        raise Exception("This is the base Region class! Use a subclass")


class RegionBox(Region):
    """
    Hypercube class.
    """

    def inside_q(self,pt):
        return all(pt > self.llim) and all(pt < self.rlim)

    def inside_q_list(self,pt_list):
        testvals = map(self.inside_q, pt_list.T)
        return map(self.inside_q, pt_list.T)


    def integrate(self,f, measure):
        samp = mcsampler.MCSampler()
        for indx in np.arange(len(self.params)):
            sampler.add_parameter(self.params[indx], functools.partial(mcsampler.uniform_samp_vector, self.llim[indx], self.rlim[indx]), None, self.llim[indx], self.rlim[indx])

        raise Exception("Not yet implemented")
            
        

class RegionEllipse(Region):

    def __init__(self, plist,center=np.array([0,0]),mtx=np.array([[1,0],[0,1]])):
        """
        *Two-dimensional* ellipse
        Create with *ordered* list of TWO names and sanity/valid region limits, PLUS
              - center  (default = [0,0])
              - mtx    (defaut =identity)  
        Ellipse boundary is defined as
                 (pt-center).mtx.(pt-center) = 1
        """
        # Parameter names
        assert len(plist)==2
        self.params = [k[0] for k in plist]
        # Ranges. 
        self.llim = [k[1] for k in plist]
        self.rlim = [k[2] for k in plist]
        assert all(np.array([self.llim < self.rlim]))   # sanity check -- just in case someone uses an integer divide, for example
        # Ellipse location
        self.center = center
        self.mtx = np.matrix(mtx)
        # Eigenvalues and eigenvectors of the mtx : important for 'intrinsic' polar coordinates
        self._eigenvalues, self._eigenvectors, self._rot= eigensystem(mtx) #np.linalg.eig(mtx)

    def convert_internal_to_global(self,pt_internal):
        """
        Converts 'polar' coordinates (internal for ellipse) to the 'cartesian' coordinates of the underlying system, including offset
        """
        pt = self.center + pt_internal[0]*(
            np.sqrt(self._eigenvalues[0])*self._eigenvectors[0]*np.cos(pt_internal[1])
            +  np.sqrt(self._eigenvalues[1])*self._eigenvectors[1]*np.sin(pt_internal[1])
            )
        return pt

    def convert_global_to_internal(self,pt):
        """
        Converts cartesian coordinates to 'ellipse' polar coordinates
        """
        xval = pt - self.center
        # 'Cartesian' coordinates, rescaled to principal axes of ellipse
        x0 = np.dot(xval, self._eigenvectors[0])*np.sqrt(self._eigenvalues[0])
        x1 = np.dot(xval, self._eigenvectors[1])*np.sqrt(self._eigenvalues[1])
        # Polar coordinates derived from principal axes
        r = np.sqrt(x0*x0+x1*x1)   # =xval.mtx.xval
        th = np.arctan2(x1,x0)
        return np.array([r,th])
        

    def inside_q(self,pt):
        assert pt.shape == (2,)
        # sanity test
        if not ( all(pt > self.llim) and all(pt < self.rlim)):
            return False
        # test if inside ellipse
        offset = np.matrix(pt - self.center)
        radius = ((offset)*self.mtx*offset.T)[0,0]
        return radius <1

    def inside_q_list(self,pt_list):
        testvals = map(self.inside_q, pt_list.T)
        return map(self.inside_q, pt_list.T)

    def draw(self,npts):
        # Brute force: use large-scale cartesian region, then select subregion via rejection sampling.  Can be *incredibly* inefficient in high-d or with a wide 'box' surrounding the ellipse
        pts = []
        nEval = 0
        while len(pts)<npts and nEval < 1e6:
            new_point  = np.array([random.uniform(self.llim[k], self.rlim[k]) for k in np.arange(len(self.rlim))])
            if self.inside_q(new_point):
                pts=pts+[new_point]
            nEval +=1
        return np.array(pts)

    # Brute force integration, not using ellipsoidal structure/coordinates
    def integrate(self,f, measure, verbose=False,adaptive=False,neff=default_sampling_neff):
        # mcsampler *requires* parameter names tied to the function -- incredibly irritating. Hence 'x','y'
        sampler = mcsampler.MCSampler()
        # Default procedure: cartesian coordinates over the whole grid (no change of coordinates)
        # Convert back to *coordinate measure* -- i.e., correct for the fact that use a PDF normalized on a square.  (should use prior_pdf=1...)
        sampler.add_parameter('x', 
               functools.partial(mcsampler.uniform_samp_vector, self.llim[0], self.rlim[0]), 
               left_limit= self.llim[0],right_limit=self.rlim[0],
               prior_pdf= lambda x: np.ones(x.shape),
               adaptive_sampling=adaptive             
                              )
        sampler.add_parameter('y', 
               functools.partial(mcsampler.uniform_samp_vector, self.llim[1], self.rlim[1]), 
               left_limit= self.llim[1],right_limit=self.rlim[1],
               prior_pdf=lambda x: np.ones(x.shape),
               adaptive_sampling=adaptive             
                              )
        # Define a function that is !=0 only inside the region.  Implicitly assume vectorized arguments (mcsampler)
        fn = lambda x,y:  f(np.array([x,y]))*measure(np.array([x,y])) if self.inside_q_list(np.array([x,y])) else 0        
        fnVec = lambda x,y:  np.array([f(np.array([x[k],y[k]]))*measure(np.array([x[k],y[k]])) if self.inside_q(np.array([x[k],y[k]])) else 0 for k in np.arange(x.shape[0])])
        # Integrate
        res, var, neff, dict_return  = sampler.integrate(fnVec, self.params[0], self.params[1], neff=neff, nmax=int(1e6),verbose=verbose)
        return res,np.sqrt(var) #* (self.rlim[0]-self.llim[0])*(self.rlim[1]-self.llim[1])



    def sample_and_marginalize(self,fnX, f,measure,verbose=False,adaptive=False,sort_by_parameter=True, sort_by_weight=False,neff=default_sampling_neff):
        """
        Returns array of pairs (fnX_k,w_k, P(<k)), where fnX is any function of the coordinates; w_k are weights; and P is the cumulative probability
        Array can be pre-sorted and the cumulative constructed in 
               - increasing parameter order (*only* if fnX returns scalars)
               - increasing *weight* order.
        Useful to produce marginalized 'intrinsic' distributions
        """

        # mcsampler *requires* parameter names tied to the function -- incredibly irritating. Hence 'x','y'
        sampler = mcsampler.MCSampler()
        # Default procedure: cartesian coordinates over the whole grid (no change of coordinates)
        # Convert back to *coordinate measure* -- i.e., correct for the fact that use a PDF normalized on a square.  (should use prior_pdf=1...)
        sampler.add_parameter('x', 
               functools.partial(mcsampler.uniform_samp_vector, self.llim[0], self.rlim[0]), 
               left_limit= self.llim[0],right_limit=self.rlim[0],
               prior_pdf= lambda x: np.ones(x.shape),
               adaptive_sampling=adaptive             
                              )
        sampler.add_parameter('y', 
               functools.partial(mcsampler.uniform_samp_vector, self.llim[1], self.rlim[1]), 
               left_limit= self.llim[1],right_limit=self.rlim[1],
               prior_pdf=lambda x: np.ones(x.shape),
               adaptive_sampling=adaptive             
                              )
        # Define a function that is !=0 only inside the region.  Implicitly assume vectorized arguments (mcsampler)
        fn = lambda x,y:  f(np.array([x,y]))*measure(np.array([x,y])) if self.inside_q_list(np.array([x,y])) else 0        
        fnVec = lambda x,y:  np.array([f(np.array([x[k],y[k]]))*measure(np.array([x[k],y[k]])) if self.inside_q(np.array([x[k],y[k]])) else 0 for k in np.arange(x.shape[0])])
        # Integrate
        res, var, neff, dict_return  = sampler.integrate(fnVec, self.params[0], self.params[1], neff=neff, nmax=int(1e6),verbose=verbose,save_intg=True)

        # Construct weights
        weights = sampler._rvs["integrand"]*sampler._rvs["joint_prior"]/sampler._rvs["joint_s_prior"]

        # Construct fnX values
        xvals = np.array([fnX( [sampler._rvs['x'][k],sampler._rvs['y'][k]]) for k in np.arange(len(weights))])

        if not(sort_by_parameter or  sort_by_weight):
            return np.array([weights,xvals])
        else:
            # Sort  arrays;  evaluate the cumulative sum
            if sort_by_weight:
                idx_sorted_index = np.lexsort((np.arange(len(weights)), weights))  # Sort the array of weights, recovering index values
            elif sort_by_parameter and len(xvals.shape) is 1:
                idx_sorted_index = np.lexsort((np.arange(len(weights)), xvals))  # Sort the array of x values, recovering index values
            else:
                raise "Cannort sort output values by multidimensional parameter fnX"
            indx_list = np.array( [[k, weights[k], xvals[k]] for k in idx_sorted_index])   # pair up with the weights again
            cum_sum = np.cumsum(indx_list[:,1])                                     # find the cumulative sum
            cum_sum = cum_sum/cum_sum[-1]                                         # normalize the cumulative sum
            return np.array([indx_list[:,2],indx_list[:,1], cum_sum]).T



####
#### SUPPORT CODE
####

#
def hyperunit(pt):
    """
    hypersphere unit vector  (r,th1,th2,...th(n-1)), expressed in cartesian coordinates.
          x/r = ( \cos \th1, \sin \th1 \cos \th2, .... \sin th1 ... \sin \thn)
    Note that this convention does *not* agree with x,y,z cartesian polar convention
    """
    # Loop, making product
    # addition operations are *list* operations (prepend)
    d = len(pt)
    ang = np.array(pt)[1:]   # drop first element
    xvec = []
    while len(ang) > 0:
        if len(ang)>1:
            prod_rest = np.prod(np.sin(ang[1:]))
        else:
            prod_rest=1
        xvec = xvec + [np.cos(ang[0])*prod_rest]

        if len(ang)>1:
            ang = ang[1:]
        else:
            ang = []

    xvec =xvec+[np.prod(np.sin(pt[1:]))]
    return xvec
#
# Routines to make various types of grids for arbitrary dimension
# Duplicated from 'effectiveFisher.py'  
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
    X
        [1,1,1,3,3,3,5,5,5]
    Y
        [2,4,6,2,4,6,2,4,6]
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


# Convenience function to return eigenvalues and eigenvectors of a matrix
def eigensystem(matrix):
    """
    Given an array-like 'matrix', returns:
        - An array of eigenvalues
        - An array of eigenvectors
        - A rotation matrix that rotates the eigenbasis
            into the original basis

    Example:
        mat = [[1,2,3],[2,4,5],[3,5,6]]
        evals, evecs, rot = eigensystem(mat)
        evals
            array([ 11.34481428+0.j,  -0.51572947+0.j,   0.17091519+0.j]
        evecs
            array([[-0.32798528, -0.59100905, -0.73697623],
                   [-0.73697623, -0.32798528,  0.59100905],
                   [ 0.59100905, -0.73697623,  0.32798528]])
        rot
            array([[-0.32798528, -0.73697623,  0.59100905],
                   [-0.59100905, -0.32798528, -0.73697623],
                   [-0.73697623,  0.59100905,  0.32798528]]))

    This allows you to translate between original and eigenbases:

        If [v1, v2, v3] are the components of a vector in eigenbasis e1, e2, e3
        Then:
            rot.dot([v1,v2,v3]) = [vx,vy,vz]
        Will give the components in the original basis x, y, z

        If [wx, wy, wz] are the components of a vector in original basis z, y, z
        Then:
            inv(rot).dot([wx,wy,wz]) = [w1,w2,w3]
        Will give the components in the eigenbasis e1, e2, e3

        inv(rot).dot(mat).dot(rot)
            array([[evals[0], 0,        0]
                   [0,        evals[1], 0]
                   [0,        0,        evals[2]]])

    Note: For symmetric input 'matrix', inv(rot) == evecs
    """
    evals, emat = np.linalg.eig(matrix)
    return evals, np.transpose(emat), emat
