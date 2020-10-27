#!usr/env/bin python
'''
GPR_Fit.py
python 2.7.13
Vera Delfavero
Date updated: 02/02/2019

The function of this script is to provide the function, "fit_gp" to
the PE code (utils_ConstructIntrinsicPosterior_GenericCoordinates.py)

That function will fit training data, using a sparse gaussian proces,
and return a function which takes only test data as an input, 
and returns only the estimated mean. 
(Because that's all the PE code uses)
This saves a lot of time, because calculating the covariance
of the test data is expensive.
This function is also a sparse gaussian proces, which should make
it at least an order of magnitude faster than the sklearn 
gaussian process code.

The basic algorithm for this can be found on page 88 of R&W's
book on Gaussian Processes and Machine Learning. 
'''
from __future__ import print_function, division, unicode_literals
######## Kernels ########

from scipy.sparse import csc_matrix, linalg as sla

def compact_kernel(Xp, Xq,
                    thetas,
                    white_noise,
                    ):
    '''
    compact_kernel: Return a sparse representation of the 
        compact kernel described on page 88 of R&W.
        Most of the functions which generate the sparse kernel
        involve non-sparse matricies.

    Inputs:
        Xp: Dense array containing input vector
        Xq: Dense array containing additional input vector
            Note: if Xq is given, the returned kernel is the
                transpose of the true kernel
        thetas: Hyperparameters
            thetas[0]: Scale length used to construct sparse matrix.
    Outputs:
        K: Sparse kernel representation (csc matrix)
    '''
    import numpy as np
    
    # If no Xq, make one
    if Xq is None:
        Xq = np.copy(Xp)

    # Copy Xp
    Xp = np.copy(Xp).astype(float)
    Xq = np.copy(Xq).astype(float)

    # Identify dimensionality
    ndim = len(thetas)
    j = np.floor(float(ndim)/2.0) + 2

    # Scale by thetas
    if len(thetas) >= 1:
        for i in range(ndim):
            Xp[:,i] = Xp[:,i]/thetas[i]
            Xq[:,i] = Xq[:,i]/thetas[i]

    # Compute the difference matrix
    x2 = ((np.ones((Xq.shape + (Xp.shape[0],)))*Xp.T).T)
    x1 = (np.ones((Xp.shape + (Xq.shape[0],)))*Xq.T)

    # Assumption: real X and real k
    r = np.linalg.norm(x2 - x1, axis = 1)

    # K = ((1 - r)^2)_+
    eta = (1.0 - r)
    eta[eta < 0.0] = 0.0
    #K = eta**2
    K = (eta**(j + 1))*((j + 1)*r + 1.0)

    if white_noise != 0.0:
        K += white_noise*np.eye(K.shape[0], K.shape[1])

    # Construct sparse matrix
    K = csc_matrix(K)

    return K

######## Fit Function ########

def GP_fit_function(x_train, y_train, thetas, white_noise,
                    kernel_function = compact_kernel,
                    **kwargs):
    '''
    Objective: return a function which can sample the distribution
        with only the test inputs as an input.
        This function only needs to return a mean.
    '''
    from sksparse.cholmod import cholesky
    ## Guarantee dimensionality ##
    ## Find initial kernel ##
    K = kernel_function(x_train, x_train, thetas, white_noise)
    # find the alphas using cholesky decomposition
    L_factor = cholesky(K)
    alpha = L_factor(y_train)

    ## Construct function ##
    def my_fit(x_sample):
        #print(len(x_sample))
        Kt = kernel_function(x_sample, x_train, thetas, white_noise = 0.0)
        y_mean = Kt.dot(alpha)
        return y_mean

    return my_fit
    
######## Application for PE code ########

def fit_gp(x_train, y_train, y_errors=None,
        sparse_coeff = 3.0,
        white_noise = 0.01
        ):
    """
    This function can directly replace fit_gp in 
    utils_ConstructIntrinsicPosterior_GenericCoordinates.py

    It is an implimentation of my sparse fitting code,
    however most of this function was copied from the original.

    Inputs:
        x_train (numpy.ndarray): 
            Training data for model
            shape: (m, n) where m is the number of points,
                and n is the dimensionality
        y_train (numpy.ndarray): 
            Training data for model
            shape: (m, ) where m is the number of points
        sparse_coeff (float):
            This hyperparameter determines the sparseness of the matrix.
            A sparse_coeff of 1. is not sparse.
            A sparse_coeff of inf is diagonal.
            What value you can use for this is largely dependent on how
                much training data you have, as well as the dimensionality
                of your data
            About each dimension in parameter space, the scale length is
                calculated for that dimension as h/hs, where h is the range
                of the training data in that dimension.
        white_noise (float):
            The white noise hyperparemeter
    Outputs:
        my_fit (function(x_sample)):
            A function which will sample our model on x_sample,
                for an x_sample like x_train
    """
    import numpy as np
    # Find the dimensionality of the data from the shape of x_train
    n_dim = len(x_train[0])
    # Set the scale for each dimension
    h = np.empty(n_dim)
    for i in range(n_dim):
        h[i] = np.max(x_train) - np.min(x_train)
    thetas = h/sparse_coeff

    print(" GP: Input sample size ", x_train.shape)
    print(" Sparse scale lengths: ", thetas)

    ## Create function ##
    my_fit = GP_fit_function(x_train, y_train, thetas, white_noise)
    print(" std: ", np.std(my_fit(x_train) - y_train))

    return my_fit
