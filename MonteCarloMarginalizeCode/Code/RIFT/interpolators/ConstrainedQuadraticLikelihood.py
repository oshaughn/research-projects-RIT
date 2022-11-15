#
#
#  based on 'gwalk'
#  Fit a quadratic to likelihoods (including overall likelihood factor).
#  Allow code to specify a guess based on fitting a gaussian to it, for speed
import numpy as np
import numpy.linalg as la
import scipy.linalg as linalg

try:
    from gwalk.utils.multivariate_normal import mu_of_params,cov_of_params, params_of_mu_cov
except:
    print(" - no gwalk - ")
#from gwalk.density.grid import Grid
import scipy.optimize


import RIFT.interpolators.BayesianLeastSquares as BayesianLeastSquares


def fit_quadratic_alt(x,y,y_err=None,x0=None,symmetry_list=None,verbose=False,hard_regularize_negative=True):
    gamma_x = None
    if not (y_err is None):
        gamma_x =np.diag(1./np.power(y_err,2))
    the_quadratic_results = BayesianLeastSquares.fit_quadratic( x, y,gamma_x=gamma_x,verbose=verbose,hard_regularize_negative=hard_regularize_negative)#x0=None)#x0_val_here)
    peak_val_est, best_val_est, my_fisher_est, linear_term_est,fn_estimate = the_quadratic_results
    cov = np.linalg.inv(my_fisher_est)
    return best_val_est, cov


def quad_residuals(x,yvals,lnL_offset,mu,icov):
#    print(lnL_offset, mu, icov)
    yvals_expected = np.zeros(len(yvals))
    for indx in np.arange(len(yvals)):
        yvals_expected[indx] = lnL_offset - 0.5* np.dot((x[indx]-mu), np.dot(icov,x[indx]-mu))
    return np.sum((yvals - yvals_expected)**2)  # least square residual, quadratic fit

# From https://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-dot-product-calculating-only-the-diagonal-entries-of-the
def quad_residuals_vector(x,yvals,lnL_offset,mu,icov):
    yvals_expected = np.zeros(len(yvals)) + lnL_offset
    dx = x - mu[:,np.newaxis].T
    # now only take the diagonal elements of this matrix: I only want to correlate an x_k with itself, not with off-
    tmp = np.einsum('ij,jk,ki->i',dx,icov,dx.T)
    yvals_expected += -0.5* tmp
    return np.sum((yvals - yvals_expected)**2)  # least square residual, quadratic fit


def fit_grid(
             sample,
             values,
             extra_guess_mu=None,
             extra_guess_cov=None,
             **kwargs
            ):
    ''' Fit multivariate normal object to samples with values.  Note the fit assumes the data is PERFECTLY quadratic,
       and ENFORCES a nonnegative-definite parameterization, so make sure you curate the data to make that possible

    Parameters
    ----------
    sample: array like, shape = (npts, ndim)
        Input sample locations
    values: array like, shape = (npts,)
        Input sample values
    limits: array like, shape = (ndim, 2)
        Input bounded interval limits
    seed: int, optional
        Input seed for random numbers
    nwalk: int, optional
        Input number of random walkers
    nstep: int, optional
        Input number of walker steps
    carryover: float, optional
        Input fraction of best guesses carried over through genetic algorithm
    sig_factor: float, optional
        Input controls jump sizes
        '''

    # Seed parameters
    #   lnL, mu, std, cov   following Vera
    if extra_guess_mu is None:
        extra_guess_mu, extra_guess_cov = fit_quadratic_alt(sample,values)

    X_alt = np.array(list([np.max(values)])+list(params_of_mu_cov(extra_guess_mu,extra_guess_cov)[0]))
#    print(X_alt)
    # define objective
    def my_objective(X):
        lnL_max = X[0]
        mu =  mu_of_params(X[1:])[0]
        cov = cov_of_params(X[1:])[0]
        icov = linalg.inv(cov)
        return quad_residuals_vector(sample, values, lnL_max, mu, icov)

    res = scipy.optimize.minimize(my_objective, X_alt)
#    print(res)
    X_out = res.x
#    print(X_out)
    lnLmax = X_out[0]
    mu_fit = mu_of_params(np.array([X_out[1:]]))[0]  # could do this myself
    cov_fit = cov_of_params(np.array([X_out[1:]]))[0]
    return(lnLmax, mu_fit,cov_fit)
