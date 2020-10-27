#!
#
# REFERENCES
#    util_ManualOverlapGrid.py
#    20160906-Pankow-ProgressPortingNRVariantToMaster/util_ManualOverlapGrid.py
#

import scipy.linalg as linalg
import numpy as np


def fit_quadratic(x,y,x0=None,variable_symmetry_list=None,gamma_x=None,prior_x_gamma=None,prior_quadratic_gamma=None,verbose=False,n_digits=None,hard_regularize_negative=False,hard_regularize_scale=1):
    """
    Simple least squares to a quadratic.
    Written out in long form so I can impose priors as needed on (a) the fit coefficients [=regularization] and (b) the 'x' coordinates.
    INPUT: 
        x = array so x[0] , x[1], x[2] are points.
        y = array of y values
    OUTPUT
        peak_val_est,
        best_val_est, 
        my_fisher_est, 
        linear_term_est,
        fit_here     : python function providing best fit.  Usually what you want, but cannot be pickled/saved to text
    OPTIONAL
        variable_symmetry_list =  list of length x, indicating symmetry under ONE discrete symmetry (so far)
    """
    x0_val = np.zeros(len(x[0]))
    if not (x0 is None):
        if verbose:
            print(" Fisher: Using reference point ", x0)
        x0_val = x0

    dim = len(x[0])   
    npts = len(x)
    if verbose:
        print(" Fisher : dimension, npts = " ,dim, npts)
    # Constant, linear, quadratic functions. 
    # Beware of lambda:  f_list = [(lambda x: k) for k in range(5)] does not  work, but this does
    #     f_list = [(lambda x,k=k: k) for k in range(5)]
    f0 = [lambda z: np.ones(len(z),dtype=np.float128)]
    # indx_lookup_linear = {}   # protect against packing errors
    # indx_here = len(f0)
    # f_linear = []
    # for k in np.arange(dim):
    #     f_linear.append( (lambda z,k=k,x0V=x0_val: z.T[k] - x0V[k]))
    #     indx_lookup_linear[k] =indx_here
    #     indx_here+=1
    f_linear = [(lambda z,k=k,x0V=x0_val: z.T[k] - x0V[k]) for k in np.arange(dim)]
    f_quad = []
    indx_lookup = {}
    indx_here =len(f0)+len(f_linear) 
    for k in np.arange(dim):
        for q in range(k,dim):
            if variable_symmetry_list:
                if variable_symmetry_list[k]*variable_symmetry_list[q] <0: 
                    if verbose:
                        print(" Not including quadratic term because of symmetry", (k,q))
                    continue  # skip the remaining part
            f_quad.append( (lambda z,k=k,q=q: (z.T[k] - x0_val[k])*(z.T[q]-x0_val[q]))   )
            indx_lookup[(k,q)] = indx_here
            indx_here+=1
    f_list=f0+f_linear + f_quad
    n_params_model = len(f_list)
    if verbose:
        print(" ---- Dimension:  --- ", n_params_model)
        print(" ---- index pattern (paired only; for manual identification of quadratic terms) --- ")
        print(indx_lookup)
    # if verbose:
    #     print " ---- check quadratic --- "
    #     for pair in indx_lookup:
    #         fn_now = f_list[indx_lookup[pair]]
    #         print " Grid test " , pair,  fn_now(np.array([1,0])), fn_now(np.array([0,1])), fn_now(np.array([1,1])) ,fn_now(np.array([1,-1])) 


    F = np.matrix(np.zeros((len(x), n_params_model),dtype=np.float128))
    for q in np.arange(n_params_model):
        fval = f_list[q](np.array(x,dtype=np.float128))
        F[:,q] = np.reshape(fval, (len(x),1))
    gamma = np.matrix( np.diag(np.ones(npts,dtype=np.float128)))
    if not(gamma_x is None):
        gamma = np.matrix(gamma_x)
    Gamma = F.T * gamma * F      # Fisher matrix for the fit
    Sigma = linalg.inv(Gamma)  # Covariance matrix for the fit. WHICH CODE YOU USE HERE IS VERY IMPORTANT.
    # if verbose:
    #     print " -- should be identity (error here is measure of overall error) --- "
    #     print "   Fisher: Matrix inversion/manipulation error ", np.linalg.norm(Sigma*Gamma - np.eye(len(Sigma))) , " which can be large if the fit coordinates are not centered near the peak"
    #     print " --  --- "
    lambdaHat =  np.array((Sigma* F.T*gamma* np.matrix(y).T))[:,0]  # point estimate for the fit parameters (i.e., fisher matrix and best fit point)
    if n_digits:
        lambdaHat = np.array(map(lambda z: round(z,n_digits),lambdaHat))
    if verbose:
        print(" Fisher: LambdaHat = ", lambdaHat)
    if verbose:
        print(" Generating predictive function ")
    def fit_here(x):
        return  np.sum(list(map(lambda z: z[1]*z[0](x), zip(f_list,lambdaHat) )),axis=0)
    if verbose:
        my_resid = y - fit_here(x)
        print(" Fisher: Residuals ", np.std(my_resid))

    ###
    ### Reconstructing quadratic terms: a bonus item
    ###
    constant_term_est = lambdaHat[0]  # Constant term
    linear_term_est = lambdaHat[1:dim+1]  # Coefficient of linear terms
    my_fisher_est = np.zeros((dim,dim),dtype=np.float64)   #  A SIGNIFICANT LIMITATION...
    for pair in indx_lookup:
        k = pair[0]; q=pair[1];
        indx_here = indx_lookup[pair]
        my_fisher_est[k,q] += -lambdaHat[indx_here]
        my_fisher_est[q,k] += -lambdaHat[indx_here]  # this will produce a factor of 2 if the two terms are identical
    if not(prior_x_gamma is None) and (prior_x_gamma.shape == my_fisher_est.shape):
        my_fisher_est += prior_x_gamma
    if verbose:
        print("  Fisher: ", my_fisher_est)
        print("  Fisher: Sanity check (-0.5)*Fisher matrix vs components (diagonal only) : ", -0.5*my_fisher_est, "versus",  lambdaHat)
    my_fisher_est_inv = linalg.inv(my_fisher_est)   # SEE INVERSE DISCUSSION
    if verbose:
        print(" Fisher: Matrix inversion/manipulation error test 2", np.linalg.norm(np.dot(my_fisher_est,my_fisher_est_inv) - np.eye(len(my_fisher_est))))
    peak_val_est = float(constant_term_est) +np.dot(linear_term_est, np.dot(my_fisher_est_inv,linear_term_est))/2
    best_val_est = x0_val +  np.dot(my_fisher_est_inv,linear_term_est)   # estimated peak location, including correction for reference point
    if verbose:
        print(" Fisher : Sanity check: peak value estimate = ", peak_val_est, " which arises as a delicate balance between ",  constant_term_est, " and ",  np.dot(linear_term_est, np.dot(my_fisher_est_inv,linear_term_est))/2)
        print(" Fisher : Best coordinate estimate = ", best_val_est)
        print(" Fisher : eigenvalues (original) ", np.linalg.eig(my_fisher_est))

    if hard_regularize_negative:
        w,v = np.linalg.eig(my_fisher_est)
        indx_neg = w<0
        # usually we are regularizing placements in spin ... this provides us with error in that dimension
        w[indx_neg] = hard_regularize_scale # 1./np.min( np.std(x,axis=0))**2   # use scatterplot of input points to set scale of this dimension

        my_fisher_est = np.dot(v.T,np.dot(np.diag(w),v))  # reconstruct matrix, after regularization
        

    return [peak_val_est, best_val_est, my_fisher_est, linear_term_est,fit_here]



def fit_quadratic_and_resample(x,y,npts,rho_fac=1,x0=None,gamma_x=None,prior_x_gamma=None,prior_quadratic_gamma=None,verbose=False,n_digits=None,hard_regularize_negative=False,hard_regularize_scale=1):
    """
    Simple least squares to a quadratic, *and* resamples from the quadratic derived from the fit.
    Critical for iterative evaluation of 
        - Fisher  matrix
        - lnLmarg (ILE)
    TO DO:
         - implement non-stochastic placement as option (e.g., like effectiveFisher.py)
    """
    # Find the fit
    the_quadratic_results = fit_quadratic(x,y,x0=x0,gamma_x=gamma_x,prior_x_gamma=prior_x_gamma,prior_quadratic_gamma=prior_quadratic_gamma,n_digits=n_digits,hard_regularize_negative=hard_regularize_negative,hard_regularize_scale=hard_regularize_scale)
    peak_val_est, best_val_est, my_fisher_est, linear_term_est,fit_here = the_quadratic_results


    # Use the inverse covariance mattrix
    my_fisher_est_inv = linalg.pinv(my_fisher_est)   # SEE INVERSE DISCUSSION
    x_new = np.random.multivariate_normal(best_val_est,my_fisher_est_inv/(rho_fac*rho_fac),size=npts)

    return x_new




if __name__ == "__main__":
    import argparse
    import sys
    import numpy as np


    print(" Testing quadratic fit code ")

    print(" Two dimensions ")
    x1 = np.linspace(-5,5,40)
    x2 = np.linspace(-1,1,10)
    x1v,x2v = np.meshgrid(x1,x2) # 
    x1f = np.ravel(x1v)
    x2f = np.ravel(x2v)
    y = -1*((x1f-0.5)*(x1f-0.5) + 2*x2f*x2f) + 1
#    y= -x1f*x1f
    x = np.array([x1f,x2f]).T
      
    the_quadratic_results = fit_quadratic( x,y,verbose=True,x0=np.array([0,0]),n_digits=5 )
    print(the_quadratic_results)

    print(" Two dimensions, imposing symmetry ")
    the_quadratic_results = fit_quadratic( x,y,verbose=True,x0=np.array([0,0]),n_digits=5,variable_symmetry_list=[1,-1] )
    print(the_quadratic_results)


    print(" One dimesion ")
    x = np.linspace(-5,5,10)
    x_mtx = np.zeros((len(x),1))
    x_mtx[:,0] = x
    y = x*x - 2*x +1   # = (x-1)^2 
    the_quadratic_results = fit_quadratic( x_mtx,y,verbose=True)
    

    print(" Two dimensions, resampling ")
    x1 = np.linspace(-5,5,40)
    x2 = np.linspace(-1,1,10)
    x1v,x2v = np.meshgrid(x1,x2) # 
    x1f = np.ravel(x1v)
    x2f = np.ravel(x2v)
    y = -1*(100*(x1f-0.5)*(x1f-0.5) + 100*x2f*x2f) + 1
    x = np.array([x1f,x2f]).T
      
    x_new = fit_quadratic_and_resample(x,y,10)
    print(x_new)
    print(np.mean(x_new[:,0]), np.std(x_new[:,0]), " Should be 0.5, ~ 0.1 ")
    print(np.mean(x_new[:,1]), np.std(x_new[:,1]), " Should be 0, ~ 0.1")


