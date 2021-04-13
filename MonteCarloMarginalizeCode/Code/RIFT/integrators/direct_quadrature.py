#! 
# USAGE
#    - directly integrate over variables.  Used for distance, phase, polarization maximization.  Typically uniform grid quadrature.
# 
#

from  scipy.integrate import quad, dblquad, tplquad, nquad
import sys
import numpy as np

def marginal_function(f, x_rest,  args_and_ranges,g=None,epsrel=1e-5,method='default'):
    """
       f: function, assumed vectorized
       x_rest: variable with correct range for f (i.e., f(x_rest) is valid).  x_rest assumed *array* of options
       args_and_ranges:  list of [ indx, [low,high],type] showing indexes of parameters and ranges.   Usually do one, two, or three dimensions. 'type' =['uniform',npts] with npts=100 is default
       g : optional additional function used to specify priors, if not uniform

       Returns:  \int f(x1,x2,....xn) g(x1) dx1     if one variable provided, etc
    """

    x = np.array([x_rest[0].copy()])  # copy initial parameters for range

    if not(g==None):
        print("Priors not yet implemented")
        sys.exit(1)

    if len(args_and_ranges) >3:
        print(" Not yet implemented for full multimensional integration")
        sys.exit(1)

    if len(args_and_ranges) ==1 and method=='default':
        indx = args_and_ranges[0][0]
        x_low = args_and_ranges[0][1][0]
        x_high = args_and_ranges[0][1][1]
        my_out = np.zeros(len(x_rest))
        for label in np.arange(len(x_rest)):
            x = np.array([x_rest[label].copy()])  # copy initial parameters for range
            def my_func(y):
                #            print(y)
                x[:,indx]=y
                return f(x)
            my_out[label] = quad(my_func, x_low, x_high,epsrel=epsrel)[0]  # scalar return value : should loop over all choices for x
        return my_out

    if len(args_and_ranges) ==2 and method=='default':
        indx_X = args_and_ranges[0][0]
        indx_Y = args_and_ranges[1][0]
        x_low = args_and_ranges[0][1][0]
        x_high = args_and_ranges[0][1][1]
        y_low = args_and_ranges[1][1][0]
        y_high = args_and_ranges[1][1][1]
        my_out = np.zeros(len(x_rest))
        for label in np.arange(len(x_rest)):
            x = np.array([x_rest[label].copy()])  # copy initial parameters for range
            def my_func(Y,X): # note definintion of dblquad
                #            print(y)
                x[:,indx_X]=X
                x[:,indx_Y]=Y
                return f(x)
            my_out[label] = dblquad(my_func, x_low, x_high,y_low, y_high,epsrel=epsrel)[0]  # scalar return value : should loop over all choices for x
        return my_out
