
import numpy as np
import scipy.integrate

xpy_default=np
has_cupy = False
try:
    import cupy   # can proceed even if cupy doesn't actually work
    junk_to_check_installed = cupy.array(5)
    xpy_default=cupy
    has_cupy=True
except:
    has_cupy=False  # just to be sure
# try:
#     import numba
#     from numba import vectorize, complex128, float64, int64
#     numba_on = True
#     print(" Numba on (priors_utils) ")
    

# except:
#     numba_on = False
#     print(" Numba off (priors_utils) ")

# https://git.ligo.org/RatesAndPopulations/lalinfsamplereweighting/blob/reviewed-post-O2/approxprior.py
will_cosmo_const = np.array( [ 1.012306, 1.136740, 0.262462, 0.016732, 0.000387 ])
p_in = will_cosmo_const[::-1]
if has_cupy:
    p_in = cupy.asarray(p_in)
def dist_prior_pseudo_cosmo(dL,nm=1,xpy=np,p_in=p_in):
    """
    dist_prior_pseudo_cosmo.  dL needs to be in Gpc for the polynomial.
    By default, our code works with distances in Mpc.  So we divide by 1e3

     Will Farr's simplified distance prior on d_L, out to z~ 4
     note it is not normalized, and the normalization depends on the d_max of interest 
    
    """
    if isinstance(dL, np.ndarray):
        return nm*4* np.pi * dL**2 / np.polyval( will_cosmo_const[::-1], dL/1e3)
    return nm*4* np.pi * dL**2 / xpy.polyval( p_in,dL/1e3)


def dist_prior_pseudo_cosmo_eval_norm(dLmin,dLmax):
    return 1./scipy.integrate.quad(dist_prior_pseudo_cosmo, dLmin,dLmax)[0]


## VARIOUS COSMOLOGY THINGS

def get_astropy_cosmology(name="Planck15"):
   from astropy import cosmology as cosmo
   return getattr(cosmo, name)


## INTERPOLATED PRIORS: General tools (target: cosmology/ILE)

def norm_cdf_and_inverse(dPdx, xlim=[0,1],vectorized=True,rtol=1e-6,method='DOP853',**kwargs):
  """
  Finds forward and inverse CDF via accurate ODE solve.  Makes sure to correct for normalization.
  Danger: be careful if integrand goes to zero before the left limit!
  Targeting reasonably high tolerance (of order 1e-6)

  Weakness: pure CPU return value.  Wrap in interpolating function call to force use of returning GPU-compatible function
     Could try https://github.com/yor-dev/cupy_ivp/tree/master
  """
  nm = scipy.integrate.quad(dPdx, xlim[0], xlim[1])[0]
 # print(nm)
  def func_inv(P,x):
    return 1/dPdx(x)
  def func_forward(x,P):
    return dPdx(x)
  sol_inv = scipy.integrate.solve_ivp(func_inv,(0,nm),[xlim[0]],vectorized=vectorized, dense_output=True,rtol=rtol,method=method, **kwargs)
  if not(sol_inv.success):
    print(sol_inv)
    raise Exception("Prior: normalization failure")
#  print(sol_inv)
  sol_forward = scipy.integrate.solve_ivp(func_forward,(xlim[0],xlim[1]),[0],vectorized=vectorized, dense_output=True,atol=1e-8,rtol=rtol,method=method, **kwargs)
  if not(sol_forward.success):
    print(sol_forward)
    raise Exception("Prior: normalization failure")
#  print(sol_forward)
  return nm, lambda x,f=sol_inv.sol, c=nm: f(x*c)[0],lambda x,f=sol_forward.sol, c=nm: f(x)[0]/c

def norm_and_inverse_via_grid_interp(dPdx, xlim=[1e-5,1], y_of_x=None, loglog=False,log_grid=True,interp_method_name='PchipInterpolator' ,final_scipy_interpolate=scipy.interpolate,final_np=np,to_gpu=lambda x: x, to_gpu_needed=False,n_grid=1000, **kwargs):
   interp_action = getattr(final_scipy_interpolate, interp_method_name)
   nm, sol_inv, sol_forward = norm_cdf_and_inverse(dPdx, xlim, **kwargs)
   if log_grid:
    # grid 1, for logarithmic spacing for interpolation of dp/dx, NOT UNIFORM. PROBLEM: linear interpolation sub-optimal on this dynamic range.
    x_vals = np.exp(np.linspace(np.log(xlim[0]), np.log(xlim[1]), num=n_grid))
   else:
    x_vals = np.linspace(xlim[0], xlim[1], num=n_grid)
   dp_vals = dPdx(x_vals)/nm # remove scale factor from dP/dx
   P_vals = sol_forward(x_vals)
   P_vals *= 1./P_vals[-1] # rescale to CDF unity at right limit. Normalization should be 1 from previous code
   dp_vals *= 1./P_vals[-1] # similarly, scale out normalization constant. Normalizatio should be 1 from previous code
   if y_of_x:
    x_vals = y_of_x(x_vals) # reparameterize
    if loglog:
      raise Exception(" interpolation of cdf/pdf: reparameterization dP/dy not enabled with log log")
   # convert to GPU if needed
   if to_gpu_needed:
    x_vals = to_gpu(x_vals)
    dp_vals = to_gpu(dp_vals)
    P_vals = to_gpu(P_vals)
   if not(loglog):
      # interpolation with default method
      dp_func = interp_action(x_vals, dp_vals)
      cdf_func = interp_action(x_vals, P_vals)
      cdf_inv_func = interp_action(P_vals, x_vals)
      # reparameterize action completed for PDF
      if y_of_x:
        dp_func = cdf_func.derivative()
      return dp_func, cdf_func, cdf_inv_func
   # interpolation with default method, using log-log variables for dynamic range stability
   #.  - warning: dp/dy not implemented
   #print(np.log(x_vals))
   dp_func_log = interp_action(final_np.log(x_vals), dp_vals) # dp(ln x). Incorrect if we have changed to y
   cdf_vals_func_log_log = interp_action(final_np.log(x_vals),final_np.log(P_vals)) # lnP (lnx)
   cdf_inv_func_log_log =interp_action(final_np.log(P_vals),final_np.log(x_vals)) # lnx (lnP)
   # wrap in functions
   cdf_func_wrapper = lambda x: final_np.exp(cdf_vals_func_log_log(final_np.log(x)))
   cdf_inv_func_wrapper = lambda x: final_np.exp(cdf_inv_func_log_log(final_np.log(x)))
   pdf_func_wrapper = lambda x: dp_func_log(final_np.log(x))
   # return
   return pdf_func_wrapper, cdf_func_wrapper, cdf_inv_func_wrapper
