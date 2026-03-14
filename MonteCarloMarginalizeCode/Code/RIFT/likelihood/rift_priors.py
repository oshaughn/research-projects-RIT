import RIFT.lalsimutils as lalsimutils
import numpy as np
import functools

from RIFT.integrators.mcsampler import uniform_samp_cos_theta, uniform_samp_theta, uniform_samp_phase

# problem: weird names, changing them is annoying!
mc_max = 100
mc_min = 1
chi_max = 1
chi_min =0
chi_small_max=1
chi_small_min=0
eta_min =0.1
lambda_max = 5000
lambda_min=0.1
ECC_MAX=0.5
ECC_MIN=0
MEANPERANO_MAX=2*np.pi
MEANPERANO_MIN=0

# mcmin, mcmax : to be defined later
def M_prior(x):  # not normalized; see section II.C of https://arxiv.org/pdf/1701.01137.pdf
    return 2*x/(mc_max**2-mc_min**2)
def q_prior(x,norm_factor=1.):
    return norm_factor/(1+x)**2  # not normalized; see section II.C of https://arxiv.org/pdf/1701.01137.pdf
def m1_prior(x):
    return 1./200
def m2_prior(x):
    return 1./200
def s1z_prior(x):
    return 1./(2*chi_max)
def s2z_prior(x):
    return 1./(2*chi_max)
def mc_prior(x):
    return 2*x/(mc_max**2-mc_min**2)
def unscaled_eta_prior_cdf(eta_min):
    r"""
    cumulative for integration of x^(-6/5)(1-4x)^(-1/2) from eta_min to 1/4.
    Used to normalize the eta prior
    Derivation in mathematica:
       Integrate[ 1/\[Eta]^(6/5) 1/Sqrt[1 - 4 \[Eta]], {\[Eta], \[Eta]min, 1/4}]
    """
    return  2**(2./5.) *np.sqrt(np.pi)*scipy.special.gamma(-0.2)/scipy.special.gamma(0.3) + 5*scipy.special.hyp2f1(-0.2,0.5,0.8, 4*eta_min)/(eta_min**(0.2))
def eta_prior(x,norm_factor=1.44):
    """
    eta_prior returns the eta prior. 
    Change norm_factor by the output 
    """
    return 1./np.power(x,6./5.)/np.power(1-4.*x, 0.5)/norm_factor
def delta_mc_prior(x,norm_factor=1.44):
    """
    delta_mc = sqrt(1-4eta)  <-> eta = 1/4(1-delta^2)
    Transform the prior above
    """
    eta_here = 0.25*(1 -x*x)
    return 2./np.power(eta_here, 6./5.)/norm_factor

def m_prior(x):
    return 1/(1e3-1.)  # uniform in mass, use a square.  Should always be used as m1,m2 in pairs. Note this does NOT restrict m1>m2.


def triangle_prior(x,R=chi_max):
    return (np.ones(x.shape)-np.abs(x/R))/R  # triangle from -R to R centered on zero
def xi_uniform_prior(x):
    return np.ones(x.shape)
def s_component_uniform_prior(x,R=chi_max):  # If all three are used, a volumetric prior
    return np.ones(x.shape)/(2.*R)
def s_magnitude_uniform_prior(x,R=chi_max):
    return np.ones(x.shape)/R
def s_component_sqrt_prior(x,R=chi_max):  # If all three are used, a volumetric prior
    return 1./(4.*R*np.sqrt(np.abs(x)/R))  # -R,R range
def s_component_gaussian_prior(x,R=chi_max/3.):
    """
    (proportinal to) prior on range in one-dimensional components, in a cartesian domain.
    Could be useful to sample densely near zero spin.
    [Note: we should use 'truncnorm' instead...]
    """
    xp = np.array(x,dtype=float)
    val= scipy.stats.truncnorm(-chi_max/R,chi_max/R,scale=R).pdf(xp)  # stupid casting problem : x is dtype 'object'
    return val

def s_component_zprior(x,R=chi_max):
    # assume maximum spin =1. Should get from appropriate prior range
    # Integrate[-1/2 Log[Abs[x]], {x, -1, 1}] == 1
    val = -1./(2*R) * np.log( (np.abs(x)/R+1e-7).astype(float))
    return val
def s_component_zprior_positive(x,R=chi_max):
    # assume maximum spin =1. Should get from appropriate prior range
    # Integrate[-1/2 Log[Abs[x]], {x, -1, 1}] == 1
    val = -1./(2*R) * np.log( (np.abs(x)/R+1e-7).astype(float))
    return val*2


def s_component_volumetricprior(x,R=1.):
    # assume maximum spin =1. Should get from appropriate prior range
    # for SPIN MAGNITUDE OF PRECESSING SPINS only
    return (1./3.* np.power(x/R,2))

def s_component_aligned_volumetricprior(x,R=1.):
    # assume maximum spin =1. Should get from appropriate prior range
    # for SPIN COMPONENT ALIGNED (s1z,s2z) for aligned spins only
    #This is a probability that is defined on x\in[-R,R], such that \int_a^b dx p(x)  is the volume of a sphere between horizontal slices at height a,b:
    #p(x)dx =  pi R^2 (1- (x/R)^2)/ (4 pi R^3 /3) = 3/4 * (1 - (x/R)^2) /R
    return (3./4.*(1- np.power(x/R,2))/R)

def lambda_prior(x):
    return np.ones(x.shape)/(lambda_max-lambda_min)   # assume arbitrary
def lambda_small_prior(x):
    return np.ones(x.shape)/(lambda_small_max -lambda_min)   # assume arbitrary


# DO NOT USE UNLESS REQUIRED FOR COMPATIBILITY
def lambda_tilde_prior(x):
    return np.ones(x.shape)/lambda_max   # 0,4000
def delta_lambda_tilde_prior(x):
    return np.ones(x.shape)/1000.   # -500,500

def gaussian_mass_prior(x,mu=0.,sigma=1.):   # actually viable for *any* prior.  
    y = np.array(x,dtype=np.float32)
    return np.exp( - 0.5*(y-mu)**2/sigma**2)/np.sqrt(2*np.pi*sigma**2)

def tapered_magnitude_prior(x,loc=0.65,kappa=19.):   # 
    """ 
    tapered_magnitude_prior is 1 inside a region and tapers to 0 outside
    The scale factor is designed so the taper is very strong and has no effect away from the region of significance
    Equivalent to
        (1 - 1/(1+f1)) / (1+f2) = f1/(1+f1)(1+f2)
    """
    y = np.array(x,dtype=np.float32) # problem of object type data
    f1 = np.exp( - (y-loc)*kappa)
    f2 = np.exp( - (y+loc)*kappa)
    
    return f1/(1+f1)/(1+f2)

def tapered_magnitude_prior_alt(x,loc=0.8,kappa=20.):   # 
    """ 
    tapered_magnitude_prior is 1 above the scale factor and 0 below it
        1/ (1+f) =
    """
    y = np.array(x,dtype=np.float32) # problem of object type data
    f1 = np.exp( - (y-loc)*kappa)
    
    return 1/(1+f1)

def eccentricity_prior(x):
    return np.ones(x.shape) / (ECC_MAX-ECC_MIN) # uniform over the interval [0.0, ECC_MAX]

def eccentricity_squared_prior(x):  # note this is INCONSISTENT with the prior above -- we are designed to give a CDF = (e/emax)^2 for example here, or more generally (e^2 - emin^2)/(emax^2-emin^2)
    return np.ones(x.shape) / (ECC_MAX**2-ECC_MIN**2) # uniform over the interval [ECC_MIN, ECC_MAX]

def meanPerAno_prior(x):
    return np.ones(x.shape) / (MEANPERANO_MAX-MEANPERANO_MIN) # uniform over the interval [MEANPERANO_MIN, MEANPERANO_MAX]

def precession_prior(x):
    return 0.5*np.ones(x.shape) # uniform over the interval [0.0, 2.0]

def unnormalized_uniform_prior(x):
    return np.ones(x.shape)
def unnormalized_log_prior(x):
    return 1./x

def normalized_Rbar_prior(x):
    return 2*x
p_Rbar = lalsimutils.p_R
def normalized_Rbar_singular_prior(x):
    return np.power(x, p_Rbar-1.)*p_Rbar
def normalized_zbar_prior(z):
    return 3.*(1.-z**2)/4.

prior_map  = { "mtot": M_prior, "q":q_prior, "s1z":s_component_uniform_prior, "s2z":functools.partial(s_component_uniform_prior, R=chi_small_max), "mc":mc_prior, "eta":eta_prior, 'delta_mc':delta_mc_prior, 'xi':xi_uniform_prior,'chi_eff':xi_uniform_prior,'delta': (lambda x: 1./2),
    's1x':s_component_uniform_prior,
    's2x':functools.partial(s_component_uniform_prior, R=chi_small_max),
    's1y':s_component_uniform_prior,
    's2y': functools.partial(s_component_uniform_prior, R=chi_small_max),
    'chiz_plus':s_component_uniform_prior,
    'chiz_minus':s_component_uniform_prior,
    'm1':m_prior,
    'm2':m_prior,
    'lambda1':lambda_prior,
    'lambda2':lambda_small_prior,
    'lambda_plus': lambda_prior,
    'lambda_minus': lambda_prior,
    'LambdaTilde':lambda_tilde_prior,
    'DeltaLambdaTilde':delta_lambda_tilde_prior,
    # Polar spin components (uniform magnitude by default)
    'chi1':s_magnitude_uniform_prior,  
    'chi2':functools.partial(s_magnitude_uniform_prior, R=chi_small_max),
    'theta1': uniform_samp_theta,
    'theta2': uniform_samp_theta,
    'cos_theta1': uniform_samp_cos_theta,
    'cos_theta2': uniform_samp_cos_theta,
    'phi1':uniform_samp_phase,
    'phi2':uniform_samp_phase,
    # Pseudo-cylindrical : note this is a VOLUMETRIC prior
    'chi1_perp_bar':normalized_Rbar_prior,
    'chi1_perp_u':unnormalized_uniform_prior,
    'chi2_perp_bar':normalized_Rbar_prior,
    'chi2_perp_u':unnormalized_uniform_prior,
    's1z_bar':normalized_zbar_prior,
    's2z_bar':normalized_zbar_prior,
    # Other priors
    'eccentricity':eccentricity_prior,
    'eccentricity_squared':eccentricity_squared_prior,
    'meanPerAno':meanPerAno_prior,
    'chi_pavg':precession_prior,
    'mu1': unnormalized_log_prior,
    'mu2': unnormalized_uniform_prior
}
