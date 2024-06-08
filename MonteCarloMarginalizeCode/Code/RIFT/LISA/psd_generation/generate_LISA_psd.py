import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--NC", default=3, type=int, help="Number of channels.")
parser.add_argument("--Tobs", default=0.5, type=float,  help="Observation time in years.")
parser.add_argument("--fmin", default=5.0e-5, type=float, help="Lowest frequency at which PSD needs to be generated.")
parser.add_argument("--fmax", default=1, type=float, help="Highest frequency at which PSD needs to be generated.")
opts=parser.parse_args()

print(f"Argument parser has the following arguments:\n{vars(opts)}")

# CONSTANTS
fm     = 3.168753575e-8   # confirm this
YRSID_SI = 31558149.763545603 # year in seconds
C_SI = 299792458.      # speed of light (m/s)
e = 0.004824185218078991
a = 149597870700. #meters
Larm = 2*np.sqrt(3)*a*e
fstar = C_SI/(2*np.pi*Larm)

path_to_file=os.path.dirname(__file__)

# FUNCTIONS
# These function were taken from LISA.py of LISA sensitivty (https://github.com/eXtremeGravityInstitute/LISA_Sensitivity)
def Pn(f):
    """
    Caclulate the Strain Power Spectral Density
    """
   
    # single-link optical metrology noise (Hz^{-1}), Equation (10)
    P_oms = (1.5e-11)**2*(1. + (2.0e-3/f)**4)
   
    # single test mass acceleration noise, Equation (11)
    P_acc = (3.0e-15)**2*(1. + (0.4e-3/f)**2)*(1. + (f/(8.0e-3))**4)
   
    # total noise in Michelson-style LISA data channel, Equation (12)
    Pn = (P_oms + 2.*(1. + np.cos(f/fstar)**2)*P_acc/(2.*np.pi*f)**4)/Larm**2
   
    return Pn
   
def SnC(f, Tobs = 0.5, NC = 3):
    """
    Get an estimation of the galactic binary confusion noise are available for
        Tobs = {0.5 yr, 1 yr, 2 yr, 4yr}
    Enter Tobs as a year or fraction of a year
    """

    # Fix the parameters of the confusion noise fit
    if (Tobs < .75*YRSID_SI):
        est = 1
    elif (0.75*YRSID_SI < Tobs and Tobs < 1.5*YRSID_SI):
        est = 2
    elif (1.5*YRSID_SI < Tobs and Tobs < 3.0*YRSID_SI):
        est = 3
    else:
        est = 4
    
    if (est==1):
        alpha  = 0.133
        beta   = 243.
        kappa  = 482.
        gamma  = 917.
        f_knee = 2.58e-3
    elif (est==2):
        alpha  = 0.171
        beta   = 292.
        kappa  = 1020.
        gamma  = 1680.
        f_knee = 2.15e-3
    elif (est==3):
        alpha  = 0.165
        beta   = 299.
        kappa  = 611.
        gamma  = 1340.
        f_knee = 1.73e-3
    else:
        alpha  = 0.138
        beta   = -221.
        kappa  = 521.
        gamma  = 1680.
        f_knee = 1.13e-3
   
    A = 1.8e-44/NC
   
    Sc  = 1. + np.tanh(gamma*(f_knee - f))
    Sc *= np.exp(-f**alpha + beta*f*np.sin(kappa*f))
    Sc *= A*f**(-7./3.)
   
    return Sc
   
def Sn(f, Tobs = 0.5, NC = 3, R_exists=False, interp_func=None):
    """ Calculate the sensitivity curve """

    if (R_exists == True): # if sensitivity curve file is provided use it
        R = interpolate.splev(f, interp_func, der=0)
    else:
        R = 3./20./(1. + 6./10.*(f/fstar)**2)*NC

    Sn = Pn(f)/R + SnC(f,  Tobs, NC)

    return Sn


NC = opts.NC # number of channels
Tobs = opts.Tobs*YRSID_SI # years

if os.path.exists(f"{path_to_file}/R.txt"):
    data = np.loadtxt(f"{path_to_file}/R.txt")
    R = data[:,1]*NC
    f = data[:,0]*fstar
    interp_func = interpolate.splrep(f, R, s=0)
    R_exists = True
else:
    print("R.txt doesn't exist.")
    R_exists = False
    interp_func = False

# fvals over which PSD will be generated, needs to be in linear space for RIFT to read it.
f  = np.linspace(opts.fmin, opts.fmax, 500001)  # these many points are sufficient, rely on RIFT's interpolation  after. RIFT doesn't like if deltaF is larger than f0.

# generate PSD
sens = Sn(f, Tobs, NC, R_exists, interp_func)

# reshape to save
f = f.reshape(-1,1)
sens = sens.reshape(-1,1)
sens_save=np.hstack([f,sens])

# Plot
plt.xlabel("Frequency [Hz]")
plt.ylabel("Characteristic strain")
plt.loglog(f, np.sqrt(f*sens))
plt.savefig(os.getcwd()+"/LISA_psd_plot.png",  bbox_inches="tight")

# Save as txt
np.savetxt(os.getcwd() +"/LISA_psd.txt", sens_save)

# Save as xml
os.system(f"convert_psd_ascii2xml --fname-psd-ascii {os.getcwd()}/LISA_psd.txt --conventional-postfix --ifo A")
