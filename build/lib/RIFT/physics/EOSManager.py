#
#  EOSManager.py 
#
# SEE ALSO
#   - util_WriteXMLWithEOS
#   - gwemlightcurves.KNTable

# SERIOUS LIMITATIONS
#   - EOSFromFile  : File i/o for each EOS creation will slow things donw.  This command is VERY trivial, so we should be able
#          to directly create the structure ourselves, using eos_alloc_tabular
#           https://github.com/lscsoft/lalsuite/blob/master/lalsimulation/src/LALSimNeutronStarEOSTabular.c

rosDebug=False

import numpy as np
import os
import sys
import lal
import lalsimulation as lalsim
from scipy.integrate import quad
import scipy.interpolate as interp
import scipy

#import gwemlightcurves.table as gw_eos_table

from . import MonotonicSpline as ms


C_CGS=2.997925*10**10 # Argh, Monica!
DENSITY_CGS_IN_MSQUARED=7.42591549e-25  # g/cm^3 m^2 //GRUnits. Multiply by this to convert from CGS -> 1/m^2 units (_geom)

###
### SERVICE 0: General EOS structure
###

class EOSConcrete:
    """
    Class characterizing a specific EOS solution.  This structure *SHOULD* 
        - auto-build the mass-radius via a TOV solve
         - provides ability to query the lambda(m) relationship and (in the future) higher-order multipole moments; etc
    As many of these features are already provided by lalsimulation, 
    """

    def __init__(self,name=None):
        self.name=name
        self.eos = None
        self.eos_fam = None
        return None

    def lambda_from_m(self, m):
        eos_fam = self.eos_fam
        if m<10**15:
            m=m*lal.MSUN_SI

        k2=lalsim.SimNeutronStarLoveNumberK2(m, eos_fam)
        r=lalsim.SimNeutronStarRadius(m, eos_fam)

        m=m*lal.G_SI/lal.C_SI**2
        lam=2./(3*lal.G_SI)*k2*r**5
        dimensionless_lam=lal.G_SI*lam*(1/m)**5

        return dimensionless_lam


    def pressure_density_on_grid_alternate(self,logrho_grid,enforce_causal=False):
        """ 
        pressure_density_on_grid.
        Input and output grid units are in SI (rho: kg/m^3; p = N/m^2)
        Pressure provided by lalsuite (=EOM integration)
        Density computed by m*n = (epsilon+p)/c^2mn exp(-h), which does NOT rely on lalsuite implementation 
        """
        dat_out = np.zeros(len(logrho_grid))
        fam = self.eos_fam
        eos = self.eos
        npts_internal = 10000
        p_internal = np.zeros(npts_internal)
        rho_internal = np.zeros(npts_internal)
        epsilon_internal = np.zeros(npts_internal)
        hmax = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos)
        if enforce_causal:
            # strip out everything except the causal part.
            hmax = lalsim.SimNeutronStarEOSMinAcausalPseudoEnthalpy(eos)
        h = np.linspace(0.0001,hmax,npts_internal)
        for indx in np.arange(npts_internal):
            p_internal[indx] = lalsim.SimNeutronStarEOSPressureOfPseudoEnthalpy(h[indx],eos)  # SI. Multiply by 10 to get CGS
            epsilon_internal[indx] =lalsim.SimNeutronStarEOSEnergyDensityOfPseudoEnthalpy(h[indx],eos)  # SI. Note factor of C^2 needed to get mass density
            rho_internal[indx] =np.exp(-h[indx])* (epsilon_internal[indx]+p_internal[indx])/(lal.C_SI**2)  # 
#        print epsilon_internal[10],rho_internal[10], p_internal[10], h[10]
        logp_of_logrho = interp.interp1d(np.log10(rho_internal),np.log10(p_internal),kind='linear',bounds_error=False,fill_value=np.inf)  # should change to Monica's spline
 #       print logrho_grid,
        return logp_of_logrho(logrho_grid)

    def pressure_density_on_grid(self,logrho_grid,reference_pair=None,enforce_causal=False):
        """ 
        pressure_density_on_grid.
        Input and output grid units are in SI (rho: kg/m^3; p = N/m^2)
        POTENTIAL PROBLEMS OF USING LALSUITE
            - lalinference_o2 / master: Unless patched, the *rest mass* density is not reliable.  
              To test with the unpatched LI version, use reference_pair to specify a low-density EOS.
              This matching is highly suboptimal, so preferably test either (a) a patched code or (b) the alternative code below
        """
        dat_out = np.zeros(len(logrho_grid))
        fam = self.eos_fam
        eos = self.eos
        npts_internal = 10000
        p_internal = np.zeros(npts_internal)
        rho_internal = np.zeros(npts_internal)
        hmax = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos)
        if enforce_causal:
            # strip out everything except the causal part.
            hmax = lalsim.SimNeutronStarEOSMinAcausalPseudoEnthalpy(eos)
        h = np.linspace(0.0001,hmax,npts_internal)
        for indx in np.arange(npts_internal):
            rho_internal[indx] = lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(h[indx],eos)  # SI. Multiply by 10^(-3) to get CGS
            p_internal[indx] = lalsim.SimNeutronStarEOSPressureOfPseudoEnthalpy(h[indx],eos)  # SI. Multiply by 10 to get CGS
        if not (reference_pair is None):
            indx_match = np.argmin( np.abs(np.log10(p_internal) - np.log10(reference_pair[1]))) # force agreement of densities at target pressure, if requested! Addresses bug /ambiguity in scaling of rest mass estimate; intend to apply in highly nonrelativistic regime
            delta_rho = np.log10(reference_pair[0]) -np.log10(rho_internal[indx_match]) 
            rho_internal *= np.power(10, delta_rho)
#            print  np.log10(np.c_[rho_internal,p_internal])
        logp_of_logrho = interp.interp1d(np.log10(rho_internal),np.log10(p_internal),kind='linear',bounds_error=False,fill_value=np.inf)  # should change to Monica's spline
 #       print logrho_grid,
        return logp_of_logrho(logrho_grid)

    def test_speed_of_sound_causal(self, test_only_under_mmax=True,fast_test=True):
        """
        Test if EOS satisfies speed of sound.
        Relies on low-level lalsimulation interpolation routines to get v(h) and as such is not very reliable

        By DEFAULT, we are testing the part of the EOS that is
             - at the largest pressure (assuming monotonic sound speed)
             - associated with the maximum mass NS that is stable
        We can also test the full table that is provided to us.
        https://git.ligo.org/lscsoft/lalsuite/blob/lalinference_o2/lalinference/src/LALInference.c#L2513
        """
        npts_internal = 1000
        eos = self.eos
        fam = self.eos_fam
        # Largest NS provides largest attained central pressure
        m_max_SI = self.mMaxMsun*lal.MSUN_SI
        if not test_only_under_mmax:
            hmax = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos)
        else:
            try:
                pmax = lalsim.SimNeutronStarCentralPressure(m_max_SI,fam)  
                hmax = lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(pmax,eos)
            except:
                # gatch gsl interpolation errors for example
                return False  
        if fast_test: 
            # https://git.ligo.org/lscsoft/lalsuite/blob/lalinference_o2/lalinference/src/LALInference.c#L2513
            try:
                vsmax = lalsim.SimNeutronStarEOSSpeedOfSoundGeometerized(hmax, eos)
                return vsmax <1.1
            except:
                # catch gsl interpolation errors for example
                return False
        else:
            if rosDebug:
                print(" performing comprehensive test ")
        h = np.linspace(0.0001,hmax,npts_internal)
#        h = np.linspace(0.0001,lalsim.SimNeutronStarEOSMinAcausalPseudoEnthalpy(eos),npts_internal)
        vs_internal = np.zeros(npts_internal)
        for indx in np.arange(npts_internal):
            vs_internal[indx] =  lalsim.SimNeutronStarEOSSpeedOfSoundGeometerized(h[indx],eos)
            if rosDebug:
                print(h[indx], vs_internal[indx])
        return not np.any(vs_internal>1.1)   # allow buffer, so we have some threshold

###
### SERVICE 1: lalsimutils structure
###
#  See https://github.com/lscsoft/lalsuite/tree/master/lalsimulation/src for available types
class EOSLALSimulation(EOSConcrete):
    def __init__(self,name):
        self.name=name
        self.eos = None
        self.eos_fam = None
        self.mMaxMsun=None


        eos = lalsim.SimNeutronStarEOSByName(name)
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        mmass = lalsim.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
        self.eos = eos
        self.eos_fam = fam
        self.mMaxMsun = mmass
        return None





###
### SERVICE 2: EOSFromFile
###

# Example directory: EOS_Tables
#dirEOSTablesBase = os.environ["EOS_TABLES"]
dirLALSimulationBase = os.environ["LALSIMULATION_DATADIR"]  # LAL table data
## Add routines to find, parse standard directory of EOS files and load suitable metadata into memory
## Follow framework of NRWaveformCatalogManager3


class EOSFromDataFile(EOSConcrete):
    """ 
    FromDataFileEquationOfState
    (just accepts filename...not attempting to parse a catalog)
    
    """
    def __init__(self,name=None,fname=None):
        self.name=name
        self.fname=fname
        self.eos = None
        self.eos_fam = None
        self.mMax = None

        self.eos, self.eos_fam = self.eos_ls()
        return None

    def eos_ls(self):
        # From Monica, but using code from GWEMLightcurves
        #  https://gwemlightcurves.github.io/_modules/gwemlightcurves/KNModels/table.html
        """
        EOS tables described by Ozel `here <https://arxiv.org/pdf/1603.02698.pdf>`_ and downloadable `here <http://xtreme.as.arizona.edu/NeutronStars/data/eos_tables.tar>`_. LALSim utilizes this tables, but needs some interfacing (i.e. conversion to SI units, and conversion from non monotonic to monotonic pressure density tables)
    """
        obs_max_mass = 2.01 - 0.04  # used
        print("Checking %s" % self.name)
        eos_fname = ""
        if os.path.exists(self.fname):
            # NOTE: Adapted from code by Monica Rizzo
            print("Loading from %s" % self.fname)
            bdens, press, edens = np.loadtxt(self.fname, unpack=True)
            press *= DENSITY_CGS_IN_MSQUARED
            edens *= DENSITY_CGS_IN_MSQUARED
            eos_name = self.name

            if not np.all(np.diff(press) > 0):
                keep_idx = np.where(np.diff(press) > 0)[0] + 1
                keep_idx = np.concatenate(([0], keep_idx))
                press = press[keep_idx]
                edens = edens[keep_idx]
            assert np.all(np.diff(press) > 0)
            if not np.all(np.diff(edens) > 0):
                keep_idx = np.where(np.diff(edens) > 0)[0] + 1
                keep_idx = np.concatenate(([0], keep_idx))
                press = press[keep_idx]
                edens = edens[keep_idx]
            assert np.all(np.diff(edens) > 0)

            # Creating temporary file in suitable units
            print("Dumping to %s" % self.fname)
            eos_fname = "./" +eos_name + "_geom.dat" # assume write acces
            np.savetxt(eos_fname, np.transpose((press, edens)), delimiter='\t')
            eos = lalsim.SimNeutronStarEOSFromFile(eos_fname)
            fam = lalsim.CreateSimNeutronStarFamily(eos)

        else:
            print(" No such file ", self.fname)
            sys.exit(0)

        mmass = lalsim.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
        self.mMaxMsun = mmass
        return eos, fam

    def p_rho_arrays(self):
        print(self.fname)
        dat_file = np.array(np.loadtxt(self.fname))
        nb=dat_file[:,0]
        p=dat_file[:,1]
        rho=dat_file[:,2] 
        return nb,p,rho
    def interp_eos_p_of_rho(self):
        nb,p,rho=self.p_rho_arrays()
        n=len(p)
        p=np.log10(p)
        rho=np.log10(rho)
        consts=ms.interpolate(rho,p)
        line_const=ms.lin_extrapolate(rho,p)
        #linearly interpolate anything outside range
        line_lower=line_const[0,:]
        line_upper=line_const[1,:]
        return consts,line_upper,line_lower

   #interpolates Log10 of data
    def interp_eos_rho_of_p(self):
        nb,p,rho=self.p_rho_arrays()
        n=len(p)
        p=np.log10(p)
        rho=np.log10(rho)
        consts=ms.interpolate(p,rho)  
        line_const=ms.lin_extrapolate(p,rho)
       #linearly interpolate anything outside range
        line_lower=line_const[0,:]
        line_upper=line_const[1,:]
        return consts,line_upper,line_lower
    def interp_eos_nb_of_p(model_name):
        nb,p,rho=self.p_rho_arrays()
        n=len(p)
        p=np.log10(p)
        nb=np.log10(nb)
        consts=ms.interpolate(p,nb)
        line_const=ms.lin_extrapolate(p,nb)
        #linearly interpolate anything outside range
        line_lower=line_const[0,:]
        line_upper=line_const[1,:]
        return consts,line_upper,line_lower





###
### SERVICE 2: Parameterized EOS (specify functions)
###

# COMMON POLYTROPE TABLE
# eos logP1 gamma1 gamma2 gamma3
# PAL6 34.380 2.227 2.189 2.159 
# SLy 34.384 3.005 2.988 2.851 
# AP1 33.943 2.442 3.256 2.908 
# AP2 34.126 2.643 3.014 2.945
# AP3 34.392 3.166 3.573 3.281  
# AP4 34.269 2.830 3.445 3.348 
# FPS 34.283 2.985 2.863 2.600 
# WFF1 34.031 2.519 3.791 3.660 
# WFF2 34.233 2.888 3.475 3.517  
# WFF3 34.283 3.329 2.952 2.589  
# BBB2 34.331 3.418 2.835 2.832 
# BPAL12 34.358 2.209 2.201 2.176 
# ENG 34.437 3.514 3.130 3.168 
# MPA1 34.495 3.446 3.572 2.887 
# MS1 34.858 3.224 3.033 1.325 
# MS2 34.605 2.447 2.184 1.855 
# MS1b 34.855 3.456 3.011 1.425 
# PS 34.671 2.216 1.640 2.365 
# GS1 34.504 2.350 1.267 2.421 
# GS2 34.642 2.519 1.571 2.314 
# BGN1H1 34.623 3.258 1.472 2.464 
# GNH3 34.648 2.664 2.194 2.304 
# H1 34.564 2.595 1.845 1.897
# H2 34.617 2.775 1.855 1.858
# H3 34.646 2.787 1.951 1.901
# H4 34.669 2.909 2.246 2.144
# H5 34.609 2.793 1.974 1.915
# H6 34.593 2.637 2.121 2.064
# H7 34.559 2.621 2.048 2.006
# PCL2 34.507 2.554 1.880 1.977 
# ALF1 34.055 2.013 3.389 2.033 
# ALF2 34.616 4.070 2.411 1.890 
# ALF3 34.283 2.883 2.653 1.952 
# ALF4 34.314 3.009 3.438 1.803

# Rizzo code: EOS_param.py
class EOSPiecewisePolytrope(EOSConcrete):
    def __init__(self,name,param_dict=None):
        self.name=name
        self.eos = None
        self.eos_fam = None
        self.mMaxMsun=None


        eos=self.eos=lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(param_dict['logP1'], param_dict['gamma1'], param_dict['gamma2'], param_dict['gamma3'])
        eos_fam=self.eos_fam=lalsim.CreateSimNeutronStarFamily(eos)
        self.mMaxMsun = lalsim.SimNeutronStarMaximumMass(eos_fam) / lal.MSUN_SI

        return None


class EOSLindblomSpectral(EOSConcrete):
    def __init__(self,name=None,spec_params=None,verbose=False,use_lal_spec_eos=False):
        if name is None:
            self.name = 'spectral'
        else:
            self.name=name
        self.eos = None
        self.eos_fam = None

        self.spec_params = spec_params
#        print spec_params

        if use_lal_spec_eos:
#            self.eos=lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4'])   # Should have this function! but only on master
            self.eos=lalsim.SimNeutronStarEOSSpectralDecomposition_for_plot(spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4'],4)
        else:
            # Create data file
            self.make_spec_param_eos(500,save_dat=True,ligo_units=True,verbose=verbose)
            # Use data file
            #print " Trying to load ",name+"_geom.dat"
            import os; #print os.listdir('.')
            cwd = os.getcwd()
            self.eos=eos = lalsim.SimNeutronStarEOSFromFile(cwd+"/"+name+"_geom.dat")
        self.eos_fam = fam=lalsim.CreateSimNeutronStarFamily(self.eos)
        mmass = lalsim.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
        self.mMaxMsun = mmass

        return None

    def test_bounded_adiabatic_index(self,bounds=[0.6,4.5]):
        """
        Gamma(p) \in bounds
        Uses xmax and other parameters from spectral result
        """
        spec_params =self.spec_params
        if not 'gamma3' in spec_params:
            spec_params['gamma3']=spec_params['gamma4']=0
        coefficients=np.array([spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4']])
        xmax = self.spec_params['xmax']
        xvals = np.linspace(0,xmax,500)
        gamma_vals = gamma_of_x(xvals, coefficients)
        if rosDebug:
            print("  Spectral EOS debug test limits: Gamma bounds", np.min(gamma_vals), np.max(gamma_vals))
        return  not( np.any(gamma_vals < bounds[0]) or np.any(gamma_vals>bounds[1]) )
            

    def make_spec_param_eos(self, npts=500, plot=False, verbose=False, save_dat=False,ligo_units=False,interpolate=False,eosname_lalsuite="SLY4"):
        """
        Load values from table of spectral parameterization values
        Table values taken from https://arxiv.org/pdf/1009.0738.pdf
        Comments:
            - eos_vals is recorded as *pressure,density* pairs, because the spectral representation is for energy density vs pressure
            - units swap between geometric and CGS
            - eosname_lalsuite is used for the low-density EOS
        """

        spec_params = self.spec_params
        if not 'gamma3' in spec_params:
            spec_params['gamma3']=spec_params['gamma4']=0
        coefficients=np.array([spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4']])
        p0=spec_params['p0']
        eps0=spec_params['epsilon0']
        xmax=spec_params['xmax'] 

        x_range=np.linspace(0,xmax,npts)
        p_range=p0*np.exp(x_range)
       
        eos_vals=np.zeros((npts,2))
        eos_vals[:,1]=p_range

        eos_vals[:,0] = epsilon(x_range,p0,eps0, coefficients)
        # for i in range(0, len(x_range)):
        #    eos_vals[i,0]=epsilon(x_range[i], p0, eps0, coefficients)
        #    if verbose==True:
        #        print "x:",x_range[i],"p:",p_range[i],"p0",p0,"epsilon:",eos_vals[i,0]
  
    #doing as those before me have done and using SLY4 as low density region
        # THIS MUST BE FIXED TO USE STANDARD LALSUITE ACCESS, do not assume the file exists
#        low_density=np.loadtxt(dirEOSTablesBase+"/LALSimNeutronStarEOS_SLY4.dat")
        low_density = np.loadtxt(dirLALSimulationBase+"/LALSimNeutronStarEOS_"+ eosname_lalsuite+".dat")
        low_density[:,0]=low_density[:,0]*C_CGS**2/(DENSITY_CGS_IN_MSQUARED)   # converts to energy density in CGS
        low_density[:,1]=low_density[:,1]*C_CGS**2/(DENSITY_CGS_IN_MSQUARED)   # converts to energy density in CGS
        low_density[:,[0, 1]] = low_density[:,[1, 0]]  # reverse order

        cutoff=eos_vals[0,:]   
        if verbose:
            print(" cutoff ", cutoff)
 
        break_pt=0
        for i in range(0, len(low_density)):
            if low_density[i,0] > cutoff[0] or low_density[i,1] > cutoff[1]:   
                break_pt=i
                break 
    
        eos_vals=np.vstack((low_density[0:break_pt,:], eos_vals)) 

        if not interpolate:
#            print eos_vals
            if ligo_units:
                eos_vals *= DENSITY_CGS_IN_MSQUARED/(C_CGS**2)  # converts to geometric units: first convert from cgs energy density to g/cm^2, then to 1/m^2.
 #               print " Rescaled "
#                print eos_vals
            
            if save_dat == True:
                np.savetxt(self.name+"_geom.dat", eos_vals[:,[1,0]])  #NOTE ORDER

            return eos_vals
        
        # Optional: interpolate in the log, to generate a denser EOS model
        # Will produce better M(R) models for LAL
        p_of_epsilon = ms.interpolate(np.log10(eos_vals[1:,0]), np.log10(eos_vals[1:,1]))
  
        new_eos_vals = np.zeros((resample_pts, 2))
        epsilon_range = np.linspace(min(np.log10(eos_vals[1:,0])), max(np.log10(eos_vals[1:,0])), resample_pts)
        new_eos_vals[:, 0] = 10**epsilon_range 
 
        for i in range(0, resample_pts):
            if verbose == True:
                print("epsilon", 10**epsilon_range[i])

            new_eos_vals[i,1] = 10**ms.interp_func(epsilon_range[i], np.log10(eos_vals[1:,0]), np.log10(eos_vals[1:,1]), p_of_epsilon)

            if verbose == True:
                print("p", new_eos_vals[i,1])
    
        new_eos_vals = check_monotonicity(new_eos_vals)
        new_eos_vals = np.vstack((np.array([0.,0.]), new_eos_vals))
        return new_eos_vals



def gamma_of_x(x, coeffs):
        """
        Eq 6 from https://arxiv.org/pdf/1009.0738.pdf
        """
        gamma=0
        # Equivalent to np.polyval(coeffs[::-1],x)
        gamma=np.polyval(coeffs[::-1],x)
        # for i in range(0,len(coeffs)):
        #     gamma+=coeffs[i]*x**i 
        gamma=np.exp(gamma)  
        return gamma
  
def mu(x, coeffs):
        """
        Eq 8 from https://arxiv.org/pdf/1009.0738.pdf
        """


        # very inefficient: does integration multiple times. Should change to ODE solve
        if isinstance(x, (list, np.ndarray)):
            def int_func(dummy,x_prime):
              return (gamma_of_x(x_prime, coeffs))**(-1)    
            y = scipy.integrate.odeint(int_func,[0],x,full_output=False).T  # x=0 needs to be value in array
            return np.exp(-1.*y)
#            val=np.zeros(len(x))
#            for i in range(0,len(x)):
#                tmp=quad(int_func, 0, x[i])
#                val[i]=tmp[0]  
#            return np.exp(-1.*val)
        else:    
            def int_func(x_prime):
              return (gamma_of_x(x_prime, coeffs))**(-1)    
            val=quad(int_func, 0, x)

        return np.exp(-1.*val[0])

def epsilon(x, p0, eps0, coeffs,use_ode=True):
        """
        Eq. 7 from https://arxiv.org/pdf/1009.0738.pdf
        """
        mu_of_x=mu(x, coeffs)  
        if use_ode and isinstance(x, (list,np.ndarray)):
          mu_intp = scipy.interpolate.interp1d(x,mu_of_x,bounds_error=False,fill_value=0)
          def int_func(dummy,x_prime):
            num = mu_intp(x_prime)*np.exp(x_prime)
            denom = gamma_of_x(x_prime, coeffs)
            return num / denom
          y= scipy.integrate.odeint(int_func,0,x,full_output=False).T  # x=0 needs to be value in array
          eps=(eps0*C_CGS**2)/mu_of_x + p0/mu_of_x * y
          return eps
        else:
          def int_func(x_prime):
            num = mu(x_prime, coeffs)*np.exp(x_prime)
            denom = gamma_of_x(x_prime, coeffs)
            return num / denom
          
        # very inefficient: does integration multiple times. Should change to ODE solve
        # Would require lookup interpolation of mu_of_x
        val=quad(int_func, 0, x)
        #val=romberg(int_func, 0, x, show=True)   
        eps=(eps0*C_CGS**2)/mu_of_x + p0/mu_of_x * val[0]
 
        return eps




###
### Utilities
###

# Les-like
def make_mr_lambda_lal(eos,n_bins=100):
    """
    Construct mass-radius curve from EOS
    Based on modern code resources (https://git.ligo.org/publications/gw170817/bns-eos/blob/master/scripts/eos-params.py) which access low-level structures
    """
    fam=lalsim.CreateSimNeutronStarFamily(eos)
    max_m = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
    min_m = lalsim.SimNeutronStarFamMinimumMass(fam)/lal.MSUN_SI
    mgrid = np.linspace(min_m,max_m, n_bins)
    mrL_dat = np.zeros((n_bins,3))
    mrL_dat[:,0] = mgrid
    for indx in np.arange(n_bins):
       mass_now = mgrid[indx]
       r = lalsim.SimNeutronStarRadius(mass_now*lal.MSUN_SI,fam)/1000.
       mrL_dat[indx,1] = r
       k = lalsim.SimNeutronStarLoveNumberK2(mass_now*lal.MSUN_SI,fam)
       c = mass_now * lal.MRSUN_SI / (r*1000.)
       mrL_dat[indx,2] = (2. / 3.) * k / c**5.

    return mrL_dat

# Rizzo
def make_mr_lambda(eos,use_lal=False):
   """
   construct mass-radius curve from EOS    
   DOES NOT YET WORK RELIABLY
   """
   if use_lal:
       make_mr_lambda_lal(eos)

   fam=lalsim.CreateSimNeutronStarFamily(eos)
 
   r_cut = 40   # Some EOS we consider for PE purposes will have very large radius!

   #set p_nuc max
   #   - start at a fiducial nuclear density
   #   - not sure what these termination conditions are designed to do ... generally this pushes to  20 km
   #   - generally this quantity is the least reliable
   p_nuc=3.*10**33   # consistent with examples
   fac_min=0
   r_fin=0
   while r_fin > r_cut+8 or r_fin < r_cut:
       # Generally tries to converge to density corresponding to 20km radius
      try: 
         answer=lalsim.SimNeutronStarTOVODEIntegrate((10**fac_min)*p_nuc, eos)      # r(SI), m(SI), lambda
      except:
          # If failure, backoff
         fac_min=-0.05
         break 
      r_fin=answer[0]
      r_fin=r_fin*10**-3  # convert to SI
#      print "R: ",r_fin
      if r_fin<r_cut:
         fac_min-=0.05
      elif r_fin>r_cut+8:
         fac_min+=0.01
   answer=lalsim.SimNeutronStarTOVODEIntegrate((10**fac_min)*p_nuc, eos)      # r(SI), m(SI), lambda
   m_min = answer[1]/lal.MSUN_SI

   #set p_nuc min
   #   - tries to converge to central pressure corresponding to maximum NS mass
   #   - very frustrating...this data is embedded in the C code
   fac_max=1.6
   r_fin=20.
   m_ref = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
   r_ref = lalsim.SimNeutronStarRadius(lalsim.SimNeutronStarMaximumMass(fam), fam)/(10**3)
   answer=None
   while r_fin > r_ref  or r_fin < 7:
       #print "Trying min:"
#       print "p_c: ",(10**fac_max)*p_nuc
       try:
          answer=lalsim.SimNeutronStarTOVODEIntegrate((10**fac_max)*p_nuc, eos)         
          if answer[0]*10**-3 < r_ref:
             break 
       except:
          fac_max-=0.05
          working=False
          while working==False:
             try:
                answer_tmp=lalsim.SimNeutronStarTOVODEIntegrate((10**fac_max)*p_nuc, eos)
                working=True
             except:
                fac_max-=0.05
          break
          #print lalsim.SimNeutronStarTOVODEIntegrate((10**fac_max)*p_nuc, eos)
       r_fin=answer[0]/10**3 # convert to km
       if rosDebug:
           print("R: ",r_fin, r_ref, " M: ", answer[1]/lal.MSUN_SI, m_ref , m_min) # should converge to maximum mass
       if r_fin>8:
          fac_max+=0.05
       if r_fin<6:
          fac_max-=0.01
#       print 10**fac_max

   #generate mass-radius curve
   npts_out = 1000
   scale=np.logspace(fac_min,fac_max,npts_out)
   
   mr_array=np.zeros((npts_out,3))
   for s,i in zip(scale,range(0,len(scale))):
#       print s
       mr_array[i,:]=lalsim.SimNeutronStarTOVODEIntegrate(s*p_nuc, eos)
      
   mr_array[:,0]=mr_array[:,0]/10**3 
   mr_array[:,1]=mr_array[:,1]/lal.MSUN_SI
   mr_array[:,2]=2./(3*lal.G_SI)*mr_array[:,2]*(mr_array[:,0]*10**3)**5
   mr_array[:,2]=lal.G_SI*mr_array[:,2]*(1/(mr_array[:,1]*lal.MSUN_SI*lal.G_SI/lal.C_SI**2))**5

#   print mr_array[:,1]

   return mr_array



def LookupCrustEpsilonAtPressure(p_ref,eosname_lalsuite="SLY4"):
    """
    Tool for spectral EOS manager to find epsilon(p) via lookup tables from the lalsuite data files.
    Units are *CGS*
    Uses linear interpolation in the log. 

    Warning: lalsuite files use lal units (epsilon, p in 1/m^2), but we will presume p and epsilon are in energy density

    """
    lal_dat =np.loadtxt(dirLALSimulationBase+"/LALSimNeutronStarEOS_"+ eosname_lalsuite+".dat")
    lal_dat[:,0]=lal_dat[:,0]*C_CGS**2/(DENSITY_CGS_IN_MSQUARED)   # converts to energy density in CGS
    lal_dat[:,1]=lal_dat[:,1]*C_CGS**2/(DENSITY_CGS_IN_MSQUARED)   # converts to energy density in CGS
#    lal_dat[:,[0, 1]] = lal_dat[:,[1, 0]]  # reverse order
    
    # Interpolate in log
    lal_dat_log = np.log10(lal_dat)   # note first sample is zero,and causes problems nominally with this interpolation
    eps_out = np.power(10.,np.interp(np.log10(p_ref),  lal_dat_log[:,0], lal_dat_log[:,1]))
    return eps_out
