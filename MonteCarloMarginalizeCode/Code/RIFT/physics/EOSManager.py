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

try:
    from natsort import natsorted
except:
    print(" - no natsorted - ")

has_reprimand=False
try: 
    import pyreprimand as pyr
    has_reprimand=True
except:
    has_reprimand=False

#import gwemlightcurves.table as gw_eos_table

try:    from .            import MonotonicSpline as ms
except: from RIFT.physics import MonotonicSpline as ms

C_CGS=lal.C_SI*100
DENSITY_CGS_IN_MSQUARED=1000*lal.G_SI/lal.C_SI**2  # g/cm^3 -> 1/m^2 //GRUnits. Multiply by this to convert from CGS -> 1/m^2 units (_geom). lal.G_SI/lal.C_SI**2 takes kg/m^3 -> 1/m^2  ||  https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_neutron_star_8h_source.html
PRESSURE_CGS_IN_MSQUARED = DENSITY_CGS_IN_MSQUARED/(lal.C_SI*100)**2


def make_compactness_from_lambda_approximate(lambda_vals):
    """
    make_compactness_from_lambda_approximate
    Eq (B1) from https://arxiv.org/pdf/1812.04803.pdf, based on Maselli et al 2013, Yagi and Yunes 2017

    Note this will yield *extreme* compactnesses for poorly-constrained GW observations, as the 'lambda' inferred will be wildly large/prior-dominated
    """

    return 0.371 -0.0391*np.log(lambda_vals) + 0.001056*np.log(lambda_vals)**2


###
### SERVICE 0: General EOS structure
###

class EOSConcrete:
    """
    Class characterizing a specific EOS solution.  This structure *SHOULD* 
        - auto-build the mass-radius via a TOV solve
         - provides ability to query the lambda(m) relationship and (in the future) higher-order multipole moments; etc
    As many of these features are already provided by lalsimulation, we just front-end them
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

        if m/lal.MSUN_SI > 0.999*self.mMaxMsun:
            return 1e-8  # not exactly zero, but tiny

        k2=lalsim.SimNeutronStarLoveNumberK2(m, eos_fam)
        r=lalsim.SimNeutronStarRadius(m, eos_fam)

        m=m*lal.G_SI/lal.C_SI**2
        lam=2./(3*lal.G_SI)*k2*r**5
        dimensionless_lam=lal.G_SI*lam*(1/m)**5

        return dimensionless_lam

    def estimate_baryon_mass_from_mg(self,m):
        """
        Estimate m_b = m_g + m_g^2/(R_{1.4}/km) based on https://arxiv.org/pdf/1905.03784.pdf Eq. (6)
        Note baryon mass can be computed exactly with a TOV solution integral (e.g., Eq. 6.21 of Haensel's book)
             N_b = 4\pi (1+z_{surf}) \int_0^R e^{Phi} (rho + P/c^2)/m_b sqrt(1-2 G m(r)/r c^2)
        but lalsuite doesn't provide access to this low-level info
        !! This function is only for use when LALEOS is created. Use RePrimAnd's baryon_mass_from_mg preferably for most other purposes!!
        """
        r1p4 =lalsim.SimNeutronStarRadius(1.4*lal.MSUN_SI, self.eos_fam)/1e3
        return m + (1./r1p4)*m**2
    
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

    def test_speed_of_sound_causal_builtin(self):
        h_max = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(self.eos)
        h_crit = lalsim.SimNeutronStarEOSMinAcausalPseudoEnthalpy(self.eos)
        if h_crit < 0.999999*h_max:
            return False
        else:
            return True

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
        
        self.eos       = lalsim.SimNeutronStarEOSByName(name)
        self.eos_fam   = lalsim.CreateSimNeutronStarFamily(self.eos)
        self.mMaxMsun  = lalsim.SimNeutronStarMaximumMass(self.eos_fam) / lal.MSUN_SI
        return None



###
### SERVICE 2: EOSFromFile
###

# Example directory: EOS_Tables
try:
    dirEOSTablesBase = os.environ["EOS_TABLES"]
except:
    dirEOSTablesBase = ''
try:
    dirLALSimulationBase = os.environ["LALSIMULATION_DATADIR"]  # LAL table data
except:
    dirLALSimulationBase=''
## Add routines to find, parse standard directory of EOS files and load suitable metadata into memory
## Follow framework of NRWaveformCatalogManager3


class EOSFromTabularData(EOSConcrete):
    """ 
    Input: 
       * Tabular data (baryon_density = n , pressure = p, energy density = \rho)
       * method for primitives: this information is partially redundant, in that \ln n_b/n_ref = \int   c^2 [d rho] / (P(rho) + rho c^2), etc
          Need some specific choice for inter-edge interpolation (and redundance resolution) 
       * Low-density approximation (if desired) for filling down to surface density.  
           WARNING: Will generally match imperfectly, need some resolution to that procedure
    Creates
        * LALSimulation data structure as desired
    Warning: 
        * Currently generates intermediate data file by writing to disk
    """
    
    def __init__(self,name=None,eos_data=None,eos_units=None,reject_phase_transitions=False,debug=False, add_low_density=False):
        if eos_data is None:
            raise Exception("EOS data required to use EOSFromTabularData")
        if not(name):
            name="default"
        self.name = name
        self.bdens = None
        self.press = None
        self.edens = None
        # Assuming CGS
        try:
            self.bdens = eos_data["baryon_density"]
            self.press = eos_data["pressure"]
            self.edens = eos_data["energy_density"]
            # Convert to geometerized units 1/m^2
            #press is in dyn/cm^2 
            #edens is in gm/cm^3 and needs to be made to 1/m^2. The conversion factor is as below.
            self.press *= PRESSURE_CGS_IN_MSQUARED
            self.edens *= DENSITY_CGS_IN_MSQUARED
            # Convert to SI units
            # self.press *= 0.1                    #Converts CGS -> SI, i.e., [Ba] -> [Pa]
            # self.edens *= 0.1*(lal.C_SI*100)**2
            
            
            '''
            Use https://www.seas.upenn.edu/~amyers/NaturalUnits.pdf for reference
            Convert Pressure in CGS to SI.
            Ba -> Pa is a factor of 0.1 because [Ba] = g/(cm s^2) = 0.1 kg/(m s^2) = 0.1 Pa
            
            Convert Density in CGS-mass density [Mass/Volume] to SI-energy density [Energy/Volume].
            Converts CGS -> SI, i.e., mass density units to energy density units g/cm^3 -> J/m^3. 
            Steps: 1 g/cm^3 -> 1000 kg/m^3 . Now multiply by c^2 to get 1000kg/m^3 * c^2 = 1000*lal.C_SI^2 J/m^3. 
            OR Steps:  1 g/cm^3 multiplied by c^2 to get 1 g/cm^3 * c^2 = (lal.C_SI*100)^2 (g cm^2/s^2)/cm^3 = (lal.C_SI*100)^2 erg/cm^3 = (lal.C_SI*100)^2 *0.1 J/m^3. QED.
            
            Convert Pressure in CGS to Geometerized Units.
            First Convert Pressure in CGS to SI units. I.e.,
            Ba = 0.1 Pa
            Then to go from Pa = kg/(m s^2) to 1/m^2 multiply by lal.G_SI/lal.C_SI^4
            Hence, to go from Ba to 1/m^2, multiply by 0.1 lal.G_SI/lal.C_SI^4, or DENSITY_CGS_IN_MSQUARED/(lal.C_SI*100)**2 = 1000*lal.G_SI/lal.C_SI**2/(lal.C_SI*100)**2
            
            Convert Density in CGS to Geometerized Units
            First convert CGS-mass density to  SI-energy density as above:
            1 g/cm^3 -> 1000*lal.C_SI^2 J/m^3
            Then to go from J/m^3 = kg/(m s^2) to 1/m^2 multiply by lal.G_SI/lal.C_SI^4
            Hence, to go from g/cm^3 to 1/m^2, multiply by 1000 lal.G_SI/lal.C_SI^2
            '''
            
        except:
            self.press = eos_data[:,0]      #LALSim EOS format
            self.edens = eos_data[:,1]
        
        if reject_phase_transitions:   # Normally lalsuite can't handle regions of constant pressure. Using a pressure/density only approach isn't suited to phase transitions
            param_dict = {'energy_density': self.edens,'pressure': self.press}
            check_monotonic(param_dict,preserve_same_length = True)
            
            self.edens = param_dict['energy_density']
            self.press = param_dict['pressure']
        # Create temporary file
        if debug:
                print("Dumping to %s" % self.fname)
        eos_fname = "./" +name + "_geom.dat" # assume write acces
        np.savetxt(eos_fname, np.transpose((self.press, self.edens)), delimiter='\t', header='pressure \t energy_density ')
        
        self.eos = lalsim.SimNeutronStarEOSFromFile(eos_fname)
        self.eos_fam = lalsim.CreateSimNeutronStarFamily(self.eos)
        return None


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
        self.mMaxMsun = None
        
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

        self.mMaxMsun = lalsim.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
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

        self.eos=lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(param_dict['logP1'], param_dict['gamma1'], param_dict['gamma2'], param_dict['gamma3'])
        self.eos_fam=lalsim.CreateSimNeutronStarFamily(self.eos)
        self.mMaxMsun = lalsim.SimNeutronStarMaximumMass(self.eos_fam) / lal.MSUN_SI

        return None

######################################################################
########################## Spectral Lindblom #########################
######################################################################

class EOSLindblomSpectral(EOSConcrete):
    def __init__(self,name=None,spec_params=None,verbose=False,use_lal_spec_eos=False,check_cs=False, check_cs_builtin=True,no_eos_fam=False):
        if name is None:
            self.name = 'spectral'
        else:
            self.name=name
        self.eos = None
        self.eos_fam = None

        self.spec_params = spec_params
#        print spec_params

        if use_lal_spec_eos:
            self.eos=lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4'])  
#            self.eos=lalsim.SimNeutronStarEOSSpectralDecomposition_for_plot(spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4'],4)        
        else:
            # Create data file
            self.make_spec_param_eos(500,save_dat=True,ligo_units=True,verbose=verbose)
            # Use data file
            #print " Trying to load ",name+"_geom.dat"
            import os; #print os.listdir('.')
            cwd = os.getcwd()
            self.eos= lalsim.SimNeutronStarEOSFromFile(cwd+"/"+name+"_geom.dat")
        if check_cs:
            if check_cs_builtin:
                # this
                valid = self.test_speed_of_sound_causal_builtin()   # call parent class method
                self.eos_fam = lalsim.CreateSimNeutronStarFamily(self.eos)
                mmass = lalsim.SimNeutronStarMaximumMass(self.eos_fam) / lal.MSUN_SI
                self.mMaxMsun = mmass
            else:
                # this test requires these quantities to be built *first*
                self.eos_fam = lalsim.CreateSimNeutronStarFamily(self.eos)
                mmass = lalsim.SimNeutronStarMaximumMass(self.eos_fam) / lal.MSUN_SI
                self.mMaxMsun = mmass
                valid = self.test_speed_of_sound_causal()   # call parent class method
            if not valid:
                raise Exception(" EOS : spectral sound speed violates speed of light ")
        elif not(no_eos_fam):
            # must create these if not performing the test
            self.eos_fam = lalsim.CreateSimNeutronStarFamily(self.eos)
            mmass = lalsim.SimNeutronStarMaximumMass(self.eos_fam) / lal.MSUN_SI
            self.mMaxMsun = mmass
        else:
            self.eos_fam=None
            self.mMaxMsun = None

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
    
        new_eos_vals = check_monotonicity(new_eos_vals)  #check_monotonicity has always been and still is undefined as of 19/3/2023. First committed on 12/4/2018 https://git.ligo.org/rapidpe-rift/rift/-/commit/e6df26c04fe0e3fdf83f080db3287f69b38f930c#299
        new_eos_vals = np.vstack((np.array([0.,0.]), new_eos_vals))
        return new_eos_vals


######################################################################
###################### CAUSAL Spectral Lindblom ######################
######################################################################

class EOSLindblomSpectralSoundSpeedVersusPressure(EOSConcrete):
    """
    Based on https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.063031  <-> https://arxiv.org/pdf/2202.12285.pdf
    
    EOS spectral representation of sound speed versus pressure, as expansion of Upsilon(p): see Eq. (11).
    Uses function call to lalsuite to implement low-level interface
    
    """
    def __init__(self,name=None,spec_params=None,verbose=False,use_lal_spec_eos=True,no_eos_fam=False):
        if name is None:
            self.name = 'cs_spectral'
        else:
            self.name=name
        self.eos = None
        self.eos_fam = None
        
        self.spec_params = spec_params
        
        if use_lal_spec_eos:
            try:
                self.eos = lalsim.SimNeutronStarEOS4ParamCausalSpectralDecomposition(spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4'])
            except:
                raise Exception("A reasonable spec_params was not provided by the user. Please provide spec_params, or turn 'use_lal_spec_eos' = False and the code will expect an EoS table to be read.")
        else:
            # Create data file
            self.make_spec_param_eos(500,save_dat=True,ligo_units=True,verbose=verbose)
            # Use data file
            #print " Trying to load ",name+"_geom.dat"
            import os; #print os.listdir('.')
            cwd = os.getcwd()
            self.eos=lalsim.SimNeutronStarEOSFromFile(cwd+"/"+name+"_geom.dat")
        if not(no_eos_fam):
            self.eos_fam = lalsim.CreateSimNeutronStarFamily(self.eos)
            self.mMaxMsun = lalsim.SimNeutronStarMaximumMass(self.eos_fam) / lal.MSUN_SI
        else:
            self.eos_fam=None
            self.mMaxMsun = None
             
        return None
    
    def make_spec_param_eos(self, xvar='energy_density', yvar='pressure',npts=500, plot=False, verbose=False, save_dat=False,ligo_units=False,interpolate=False,eosname_lalsuite="SLY4"):
        """
        Load values from table of spectral parameterization values
        from separate calculations.
        Comments:
            - eos_vals is recorded as *pressure,density* pairs, because the spectral representation is for energy density vs pressure
            - units swap between geometric and CGS
            - eosname_lalsuite is used for the low-density EOS
        """
        spec_params = self.spec_params
        if not 'gamma3' in spec_params:
            spec_params['gamma3']=spec_params['gamma4']=0
        
        try :
            eos = lalsim.SimNeutronStarEOS4ParamCausalSpectralDecomposition(spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4'])
        except:
            raise Exception(" Did not load LALSimulation with Causal Spectral parameterization.")
        
        
        maxenthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos)
        #minenthalpy = lalsim.SimNeutronStarEOSMinAcausalPseudoEnthalpy(eos)
        enthalpy_index = 0.005
        enthalpy, rho, epsilon, press, speed = [], [], [], [], []
        
        Den_SI_to_CGS = 0.001 # kg m^-3 -> g cm^-3
        Energy_SI_to_CGS = 10/(lal.C_SI*100)**2 # J m^-3 *10 -> erg/cm^3 /c^2 -> g cm^-3
        Press_SI_to_CGS = 10 # Pa -> Ba ~ g cm^-1 s^-2
        
        while enthalpy_index < maxenthalpy:
            rho.append(lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(enthalpy_index, eos)*Den_SI_to_CGS)
            epsilon.append(lalsim.SimNeutronStarEOSEnergyDensityOfPseudoEnthalpy(enthalpy_index, eos)*Energy_SI_to_CGS) 
            press.append(lalsim.SimNeutronStarEOSPressureOfPseudoEnthalpy(enthalpy_index, eos)*Press_SI_to_CGS)
            speed.append(lalsim.SimNeutronStarEOSSpeedOfSound(enthalpy_index,eos)*100)    # meters -> cm
            enthalpy.append(enthalpy_index)
            enthalpy_index = enthalpy_index*1.01
        
        enthalpy, rho, epsilon, press, speed  = np.array(enthalpy), np.array(rho), np.array(epsilon), np.array(press), np.array(speed)
        
        extraction_dict_lalsim_raw = {
            'pseudo_enthalpy': enthalpy,
            'rest_mass_density': rho,                     # g cm^-3
            'baryon_density': rho/(lal.AMU_SI*1e3),       # cm^-3
            'pressure': press,                            # dyn cm^-2 ~ g cm^-1 s^-2
            'energy_density': epsilon,                    # g cm^-3
            'sound_speed_over_c': speed/(lal.C_SI*100)    # [c]
            }
        
        new_eos_vals = np.column_stack((extraction_dict_lalsim_raw[xvar], extraction_dict_lalsim_raw[yvar])) # CGS units
        
        return new_eos_vals



# https://github.com/oshaughn/RIT-matters/blob/master/communications/20230130-ROSKediaYelikar-EOSManagerSpectralUpdates/demo_reprimand.py
class EOSReprimand(EOSConcrete):
    """Pass param_dict as the dictionary of 'pseudo_enthalpy','rest_mass_density','energy_density','pressure','sound_speed_over_c' for being resolved into a TOV sequence. CGS Units only except sound_speed_over_c.
    Instead you can send a lalsim_eos which processes lalsim eos object type and produces a TOV sequence.
    load_eos takes a 2D array with pressure, energy_density and rest_mass_density (not tested).
    """
    def __init__(self,name=None,param_dict=None,lalsim_eos=None,load_eos = None, specific_internal_energy = True, m_b_units = lal.MP_SI, RePrimAnd_scale = 1e-6):
        self.name              = name
        self.pyr_eos           = None # REQUIRED, new name for reprimand structure. Provided so we can also back-port converting between two
        self.tov_seq_reprimand = None   # Stores RePrimAnd eos object
        self.eos_lal           = lalsim_eos  # NOT required, only would be useful for talking to lalsim
        self.mMaxMsun          = None  # required
        self._pyr_mrL_dat      = None # internal data for M_g, R, lambda, M_b
        self.m_b_units         = m_b_units # Base units for baryon mass for the EOS. To see why it was added in the first place, see RIFT v 0.0.15.8 and https://git.ligo.org/rapidpe-rift/rapidpe_rift_review_o4/-/wikis/Review:20230509.
        if self.eos_lal is None and load_eos is not None: self.eos_lal = EOSFromTabularData(eos_data=load_eos).eos
        
        if param_dict:
            self.update(param_dict,specific_internal_energy, RePrimAnd_scale)
        elif self.eos_lal is not None: # process LALSim EOS object
            min_pseudo_enthalpy = 0.005
            max_pseudo_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(self.eos_lal)
            hvals = max_pseudo_enthalpy* 10**np.linspace( np.log10(min_pseudo_enthalpy/max_pseudo_enthalpy),  0,num=500)
            qry_object = QueryLS_EOS(self.eos_lal)
            
            param_dict = dict()
            param_dict['pseudo_enthalpy'] = qry_object.extract_param('pseudo_enthalpy',hvals)
            param_dict['rest_mass_density'] = qry_object.extract_param('rest_mass_density',hvals)
            param_dict['energy_density'] = qry_object.extract_param('energy_density',hvals)
            param_dict['pressure'] = qry_object.extract_param('pressure',hvals)
            param_dict['sound_speed_over_c'] = qry_object.extract_param('sound_speed_over_c',hvals)
            
            #just_check_monotonicity_and_causality(param_dict)
            #param_dict = eos_monotonic_parts_and_causal_sound_speed(param_dict,preserve_same_length = False) # Don't enable by default. First check if monotonicity or causality is violated indeed, and document if it does.
            self.update(param_dict,specific_internal_energy, RePrimAnd_scale)
        else:
            print(" Warning: Empty EOS object created")
        return None
    
    def update(self,param_dict,specific_internal_energy, RePrimAnd_scale = 1e-6):
        # minimum required input, cgs units like everything else above
        # for example, you could get this from QueryLS_EOS.extract_param(name, xvals) for xvals your parameter
        #p_enthalpy = param_dict['pseudo_enthalpy']
        rho    = param_dict['rest_mass_density']
        edens  = param_dict['energy_density']
        press  = param_dict['pressure']
        cs     = param_dict['sound_speed_over_c']
        
        unew = pyr.units.geom_solar(msun_si=lal.MSUN_SI) # the argument normalizes to our units.
        spec_int_energy = edens/(rho*1.66e-27/self.m_b_units) -1  # 1.66e-27 kg is the baryon mass used in reprimand. 
        
        # 1000, and 0.1 are converting cgs quantities to SI.
        spec_int_energy_unew = spec_int_energy
        rho_unew = rho*1.66e-27/self.m_b_units*1000/unew.density # 1.66e-27/self.m_b_units is multiplied to convert rho to RePrimAnd's mass scale.
        press_unew = press*0.1/unew.pressure
        
        temp, efrac= [], []
        n_poly=1.7115960633290546  # polytropic index below lowest tabular data. Not good, should have full range.
        eps_0 = 0.0  # energy density at zero pressure
        pts_per_mag =800  # points log spaced per decaded in some parameter
        isentropic = True
        rgrho = pyr.range(min(rho_unew)*1.0000001, max(rho_unew) / 1.0000001)
        
        # Instantiate EOS
        if specific_internal_energy: self.pyr_eos = pyr.make_eos_barotr_spline(rho_unew, spec_int_energy_unew, press_unew, cs, temp, efrac, isentropic, rgrho, n_poly, unew, pts_per_mag)
        else: self.pyr_eos = pyr.make_eos_barotr_spline(rho_unew, press_unew, cs, temp, efrac, rgrho, n_poly, eps_0, unew, pts_per_mag)
        # pyr.make_eos_barotr_spline(gm1, rho_unew, spec_int_energy_unew, press_unew, cs, temp, efrac, isentropic, rgrho, n_poly, unew, pts_per_mag)
        
        self._pyr_mrL_dat, self.tov_seq_reprimand = make_mr_lambda_reprimand(self.pyr_eos,return_eos_object=True, m_b_units = self.m_b_units, RePrimAnd_scale = RePrimAnd_scale)
        self.mMaxMsun = max(self._pyr_mrL_dat[:,0])
        
        return None
    
    def baryon_mass_from_mg(self,mg):
        return self.tov_seq_reprimand.bary_mass_from_grav_mass(mg)*self.m_b_units/1.66e-27
    
    def lambda_from_m(self,mg):
        try:    # single element
            # If mg is in solar mass units(~2), it is << 1e15
            if mg <1e15: return self.tov_seq_reprimand.lambda_tidal_from_grav_mass(mg)
            # If mg is in SI units (~4e30), it is >> 1e15            
            return self.tov_seq_reprimand.lambda_tidal_from_grav_mass(mg/lal.MSUN_SI)
        except: # multi element array
            mg = np.array(mg)
            if mg[0] <1e15: return self.tov_seq_reprimand.lambda_tidal_from_grav_mass(mg)
            return self.tov_seq_reprimand.lambda_tidal_from_grav_mass(mg/lal.MSUN_SI)


# RePrimAnd
def make_mr_lambda_reprimand(eos,n_bins=800,save_tov_sequence=False,read_tov_sequence=False,return_eos_object=False, m_b_units = lal.MP_SI, RePrimAnd_scale = 1e-6):
    """
    Construct mass-radius curve from EOS using RePrimAnd (https://wokast.github.io/RePrimAnd/tov_solver.html).
    Parameter `eos` should be in RePrimAnd's eos object format, made with something like `make_eos_barotr_spline` (https://wokast.github.io/RePrimAnd/eos_barotr_ref.html).
    By default this returns the Mass_g-Radius-Lambda-Mass_b. But if `return_eos_object` is True, this will also return the RePrimAnd EOS object.
    Wolfgang Kastaun, Jay Vijay Kalinani, and Riccardo Ciolfi. Robust recovery of primitive variables in relativistic ideal magnetohydrodynamics. Phys. Rev. D, 103(2):023018, 2021. doi:10.1103/PhysRevD.103.023018.
    Roland Haas and Wolfgang Kastaun. (2023). wokast/RePrimAnd: Release 1.6 (v1.6). Zenodo. https://doi.org/10.5281/zenodo.7700296
    """
    
    """
    For units refer: https://wokast.github.io/RePrimAnd/little_helpers.html#units
    uni = pyr.units.geom_solar(g_si=6.673e-11)
    uni.length, uni.time, uni.mass
    pyr.units.geom_meter() is such that length = 1, time = 1/c, mass = Mo/1000 in [kg] . length and time are related (factor of c). mass is related to G.
    geom_meter(g_si) : length = 1, time = length /3e8, mass = 1.3465e+27 * [6.6743e-11/g_si]
    
    pyr.units.geom_solar() is such that length = 1476, time = length/c, mass = Mo in [kg] . length and time are related (factor of c) and to G. mass is independent.
    geom_solar(msun_si, g_si) : length = 1476*[g_si/6.6743e-11]*[msun_si/1.988e+30], time = length /3e8, mass = msun_si
    
    geom_umass(umass, g_si) : divides everything such that mass = umass given. DID NOT CHECK g_si dependence.
    # Units of quantities
    density is in SI/6.1758e+20 i.e. 1.98841e+30/1476.625**3 = pyr.units.geom_solar().mass / pyr.units.geom_solar().length**3
    """
    assert has_reprimand
    
    #Make TOV sequence
    acc_tov=RePrimAnd_scale*1e-2; acc_deform=RePrimAnd_scale; minsteps=500; num_samp=2000; mgrav_min=0.3
    acc = pyr.tov_acc_simple(acc_tov, acc_deform, minsteps)
    try: seq = pyr.make_tov_branch_stable(eos, acc, num_samp=num_samp, mgrav_min=mgrav_min)
    except:
        if read_tov_sequence:
            sol_units = pyr.units.geom_solar(msun_si=lal.MSUN_SI)
            seq = pyr.load_star_branch(eos, sol_units)
        else: raise Exception("No EOS supplied.")
    if save_tov_sequence:
            try:
                #bpath = p.parent
                #spath = bpath / "tov.seq.h5"
                spath = "tov.seq.h5"
                pyr.save_star_branch(str(spath), seq)
            except:
                raise Exception("Did not work. Need to send path properly.")
    #Make M-R-L relation
    u = seq.units_to_SI
    rggm1 = seq.range_center_gm1
    gm1 = np.linspace(rggm1.min, rggm1.max, n_bins)
    
    mrL_dat = np.zeros((len(gm1),4))#((n_bins,3))
    mrL_dat[:,0]  = seq.grav_mass_from_center_gm1(gm1) # Mg [Mo]
    mrL_dat[:,1]  = seq.circ_radius_from_center_gm1(gm1)*u.length/1e3 #radius [km]
    mrL_dat[:,2]  = seq.lambda_tidal_from_center_gm1(gm1)
    mrL_dat[:,3]  = seq.bary_mass_from_center_gm1(gm1)*m_b_units/1.66e-27  # Mb [Mo]. Value 1.66e-27 kg is the baryon mass used in reprimand. 
    
    c = mrL_dat[:,0]/mrL_dat[:,1]    #compactness
    
    if return_eos_object: return mrL_dat, seq
    
    return mrL_dat


def just_check_monotonicity_and_causality(param_dict):
    """
    To be used for only checking monotonicity and causality
    True means good. False means violation.
    """
    monotonicity_and_causality = {'pseudo_enthalpy_is_monotonic': True,
                      'rest_mass_density_is_monotonic': True,
                      'energy_density_is_monotonic': True,
                      'pressure_is_monotonic': True,
                      'sound_speed_over_c_is_causal': True}
    for param in param_dict:
        if param == 'sound_speed_over_c':
            if not all(param_dict[param]<=1): monotonicity_and_causality['sound_speed_over_c_is_causal'] = False
        else:
            if not all(np.diff(param_dict[param])>0) : monotonicity_and_causality[param+'_is_monotonic'] = False
    return monotonicity_and_causality


def check_monotonic(monotonic_params, other_params=None, preserve_same_length = False):
    """
    Checks monotonicity of monotonic_params and removes non-monotonic parts in it for both monotonic_params and other_params.
    By default this will reduce the length of data due to deletion of non-monotonic patches, but preserve_same_length can be turned true to keep length of data intact.
    """
    if not preserve_same_length:
        for param in monotonic_params:
            i = 0
            while i < len(monotonic_params[param])-1:
                if monotonic_params[param][i+1] <= monotonic_params[param][i]:
                    for param2 in monotonic_params:
                        try: monotonic_params[param2] = np.delete(monotonic_params[param2], i+1)
                        except:    del monotonic_params[param2][i+1]
                    if other_params is None: pass
                    else:
                        for param2 in other_params:
                            try: other_params[param2] = np.delete(other_params[param2], i+1)
                            except:    del other_params[param2][i+1]
                    i-=1
                i+=1
    else:
        for param in monotonic_params:
            i = 0
            while i < len(monotonic_params[param])-1:
                if monotonic_params[param][i+1] <= monotonic_params[param][i]:
                    monotonic_params[param][i+1] = monotonic_params[param][i]*1.01
                    if other_params is not None: 
                        if 'sound_speed_over_c' in other_params:other_params['sound_speed_over_c'][i+1] = 0
                i+=1
    return

def check_sound_speed_causal(param_dict, preserve_same_length = False):
    """Checks if sound speed exceeds 1 anywhere, and depending on the option `preserve_same_length` removes that region or forces it =1. """
    below_speed_of_light = np.where(param_dict['sound_speed_over_c']<= 1)
    if not preserve_same_length :
        param_dict = {'pseudo_enthalpy': param_dict['pseudo_enthalpy'][below_speed_of_light],
                      'rest_mass_density': param_dict['rest_mass_density'][below_speed_of_light],
                      'energy_density': param_dict['energy_density'][below_speed_of_light],
                      'pressure': param_dict['pressure'][below_speed_of_light],
                      'sound_speed_over_c': param_dict['sound_speed_over_c'][below_speed_of_light]}
    else: param_dict['sound_speed_over_c'][np.where(param_dict['sound_speed_over_c']> 1)[0]] = 1
    return param_dict


def eos_monotonic_parts_and_causal_sound_speed(param_dict, preserve_same_length = False):
    """rest_mass_density , energy_density, pressure, pseudo_enthalpy, and sound_speed_over_c are the dictionary parameters that this function expects.
    """
    param_dict_main = {'rest_mass_density': param_dict['rest_mass_density'], 'energy_density': param_dict['energy_density'],'pressure': param_dict['pressure']}
    param_dict_others = {'pseudo_enthalpy': param_dict['pseudo_enthalpy'], 'sound_speed_over_c': param_dict['sound_speed_over_c']}
    
    check_monotonic(param_dict_main, param_dict_others, preserve_same_length=preserve_same_length)
    
    param_dict = {'pseudo_enthalpy': param_dict_others['pseudo_enthalpy'],
                  'rest_mass_density': param_dict_main['rest_mass_density'],
                  'energy_density':param_dict_main['energy_density'],
                  'pressure':param_dict_main['pressure'],
                  'sound_speed_over_c':param_dict_others['sound_speed_over_c']
                  }
    param_dict = check_sound_speed_causal(param_dict, preserve_same_length=preserve_same_length)
    return param_dict

####
#### SUPPORT CODE FOLLOWS

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
    '''
    Construct mass-radius curve from EOS
    Based on modern code resources (https://git.ligo.org/publications/gw170817/bns-eos/blob/master/scripts/eos-params.py) which access low-level structures
    '''
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



###
###  EOSSequence : For large sets of EOS we must access simultaneously (100 Mb plus), pretabulated
# 
#   These will be a different data structure, where we don't necessariliy provide all the EOSConcrete structures, 
#   Example: https://zenodo.org/record/6502467#.YulOeaRE1Pw 
###



###
### SERVICE 0: General EOS structure
###

class EOSSequenceLandry:
    """
    Class characterizing a sequence of specific EOS solutions, using the Landry format.
    Assumes user provides (a) EOS realization, (b) precomputed results from TOV solve; and (c) discrete ID

    PENDING
       - mMax access
    """

    def __init__(self,name=None,fname=None,load_eos=False,load_ns=False,oned_order_name=None,oned_order_mass=None,no_sort=True,verbose=False,eos_tables_units=None):
        import h5py
        self.name=name
        self.fname=fname
        self.eos_ids = None
        self.eos_names = None   # note this array can be SORTED, use the oned_order_indx_original for original order
        self.eos_tables = None
        self.eos_tables_units = None
        self.eos_ns_tov = None
        self.oned_order_name = None
        self.oned_order_mass=oned_order_mass
        self.oned_order_values=None
        self.oned_order_indx_original = None
        self.oned_order_indx_sorted = None
        self.oned_order_sorted =False
        self.verbose=verbose
        with h5py.File(self.fname, 'r') as f:
            names = list(f['ns'].keys())
            names = natsorted(names)  # sort them sanely
            self.eos_ids = list(f['id'])
            self.eos_names = np.array(names,dtype=str)
            # The following loads a LOT into memory, as a dictionary
            if load_ns:
                if verbose:
                    print(" EOSSequenceLandry: Loading TOV results for {}".format(fname))
                # Convert to dictionary, so not closed.  Note this sucks up a lot of i/o time, and ideally we don't close the file
                self.eos_ns_tov = {}
                for name in names:
                    self.eos_ns_tov[name] = np.array(f['ns'][name])
                if verbose:
                    print(" EOSSequenceLandry: Completed TOV i/o {}".format(fname))
                create_order = False
                if oned_order_name == 'R' or oned_order_name=='r':
                    create_order=True
                    self.oned_order_name='R'  # key value in fields
                if oned_order_name == 'Lambda' or oned_order_name== 'lambda':
                    create_order=True
                    self.oned_order_name='Lambda'  # key value in fields
                if not(self.oned_order_mass):
                    # Can't order if we don't have a reference mass
                    create_order=False
                if create_order:
                    self.oned_order_indx_original = np.arange(len(self.eos_names))
                    vals = np.zeros(len(self.eos_names))
                    if self.oned_order_name =='Lambda':
                        for indx in np.arange(len(self.eos_names)):
                            vals[indx] =self.lambda_of_m_indx(self.oned_order_mass,indx)
                    if self.oned_order_name =='R':
                        for indx in np.arange(len(self.eos_names)):
                            vals[indx] =self.R_of_m_indx(self.oned_order_mass,indx)

                    # provide a list of sorted indexes, for faster lookup later if needed
                    indx_sorted = np.argsort(vals)
                    self.oned_order_indx_sorted = indx_sorted
                    if no_sort:
                        self.oned_order_values = vals
                    else:
                        # resort 'names' field with new ordering
                        # is it actually important to do the sorting?  NO, code should work with original lexographic order, since we only use nearest neighbors!
                        if verbose: 
                            print(indx_sorted)
                        self.eos_names = self.eos_names[indx_sorted]  
                        self.oned_order_values = vals[indx_sorted]
                        self.oned_order_indx_original =  self.oned_order_indx_original[indx_sorted]
                        self.oned_order_indx_sorted = np.arange(len(self.eos_names))
                        self.oned_order_sorted =True

            if load_eos:
                self.eos_tables = {}
                # Askold: generally we assume keys for the 'eos' and 'ns' are the same, but if they are not, we raise the error
                try:
                    if eos_tables_units in ['cgs', 'si', 'CGS', 'SI'] or eos_tables_units is None:
                        self.eos_tables_units = eos_tables_units
                    else:
                        raise ValueError("Invalid units for EOS tables. Please use 'cgs' or 'si'.")

                    # change the units for the EOS tables to the ones specified by the user. 
                    # NOTE! Assumes input units of the EOS tables in format: pressure/c^2 (g/cm^3), energy density/c^2 (g/cm^3) and baryon density (g/cm^3)
                    if self.eos_tables_units == 'si' or self.eos_tables_units == 'SI':
                        # constants
                        c_si = 2.99792458e8
                        eos_convert_dict = {'pressurec2': 1e3 *  c_si**2, 'energy_densityc2': 1e3 * c_si**2, 'baryon_density': 1e3, 
                        'output_units': 'pressure - N/m^2, energy density - J/m^3, baryon density - kg/m^3'}
                        eos_units_verbose = ' EOSSequenceLandry: EOS tables are converted to SI units'
                        eos_dtype_names = ('pressure', 'energy_density', 'baryon_density')
                    elif self.eos_tables_units == 'cgs' or self.eos_tables_units == 'CGS':
                        # constants
                        c_cgs = 2.99792458e10
                        eos_convert_dict = {'pressurec2': c_cgs**2, 'energy_densityc2': c_cgs**2, 'baryon_density': 1, 
                        'output_units': 'pressure - dyn/cm^2, energy density - erg/cm^3, baryon density - g/cm^3'}
                        eos_units_verbose = ' EOSSequenceLandry: EOS tables are converted to CGS units'
                        eos_dtype_names = ('pressure', 'energy_density', 'baryon_density')
                    else:
                        eos_convert_dict = {'pressurec2': 1, 'energy_densityc2': 1, 'baryon_density': 1, 
                        'output_units': 'pressure/c^2 - g/cm^3, energy density/c^2 - g/cm^3, baryon density - g/cm^3'}
                        eos_units_verbose = ' EOSSequenceLandry: EOS tables are not converted to any units'
                        eos_dtype_names = ('pressurec2', 'energy_densityc2', 'baryon_density')

                    if verbose:
                        print(" EOSSequenceLandry: Loading EOS results for {}".format(fname))
                    for name in names:
                        eos_table_orig_units = np.array(f['eos'][name])
                        # convert the units for pressure, energy density and baryon density
                        eos_table_conv_units = np.zeros(eos_table_orig_units.shape, dtype = eos_table_orig_units.dtype)
                        for key in eos_table_orig_units.dtype.names:
                            eos_table_conv_units[key] = eos_table_orig_units[key] * eos_convert_dict[key]
                        eos_table_conv_units.dtype.names = eos_dtype_names
                        self.eos_tables[name] = eos_table_conv_units
                    if verbose:
                        print(eos_units_verbose)
                        print(" Units for the EOS tables: {}".format(eos_convert_dict['output_units']))
                        print(" EOSSequenceLandry: Completed EOS i/o {}".format(fname))
        
                except KeyError:
                    raise KeyError("EOSSequenceLandry: Warning: 'eos' and 'ns' keys are not the same")
                if verbose:
                    print(" EOSSequenceLandry: Completed EOS i/o {}".format(fname))

        return None

#    Askold: this seems to be duplicated with the other m_max_of_indx function 
#    def mmax_of_indx(self,indx):
#        name = self.eos_names[indx]
#        return np.max(self.eos_ns_tov[name]['M'])

    def lambda_of_m_indx(self,m_Msun,indx):
        """
        lambda(m) evaluated for a *single* m_Msun value (almost always), for a specific indexed EOS
        
        Generally we assume the value is UNIQUE and associated with a single stable phase
        """
        if self.eos_ns_tov is None:
            raise Exception(" Did not load TOV results ")
        indx_here= indx
        if self.oned_order_sorted:   # undo  sorting, look up using ORIGINAL INDEXING
            indx_here = self.oned_order_indx_original[indx]  
        name = self.eos_names[indx_here]
        if self.verbose:
            print(" Loading from {}".format(name))
        dat = np.array(self.eos_ns_tov[name])
        # Sort masses
        indx_sort = np.argsort(dat["M"])
        # Interpolate versus m, ASSUME single-valued / no phase transition ! 
        # Interpolate versus *log lambda*, so it is smoother and more stable
        valLambda = np.log(dat["Lambda"][indx_sort])
        valM = dat["M"][indx_sort]
        return np.exp(np.interp(m_Msun, valM, valLambda))

    def R_of_m_indx(self,m_Msun,indx):
        """
        R(m) evaluated for a *single* m_Msun value (almost always), for a specific indexed EOS
        
        Generally we assume the value is UNIQUE and associated with a single stable phase; should FIX?
        """
        if self.eos_ns_tov is None:
            raise Exception(" Did not load TOV results ")
        indx_here= indx
        if self.oned_order_sorted:   # undo  sorting, look up using ORIGINAL INDEXING
            indx_here = self.oned_order_indx_original[indx]  
        name = self.eos_names[indx_here]
        if self.verbose:
            print(" Loading from {}".format(name))
        dat = np.array(self.eos_ns_tov[name])
        # Sort masses
        indx_sort = np.argsort(dat["M"])
        # Interpolate versus m, ASSUME single-valued / no phase transition ! 
        # Interpolate versus *log lambda*, so it is smoother and more stable
        valR = np.log(dat["R"][indx_sort])
        valM = dat["M"][indx_sort]
        return np.exp(np.interp(m_Msun, valM, valR))

    def m_max_of_indx(self,indx):
        if self.eos_ns_tov is None:
            raise Exception(" Did not load TOV results ")
        indx_here= indx
        if self.oned_order_sorted:   # undo  sorting, look up using ORIGINAL INDEXING
            indx_here = self.oned_order_indx_original[indx]  
        name = self.eos_names[indx_here]
        if self.verbose:
            print(" Loading from {}".format(name))
        
        return np.max(self.eos_ns_tov[name]['M'])

    def lookup_closest(self,order_val):
        """
        Given a proposed ordering statistic value, provides the *index* of the closest value.  Assumes *scalar* input
        Does not require ordering 
        """
        if self.eos_ns_tov is None:
            raise Exception(" Did not load TOV results ")
        if self.oned_order_values is None:
            raise Exception(" Did not generate ordering statistic ")

        return np.argmin( np.abs(order_val - self.oned_order_values))

    def interpolate_eos_tables(self, interp_base: str, n_points: int = 1000, verbose: bool = None):
        """
        Interpolates all EOS tables to the same grid. User can choose the base for interpolation, and the number of points on the output grid. 
        User must load the EOS tables when initializing `EOSSequenceLandry` class. (`EOSSequenceLandry(load_eos=True)`)

        Parameters
        ----------
        interp_base : str
            The column name to be used as a base for interpolation. 
            Should be one of 'pressure', 'energy_density, 'baryon_density', if units are GGS or SI!
            If no unit conversion applied (`EOSSequenceLandry(eos_tables_units=None)`), should be one of 'pressurec2', 'energy_densityc2', 'baryon_density'.
        n_points : int
            The number of points on the output grid. Default is 1000.
        verbose : bool
            If True, prints the progress of the interpolation. Default is self.verbose. (verbose from the class initialization)
        
        Returns
        -------
        eos_tables_interp : dict
            A dictionary with the same keys as self.eos_tables, but with the values being the interpolated tables.
        
        Raises
        ------
        ValueError
            If the EOS tables are not loaded.
        KeyError
            If the interp_base is not one of the allowed column names.
        
        Developer Notes
        ---------------
        * This is a simple interpolation, and can be extended to thermodynamical interpolation in the future.
        * The interpolation is done in log-space, so the output grid is logarithmically spaced.
        * The interpolation base is returned as the uniformly spaced grid in log-scale. Can be extended to interpolate the base grid as well in the future.
        * The interpolated columns remain in their original limits. Might need to add extrapolation in the future.
        * Visualization of the progress using tqdm can be added in the future.
        """

        if self.eos_tables is None:
            raise ValueError("No EOS tables loaded. Please load the EOS tables first.")

        # allowed column names for interpolation are loaded from the EOS tables columns
        allowed_interp_base = self.eos_tables[self.eos_names[0]].dtype.names
        if interp_base not in allowed_interp_base:
            raise KeyError(f"Invalid interp_base: {interp_base}. Allowed values are: {allowed_interp_base}")

        # we should have the same grid for all EOS tables thus have the same min and max values
        eos_interp_range = {key: [np.inf, -np.inf] for key in allowed_interp_base}
        for name in self.eos_names:
            for key in eos_interp_range:
                eos_interp_range[key][0] = min(eos_interp_range[key][0], np.min(self.eos_tables[name][key]))
                eos_interp_range[key][1] = max(eos_interp_range[key][1], np.max(self.eos_tables[name][key]))

        # generate the interpolation base grid (in log-space)
        eos_tables_interp = {}
        interp_base_ref = np.linspace(np.log10(eos_interp_range[interp_base][0]), np.log10(eos_interp_range[interp_base][1]), n_points)

        if verbose is None:
            verbose = self.verbose
        if verbose:
            print(f"Interpolating EOS tables to {n_points} points using {interp_base} as the base for interpolation.")

        # interpolate all EOS tables to the same grid
        # simple interpolation, we can add thermodynamical interpolation later if needed
        for name in self.eos_names:
            eos_tables_interp[name] = {}
            for key in allowed_interp_base:
                # we interpolate in log-space, and then convert back to linear space
                interp_value_log = np.interp(interp_base_ref, np.log10(self.eos_tables[name][interp_base]), np.log10(self.eos_tables[name][key]))
                eos_tables_interp[name][key] = np.power(10, interp_value_log)
            eos_tables_interp[name][interp_base] = np.power(10, interp_base_ref)

        if verbose:
            print("Completed EOS interpolation.")

        return eos_tables_interp


####
#### General lalsimulation interfacing
####


class QueryLS_EOS:
    """
    ExtractorFromEOS
      Class to repeatedly query a single lalsuite EOS object, using a common interface (e.g., to extract array outputs by name, unit conversions, etc)
    """
    def __init__(self,eos):
        self.eos = eos
        # Primitive extractors.  Assume I need to vectorize these, and that it isn't available
        
        Den_SI_to_CGS = 0.001 # kg m^-3 -> g cm^-3
        Energy_SI_to_CGS = 10/(lal.C_SI*100)**2 # J m^-3 *10 -> erg/cm^3 /c^2 -> g cm^-3
        Press_SI_to_CGS = 10 # Pa -> Ba ~ g cm^-1 s^-2
        
        extraction_dict_lalsim_raw = {
            'pseudo_enthalpy'   : lambda x: x,
            'rest_mass_density' : lambda x: lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(x, eos)*Den_SI_to_CGS,
            'baryon_density'    : lambda x: lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(x, eos)*Den_SI_to_CGS/(lal.MP_SI*1e3),  #  cm^-3
            'pressure'          : lambda x: lalsim.SimNeutronStarEOSPressureOfPseudoEnthalpy(x, eos)*Press_SI_to_CGS,
            'energy_density'    : lambda x: lalsim.SimNeutronStarEOSEnergyDensityOfPseudoEnthalpy(x,eos)*Energy_SI_to_CGS,
            'sound_speed_over_c': lambda x: lalsim.SimNeutronStarEOSSpeedOfSound(x,eos)/lal.C_SI
            }
        self.extraction_dict_lalsim = {}
        for name in         extraction_dict_lalsim_raw:
            self.extraction_dict_lalsim[name] = np.vectorize(extraction_dict_lalsim_raw[name])
    
    def convert(self,var, var_name = None):
        if not(var_name):
            raise Exception("Variable required to convert.")
        if var_name == 'rest_mass_density':
            return var*lal.C_SI**2/(lal.QE_SI*1e48) # MeV fm^-3    ## c**2/(coulomb charge) * 1/(10**39 * 10**3 * 10**6) See https://en.wikipedia.org/wiki/Electronvolt#Mass for a handy conversion. lal.C_SI**2/(lal.QE_SI*10**48) = 5.6096*10**-13
        if var_name == 'energy_density':
            return var*lal.C_SI**2/(lal.QE_SI*1e48) # MeV fm^-3
        if var_name == 'energy_density_n_sat':
            return var/(2.7e14) # nuclear saturation density in cgs = 2.7*10**14. ~ 0.16 fm^-3
        if var_name == 'sound_speed_to_cm_s':
            return var*lal.C_SI*1e2   # cm s-1
        if var_name == 'pressure':
            raise Exception('not yet implemented')
            return ##Check this conversion. possibly same as density with or without c**2. [MeV fm^-3]
        
        # IN PROGRESS
        #   - adiabatic index
        #Gamma = (self.extraction_dict_lalsim['energy_density']()*(lal.C_SI*100)**2 + self.extraction_dict_lalsim['pressure']())/(self.extraction_dict_lalsim['pressure']()) *np.square(self.extraction_dict_lalsim['sound_speed_over_c']())
    
    def extract_param(self, p, pseudo_enthalpy):
        return self.extraction_dict_lalsim[p](pseudo_enthalpy)
