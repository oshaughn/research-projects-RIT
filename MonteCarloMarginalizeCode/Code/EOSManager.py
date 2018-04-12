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
#import gwemlightcurves.table as gw_eos_table


C_CGS=2.997925*10**10 # Argh, Monica!

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


    def pressure_density_on_grid(self,logrho_grid):
        """ 
        pressure_density_on_grid
        """
        dat_out = np.zeros(len(logrho_grid))
        fam = self.eos_fam
        eos = self.eos
        npts_internal = 10000
        p_internal = np.zeros(npts_internal)
        rho_internal = np.zeros(npts_internal)
        h = np.linspace(0.0001,lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(eos),npts_internal)
        for indx in np.arange(npts_internal):
            rho_internal[indx] = lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(h[indx],eos)
            p_internal[indx] = lalsim.SimNeutronStarEOSPressureOfPseudoEnthalpy(h[indx],eos)
        logp_of_logrho = interp.interp1d(np.log10(rho),np.log10(p),kind='linear')
        return logp_of_logrho(logrho_grid)


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
dirEOSTablesBase = os.environ["EOS_TABLES"]
## Add routines to find, parse standard directory of EOS files and load suitable metadata into memory
## Follow framework of NRWaveformCatalogManager


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
        print "Checking %s" % self.name
        eos_fname = ""
        if os.path.exists(self.fname):
            # NOTE: Adapted from code by Monica Rizzo
            print "Loading from %s" % self.fname
            bdens, press, edens = np.loadtxt(self.fname, unpack=True)
            press *= 7.42591549e-25
            edens *= 7.42591549e-25
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
            print "Dumping to %s" % self.fname
            eos_fname = "./" +eos_name + "_geom.dat" # assume write acces
            np.savetxt(eos_fname, np.transpose((press, edens)), delimiter='\t')
            eos = lalsim.SimNeutronStarEOSFromFile(eos_fname)
            fam = lalsim.CreateSimNeutronStarFamily(eos)

        else:
            print " No such file ", self.fname
            sys.exit(0)

        mmass = lalsim.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
        self.mMaxMsun = mmass
        return eos, fam

    def p_rho_arrays(self):
        print self.fname
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


class EOSLindblomSpectral:
    def __init__(self,name=None,spec_params=None):
        if name is None:
            self.name = 'spectral'
        else:
            self.name=name
        self.eos = None
        self.eos_fam = None

        self.spec_params = spec_params
        print spec_params

        # Create data file
        self.make_spec_param_eos(500,save_dat=True,ligo_units=True)

        # Use data file
        print " Trying to load ",name+"_geom.dat"
        import os; #print os.listdir('.')
        cwd = os.getcwd()
        self.eos=eos = lalsim.SimNeutronStarEOSFromFile(cwd+"/"+name+"_geom.dat")
        self.fam = fam=lalsim.CreateSimNeutronStarFamily(eos)
        mmass = lalsim.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
        self.mMaxMsun = mmass


#        my_fromfile_eos =EOSFromDataFile(fname=name+"_spec.dat")
#        self.eos = my_fromfile_eos.eos
#        self.fam = my_fromfile_eos.fam
#        self.mMaxMsun = my_fromfile_eos.mMaxMsun
        return None



    def make_spec_param_eos(self, npts, plot=False, verbose=False, save_dat=False,ligo_units=False):
        """
        Load values from table of spectral parameterization values
        Table values taken from https://arxiv.org/pdf/1009.0738.pdf
        """

        spec_params = self.spec_params
        coefficients=np.array([spec_params['gamma1'], spec_params['gamma2'], spec_params['gamma3'], spec_params['gamma4']])
        p0=spec_params['p0']
        eps0=spec_params['epsilon0']
        xmax=spec_params['xmax'] 

    #make p range
        x_range=np.linspace(0,xmax,npts)
        p_range=p0*np.exp(x_range)
       
        eos_vals=np.zeros((npts,2))
        eos_vals[:,1]=p_range

        for i in range(0, len(x_range)):
            eos_vals[i,0]=epsilon(x_range[i], p0, eps0, coefficients)
            if verbose==True:
                print "x:",x_range[i],"p:",p_range[i],"p0",p0,"epsilon:",eos_vals[i,0]
  
    #doing as those before me have done and using SLY4 as low density region
        # THIS MUST BE FIXED TO USE STANDARD LALSUITE ACCESS, do not assume the file exists
        low_density=np.loadtxt(dirEOSTablesBase+"/LALSimNeutronStarEOS_SLY4.dat")
        low_density[:,0]=low_density[:,0]*C_CGS**2/(7.42591549*10**(-25))
        low_density[:,1]=low_density[:,1]*C_CGS**2/(7.42591549*10**(-25))
        low_density[:,[0, 1]] = low_density[:,[1, 0]]

        cutoff=eos_vals[0,:]   
 
        for i in range(0, len(low_density)):
            if low_density[i,0] > cutoff[0] or low_density[i,1] > cutoff[1]:   
                break_pt=i
                break 
    
        eos_vals=np.vstack((low_density[0:break_pt,:], eos_vals)) 

        # if plot==True:
        #     plt.plot(np.log10(eos_vals[:,0]/(C_CGS**2)), np.log10(eos_vals[:,1]/(C_CGS**2)), 'o')
        #     plt.xlabel("Energy Density")
        #     plt.ylabel("Pressure")
        #     plt.show() 

        if ligo_units:
            eos_vals *= 7.42591549e-25/(C_CGS**2)

        if save_dat == True:
            np.savetxt(self.name+"_geom.dat", eos_vals)

        return eos_vals


def gamma_of_x(x, coeffs):
        """
        Eq 6 from https://arxiv.org/pdf/1009.0738.pdf
        """
        gamma=0
        for i in range(0,len(coeffs)):
            gamma+=coeffs[i]*x**i 
        gamma=np.exp(gamma)  
        return gamma
  
def mu(x, coeffs):
        """
        Eq 8 from https://arxiv.org/pdf/1009.0738.pdf
        """

        def int_func(x_prime):
            return (gamma_of_x(x_prime, coeffs))**(-1)    

        if isinstance(x, (list, np.ndarray)):
            val=np.zeros(len(x))
            for i in range(0,len(x)):
                tmp=quad(int_func, 0, x[i])
                val[i]=tmp[0]  
        else:    
            val=quad(int_func, 0, x)

        return np.exp(-1.*val[0])

def epsilon(x, p0, eps0, coeffs):
        """
        Eq. 7 from https://arxiv.org/pdf/1009.0738.pdf
        """
        def int_func(x_prime):
            num = mu(x_prime, coeffs)*np.exp(x_prime)
            denom = gamma_of_x(x_prime, coeffs)
            return num / denom
          
        mu_of_x=mu(x, coeffs)  
        val=quad(int_func, 0, x)
        #val=romberg(int_func, 0, x, show=True)   
        eps=(eps0*C_CGS**2)/mu_of_x + p0/mu_of_x * val[0]
 
        return eps




###
### Utilities
###

# Rizzo
def make_mr_lambda(eos):
   """
   construct mass-radius curve from EOS    
   DOES NOT YET WORK RELIABLY
   """
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
           print "R: ",r_fin, r_ref, " M: ", answer[1]/lal.MSUN_SI, m_ref , m_min # should converge to maximum mass
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
