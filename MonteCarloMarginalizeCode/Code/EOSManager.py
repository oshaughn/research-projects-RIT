#
#  EOSManager.py 
#
# SEE ALSO
#   - util_WriteXMLWithEOS
#   - gwemlightcurves.KNTable



import numpy as np
import os
import sys
import lal
import lalsimulation as lalsim
#import gwemlightcurves.table as gw_eos_table


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


###
### SERVICE 1: lalsimutils structure
###
#  See https://github.com/lscsoft/lalsuite/tree/master/lalsimulation/src for available types
class EOSLALSimulation:
    def __init__(self,name):
        self.name=name
        self.eos = None
        self.eos_fam = None
        self.mMaxMsun=None


        eos = lalsim.SimNeutronStarEOSByName(name)
        fam = lalsim.CreateSimNeutronStarFamily(eos)
        mmass = lalsim.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
        self.mMaxMsun = mmass
        return None


###
### SERVICE 2: EOSFromFile
###

# Example directory: EOS_Tables
dirEOSTablesBase = os.environ["EOS_TABLES"]
## Add routines to find, parse standard directory of EOS files and load suitable metadata into memory
## Follow framework of NRWaveformCatalogManager


class EOSFromDataFile:
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

# Rizzo code: EOS_param.py
class EOSPiecewisePolytrope:
    def __init__(self,name,param_dict=None):
        self.name=name
        self.eos = None
        self.eos_fam = None
        self.mMaxMsun=None


        eos=lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(param_dict['logP1'], param_dict['gamma1'], param_dict['gamma2'], param_dict['gamma3'])
        eos_fam=lalsim.CreateSimNeutronStarFamily(eos)
        mmass = lalsim.SimNeutronStarMaximumMass(fam) / lal.MSUN_SI
        self.mMaxMsun = mmass

        return None


class EOSLindblomSpectral:
    def __init__(self,name):
        self.name=name
        self.eos = None
        self.eos_fam = None
        return None





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
 
   #set p_nuc max
   p_nuc=3*10**33
   fac_min=0
   r_fin=0
   while r_fin > 28 or r_fin < 20:
      #print "Trying:"
      #print "p_c: ",(10**fac_min)*p_nuc
      try: 
         answer=lalsim.SimNeutronStarTOVODEIntegrate((10**fac_min)*p_nuc, eos)     
      except:
         fac_min=-0.05
         break 
      r_fin=answer[0]
      r_fin=r_fin*10**-3
      #print "R: ",r_fin
      if r_fin<20:
         fac_min-=0.05
      elif r_fin>28:
         fac_min+=0.01

   #set p_nuc min
   fac_max=1.6
   r_fin=20.
   while r_fin > lalsim.SimNeutronStarRadius(lalsim.SimNeutronStarMaximumMass(fam), fam)/(10**3) or r_fin < 7:
       #print "Trying min:"
       #print "p_c: ",(10**fac_max)*p_nuc
       try:
          answer=lalsim.SimNeutronStarTOVODEIntegrate((10**fac_max)*p_nuc, eos)         
          if answer[0]*10**-3 < lalsim.SimNeutronStarRadius(lalsim.SimNeutronStarMaximumMass(fam), fam)/(10**3):
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
       r_fin=answer[0]
       r_fin=r_fin*10**-3
       #print "R: ",r_fin
       if r_fin>8:
          fac_max+=0.05
       if r_fin<6:
          fac_max-=0.01
#       print 10**fac_max

   #generate mass-radius curve
   scale=np.logspace(fac_min,fac_max,200)
#   print scale 
   
   mr_array=np.zeros((200,3))
   for s,i in zip(scale,range(0,len(scale))):
#       print s
       mr_array[i,:]=lalsim.SimNeutronStarTOVODEIntegrate(s*p_nuc, eos)
      
   mr_array[:,0]=mr_array[:,0]/10**3 
   mr_array[:,1]=mr_array[:,1]/lal.MSUN_SI
   mr_array[:,2]=2./(3*lal.G_SI)*mr_array[:,2]*(mr_array[:,0]*10**3)**5
   mr_array[:,2]=lal.G_SI*mr_array[:,2]*(1/(mr_array[:,1]*lal.MSUN_SI*lal.G_SI/lal.C_SI**2))**5

   return mr_array
