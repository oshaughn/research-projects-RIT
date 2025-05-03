# Copyright (C) 2012  Evan Ochsner, R. O'Shaughnessy
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
A collection of useful data analysis routines
built from the SWIG wrappings of LAL and LALSimulation.
"""
import sys
import copy
import types
has_external_teobresum=False
import os
info_use_ext = True
if 'RIFT_NO_EXTERNAL' in os.environ:
    info_use_ext = False
    has_external_teobresum=False
else:
 try:
    import EOBRun_module
    has_external_teobresum=True
 except:
    has_external_teobresum=False
    True; # print(" - no EOBRun (TEOBResumS) - ")
info_use_resum_polarizations = False
if 'RIFT_RESUM_POLARIZATIONS' in os.environ:
    info_use_resum_polarizations=True  # fallback to ChooseTDModesFromPolarizations for TEOBResumS (since ChooseTDModes is not available at present)
from six.moves import range

log_loud = False
if 'RIFT_LOUD' in os.environ:
    log_loud = True

import numpy as np
from numpy import sin, cos
from scipy import interpolate
from scipy import signal
from scipy.optimize import fsolve
import scipy  # for decimate
try:
    import precession
except ImportError:
    print('Import Error - module missing. Please install the module "precession."')
def safe_int(mystr):
   try:
        return int(mystr)
   except:
        return None
sci_ver = list(map(safe_int, scipy.version.version.split('.')))  # scipy version number as int list.

from igwn_ligolw import lsctables, utils, ligolw #, table, ,ilwd # check all are needed
from glue.lal import Cache

lalmetaio_old_style=True
import lalmetaio
if not(hasattr(lalmetaio.SimInspiralTable, 'geocent_end_time_ns')):
#    print(" NEW style lalmetaio ")
    lalmetaio_old_style=False  # no geocent_end_time_ns

import lal
import lalsimulation as lalsim
#import lalinspiral
import lalmetaio

#from pylal import seriesutils as series
from lal.series import read_psd_xmldoc
from lalframe import frread

import RIFT.misc.tools as tools 

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, R. O'Shaughnessy"


rosDebugMessagesContainer = [False]
rosDebugMessagesLongContainer = [False]
if log_loud:
    print( "[Loading lalsimutils.py : MonteCarloMarginalization version]",file=sys.stderr)
    print( "  scipy : ", scipy.__version__, file=sys.stderr)
    print("  numpy : ", np.__version__,file=sys.stderr)

TOL_DF = 1.e-6 # Tolerence for two deltaF's to agree


#spin_convention = "radiation"
spin_convention = "L"

# from functools import wraps
# def strip_ilwdchar(_ContentHandler):
#     """Wrap a LIGO_LW content handler to swap ilwdchar for int on-the-fly
#     when reading a document
#     This is adapted from :func:`ligo.skymap.utils.ilwd`, copyright
#     Leo Singer (GPL-3.0-or-later).
#     This is taken directly from https://github.com/gwpy/gwpy/blob/master/gwpy/io/ligolw.py#L92
#     """
#     from igwn_ligolw.lsctables import TableByName
#     from igwn_ligolw.table import (Column, TableStream)
#     from igwn_ligolw.types import (FromPyType, ToPyType)

#     class IlwdMapContentHandler(_ContentHandler):

#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             self._idconverter = {}

#         @wraps(_ContentHandler.startColumn)
#         def startColumn(self, parent, attrs):
#             result = super().startColumn(parent, attrs)

#             # if an old ID type, convert type definition to an int
#             if result.Type == "ilwd:char":
#                 old_type = ToPyType[result.Type]

#                 def converter(old):
#                     return int(old_type(old))

#                 self._idconverter[(id(parent), result.Name)] = converter
#                 result.Type = FromPyType[int]

#             try:
#                 validcolumns = TableByName[parent.Name].validcolumns
#             except KeyError:  # parent.Name not in TableByName
#                 return result
#             if result.Name not in validcolumns:
#                 stripped_column_to_valid_column = {
#                     Column.ColumnName(name): name
#                     for name in validcolumns
#                 }
#                 if result.Name in stripped_column_to_valid_column:
#                     result.setAttribute(
#                         'Name',
#                         stripped_column_to_valid_column[result.Name],
#                     )

#             return result

#         @wraps(_ContentHandler.startStream)
#         def startStream(self, parent, attrs):
#             result = super().startStream(parent, attrs)
#             if isinstance(result, TableStream):
#                 loadcolumns = set(parent.columnnames)
#                 if parent.loadcolumns is not None:
#                     loadcolumns &= set(parent.loadcolumns)
#                 pid = id(parent)
#                 result._tokenizer.set_types([
#                     self._idconverter.pop((pid, colname), pytype)
#                     if colname in loadcolumns else None
#                     for pytype, colname in zip(
#                         parent.columnpytypes,
#                         parent.columnnames,
#                     )
#                 ])
#             return result

#     return IlwdMapContentHandler


# updated in igwn_ligolw
#cthdler = strip_ilwdchar(ligolw.LIGOLWContentHandler) #defines a content handler to load xml grids
cthdler = ligolw.LIGOLWContentHandler
#lsctables.use_in(cthdler)


waveform_approx_limit_dict = {
  "NRHybSur3dq8": {  "q-min":1./8, "chi-max":0.8
     },
  "NRSur7dq2": {  "q-min":1./2, "chi-max":0.8
     },
  "NRSur7dq4": { "q-min":1./4, "chi-max":0.8
     },

}



# Check lal version (lal.LAL_MSUN_SI or not).  Enables portability through the version transition.
try:
    x=lal.LAL_MSUN_SI
except:
#    print  " New style : no LAL prefix"
    lsu_MSUN=lal.MSUN_SI
    lsu_PC = lal.PC_SI
    lsu_G = lal.G_SI
    lsu_C = lal.C_SI
    lsu_PI=lal.PI
    lsu_TAPER_NONE=lalsim.SIM_INSPIRAL_TAPER_NONE
    lsu_TAPER_START=lalsim.SIM_INSPIRAL_TAPER_START
    lsu_TAPER_END=lalsim.SIM_INSPIRAL_TAPER_END
    lsu_TAPER_STARTEND=lalsim.SIM_INSPIRAL_TAPER_STARTEND
    lsu_DimensionlessUnit = lal.DimensionlessUnit
    lsu_HertzUnit = lal.HertzUnit
    lsu_SecondUnit = lal.SecondUnit

    lsu_PNORDER_NEWTONIAN = lalsim.PNORDER_NEWTONIAN
    lsu_PNORDER_HALF = lalsim.PNORDER_HALF
    lsu_PNORDER_ONE = lalsim.PNORDER_ONE
    lsu_PNORDER_ONE_POINT_FIVE = lalsim.PNORDER_ONE_POINT_FIVE
    lsu_PNORDER_TWO = lalsim.PNORDER_TWO
    lsu_PNORDER_TWO_POINT_FIVE = lalsim.PNORDER_TWO_POINT_FIVE
    lsu_PNORDER_THREE = lalsim.PNORDER_THREE
    lsu_PNORDER_THREE_POINT_FIVE = lalsim.PNORDER_THREE_POINT_FIVE
else:
    print(" Old style :  LAL prefix")
    lsu_MSUN=lal.LAL_MSUN_SI
    lsu_PC=lal.LAL_PC_SI
    lsu_G = lal.LAL_G_SI
    lsu_C = lal.LAL_C_SI
    lsu_PI = lal.LAL_PI
    lsu_TAPER_NONE=lalsim.LAL_SIM_INSPIRAL_TAPER_NONE
    lsu_TAPER_START=lalsim.LAL_SIM_INSPIRAL_TAPER_START
    lsu_TAPER_END=lalsim.LAL_SIM_INSPIRAL_TAPER_END
    lsu_TAPER_STARTEND=lalsim.LAL_SIM_INSPIRAL_TAPER_STARTEND
    lsu_DimensionlessUnit = lal.lalDimensionlessUnit
    lsu_HertzUnit = lal.lalHertzUnit
    lsu_SecondUnit = lal.lalSecondUnit

    lsu_PNORDER_NEWTONIAN = lalsim.LAL_PNORDER_NEWTONIAN
    lsu_PNORDER_HALF = lalsim.LAL_PNORDER_HALF
    lsu_PNORDER_ONE = lalsim.LAL_PNORDER_ONE
    lsu_PNORDER_ONE_POINT_FIVE = lalsim.LAL_PNORDER_ONE_POINT_FIVE
    lsu_PNORDER_TWO = lalsim.LAL_PNORDER_TWO
    lsu_PNORDER_TWO_POINT_FIVE = lalsim.LAL_PNORDER_TWO_POINT_FIVE
    lsu_PNORDER_THREE = lalsim.LAL_PNORDER_THREE
    lsu_PNORDER_THREE_POINT_FIVE = lalsim.LAL_PNORDER_THREE_POINT_FIVE

try:
    lalSEOBv4 = lalsim.SEOBNRv4
    lalIMRPhenomD = lalsim.IMRPhenomD
except:
    lalSEOBv4 =-1
    lalIMRPhenomD = -2

try:
    lalTEOBv2 = -3 # not implemented
    lalTEOBv4 = lalsim.SEOBNRv4T
except:
    lalTEOBv2 = -3
    lalTEOBv4 = -4
try:
    lalSEOBNRv4HM = lalsim.SEOBNRv4HM
except:
    lalSEOBNRv4HM = -5

try:
    lalSEOBNRv4P = lalsim.SEOBNRv4P
    lalSEOBNRv4PHM = lalsim.SEOBNRv4PHM
except:
    lalSEOBNRv4P = -6
    lalSEOBNRv4PHM = -7

try:
    lalNRSur7dq4 = lalsim.NRSur7dq4
    lalNRSur7dq2 = lalsim.NRSur7dq2
    lalNRHybSur3dq8 = lalsim.NRHybSur3dq8
except:
    lalNRSur7dq4 = -8
    lalNRSur7dq2 = -9
    lalNRHybSur3dq8 = -10

try:
   lalIMRPhenomHM = lalsim.IMRPhenomHM
   lalIMRPhenomXHM = lalsim.IMRPhenomXHM
   lalSEOBNRv4HM_ROM = lalsim.SEOBNRv4HM_ROM
   lalIMRPhenomXP = lalsim.IMRPhenomXP
   lalIMRPhenomXPHM = lalsim.IMRPhenomXPHM
   lalIMRPhenomXO4a = lalsim.IMRPhenomXO4a
   
except:
   lalIMRPhenomXP = -11
   lalIMRPhenomHM = -14
   lalIMRPhenomXHM = -15
   lalSEOBNRv4HM_ROM = -16
   lalIMRPhenomXPHM = -17
   lalIMRPhenomXO4a = -18

pending_FD_approx = ['IMRPhenomXP_NRTidalv2','IMRPhenomXP_NRTidalv3']
pending_approx_code = {}
pending_default_code = -19
for name in pending_FD_approx:
    if hasattr(lalsim, name):
        pending_approx_code[name] = getattr(lalsim, name)
    else:
        pending_approx_code[name] = pending_default_code
        pending_default_code += -1  # same convention as above
def check_FD_pending(code):
    return code in  pending_approx_code.values()  # just test if it is in the list of pending, NOT that useful since everything appended here.



try:
   my_junk = lalsim.SimInspiralChooseFDModes
   is_ChooseFDModes_present =True
except:
   is_ChooseFDModes_present =False

try:
   lalIMRPhenomTP = lalsim.IMRPhenomTP
   lalIMRPhenomTPHM = lalsim.IMRPhenomTPHM
except:
   lalIMRPhenomTP = -12
   lalIMRPhenomTPHM = -13

MsunInSec = lal.MSUN_SI*lal.G_SI/lal.C_SI**3

def modes_to_k(modes):
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]
def mean_anomaly_from_true(true_anomaly, eccentricity):
    return (
        np.arctan2(
            -np.sqrt(
                1 - eccentricity**2)
            * np.sin(true_anomaly),
            - eccentricity - np.cos(true_anomaly)
        ) + np.pi
        - eccentricity
        * np.sqrt(1 - eccentricity**2)
        * np.sin(true_anomaly)
        / (
            1 + eccentricity
            * np.cos(true_anomaly)
        )
    )
def eccentric_anomaly_from_mean(mean_anomaly, eccentricity):
    func = lambda E : E - eccentricity*np.sin(E) - mean_anomaly
    return fsolve(func, x0=mean_anomaly)
def true_anomaly_from_eccentric(eccentric_anomaly, eccentricity):
    # formula from https://ui.adsabs.harvard.edu/abs/1973CeMec...7..388B/abstract
    # avoids numerical issues

    beta = eccentricity / (1+np.sqrt(1 - eccentricity**2))
    return eccentric_anomaly + 2 * np.arctan2(beta * np.sin(eccentric_anomaly), 1 - beta * np.cos(eccentric_anomaly))

# https://www.lsc-group.phys.uwm.edu/daswg/projects/lal/nightly/docs/html/_l_a_l_sim_inspiral_8c_source.html#l02910
def lsu_StringFromPNOrder(order):
    if (order == lsu_PNORDER_NEWTONIAN):
        return "newtonian"
    elif (order == lsu_PNORDER_HALF):
        return "oneHalfPN"
    elif (order == lsu_PNORDER_ONE):
        return "onePN"
    elif (order == lsu_PNORDER_ONE_POINT_FIVE):
        return "onePointFivePN"
    elif (order == lsu_PNORDER_TWO):
        return "twoPN"
    elif (order == lsu_PNORDER_TWO_POINT_FIVE):
        return "twoPointFivePN"
    elif (order == lsu_PNORDER_THREE):
        return "threePN"
    elif (order == lsu_PNORDER_THREE_POINT_FIVE):
        return "threePointFivePN"
    elif (order == -1):  # highest available
        return "threePointFivePN"
    else:
        raise ("Unknown PN order ", order)

#
# Class to hold arguments of ChooseWaveform functions
#

valid_params = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'chi1_perp', 'chi2_perp', 'chi1_perp_bar', 'chi2_perp_bar','chi1_perp_u', 'chi2_perp_u', 's1z_bar', 's2z_bar', 'lambda1', 'lambda2', 'theta','phi', 'phiref',  'psi', 'incl', 'tref', 'dist', 'mc', 'mc_ecc', 'eta', 'delta_mc', 'chi1', 'chi2', 'thetaJN', 'phiJL', 'theta1', 'theta2', 'cos_theta1', 'cos_theta2',  'theta1_Jfix', 'theta2_Jfix', 'psiJ', 'beta', 'cos_beta', 'sin_phiJL', 'cos_phiJL', 'phi12', 'phi1', 'phi2', 'LambdaTilde', 'DeltaLambdaTilde', 'lambda_plus', 'lambda_minus', 'q', 'mtot','xi','chiz_plus', 'chiz_minus', 'chieff_aligned','fmin','fref', "SOverM2_perp", "SOverM2_L", "DeltaOverM2_perp", "DeltaOverM2_L", "shu","ampO", "phaseO",'eccentricity','eccentricity_squared', 'chi_pavg','mu1','mu2','eos_table_index','meanPerAno','a6c','E0','p_phi0','hypclass']

# so far, used for puffball, to prevent insanity (infinite growth) and/or death to downselect
#   - note we also provide for extrinsic: RA (phi), phiref, psi, just in case we need it in the future
periodic_params = {'phi1':2*np.pi, 'phi2':2*np.pi, 'phiref':2*np.pi, 'psi':np.pi, 'meanPerAno':2*np.pi, 'phi':2*np.pi, 'phiJL':2*np.pi, 'psiJ':2*np.pi}

tex_dictionary  = {
 "mtot": r'$M$',
 "mc": r'${\cal M}_c$',
 "mc_ecc": r'${\cal M}_{\rm c,ecc}$',
 "m1": '$m_1$',
 "m2": '$m_2$',
 "m1_source": r'$m_{1,source}$',
 "m2_source": r'$m_{2,source}$',
 "mtotal_source": r'$M_{source}$',
  "q": "$q$",
  "delta" : r"$\delta$",
  "delta_mc" : r"$\delta$",
  "beta" : r"$\beta$",
  "cos_beta" : r"$\cos(\\beta)$",
  "sin_beta" : r"$\sin(\\beta)$",
  "sin_phiJL" : r"$\sin(\\phi_{JL})$",
  "cos_phiJL" : r"$\cos(\\phi_{JL})$",
  "phi12" : r"$\phi_{12}$",
  "DeltaOverM2_perp" : r"$\Delta_\perp$",
  "DeltaOverM2_L" : r"$\Delta_{||}$",
  "SOverM2_perp" : r"$S_\perp$",
  "SOverM2_L" : r"$S_{||}$",
  "eta": r"$\eta$",
  "chi_eff": r"$\chi_{eff}$",
  "xi": r"$\chi_{eff}$",
  "chi_p": r"$\chi_{p}$",
   "chiMinus":r"$\chi_{eff,-}$",
  "chiz_plus":r"$\chi_{z,+}$",
  "chiz_minus":r"$\chi_{z,-}$",
  "chi_pavg":r"$\langle\chi_{p}\rangle$", 
  "lambda_plus":r"$\lambda_{+}$",
  "lambda_minus":r"$\lambda_{-}$",
  "s1z": r"$\chi_{1,z}$",
  "s2z": r"$\chi_{2,z}$",
  "s1x": r"$\chi_{1,x}$",
  "s2x": r"$\chi_{2,x}$",
  "s1y": r"$\chi_{1,y}$",
  "s2y": r"$\chi_{2,y}$",
  "eccentricity":"$e$",
  "meanPerAno":"$l_{gw}$",
  "E0" : "$E_0 / M$",
  "p_phi0" : r"$p__{\phi}^0",  
  "a6c":"$a^c_6$",
  # tex labels for inherited LI names
 "a1z": r'$\chi_{1,z}$',
 "a2z": r'$\chi_{2,z}$',
 "mtotal": r'$M_{tot}$',
 "theta1":r"$\theta_1$",
 "theta2":r"$\theta_2$",
 "phi1":r"$\phi_1$",
 "phi2":r"$\phi_2$",
 "cos_theta1":r"$\cos \\theta_1$",
 "cos_theta2":r"$\cos \\theta_2$",
 "chi1_perp": r"$\chi_{1,\perp}$",
 "chi2_perp": r"$\chi_{2,\perp}$",
  'chi1':r'$|\chi_1|$',
  'chi2':r'$|\chi_2|$',
  'lambda1':r'$\lambda_1$',
  'lambda2':r'$\lambda_2$',
  'LambdaTilde': r'$\tilde{\Lambda}$',
  'lambdat': r'$\tilde{\Lambda}$',
  'DeltaLambdaTilde': r'$\Delta\tilde{\Lambda}$',
  'dlambdat': r'$\Delta\tilde{\Lambda}$',
  'distance':r'$d_L$',
  'mu1':r'$\mu_1$',
  'mu2':r'$\mu_2$',
}

# Exponent used to define the u coordinate scaling
p_R = 0.25

class ChooseWaveformParams:
    """
    Class containing all the arguments needed for SimInspiralChooseTD/FDWaveform
    plus parameters theta, phi, psi, radec to go from h+, hx to h(t)

    if radec==True: (theta,phi) = (DEC,RA) and strain will be computed using
            XLALSimDetectorStrainREAL8TimeSeries
    if radec==False: then strain will be computed using a simple routine 
            that assumes (theta,phi) are spherical coord. 
            in a frame centered at the detector
    """
    def __init__(self, phiref=0., deltaT=1./4096., m1=10.*lsu_MSUN, 
            m2=10.*lsu_MSUN, s1x=0., s1y=0., s1z=0., 
            s2x=0., s2y=0., s2z=0., fmin=40., fref=0., dist=1.e6*lsu_PC,
            incl=0., lambda1=0., lambda2=0., waveFlags=None, nonGRparams=None,
            ampO=0, phaseO=7, approx=lalsim.TaylorT4, 
            theta=0., phi=0., psi=0., tref=0., radec=False, detector="H1",
            deltaF=None, fmax=0., # for use w/ FD approximants
            taper=lsu_TAPER_NONE, # for use w/TD approximants
            eccentricity=0., # make eccentricity a parameter
            meanPerAno=0., # make meanPerAno a parameter
            E0=0., # make E0/M a parameter
            p_phi0=0., # make j_hyp a parameter
            a6c=10000. # EOB Parameter
    ):
        self.phiref = phiref
        self.deltaT = deltaT
        self.m1 = m1
        self.m2 = m2
        self.s1x = s1x
        self.s1y = s1y
        self.s1z = s1z
        self.s2x = s2x
        self.s2y = s2y
        self.s2z = s2z
        self.fmin = fmin
        self.fref = fref
        self.dist = dist
        self.incl = incl
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.waveFlags = waveFlags
        self.nonGRparams = nonGRparams
        self.ampO = ampO
        self.phaseO = phaseO
        self.approx = approx
        self.theta = theta     # DEC.  DEC =0 on the equator; the south pole has DEC = - pi/2
        self.phi = phi         # RA.   
        self.psi = psi
        self.a6c=a6c
        self.longAscNodes = self.psi # port to master
        self.eccentricity=eccentricity
        self.meanPerAno=meanPerAno
        self.E0 = E0
        self.p_phi0 = p_phi0
        self.tref = tref
        self.radec = radec
        self.detector = "H1"
        self.deltaF=deltaF
        self.fmax=fmax
        self.taper = taper
        self.snr = None  # Only used for compatibility with AMR grid rapid_pe, should not usually be setting or using this
        self.eos_table_index = None  # only used for compatibility with tabular EOS formalisms, note user MUST always provide lambda1, lambda2 correctly for waveform generators!  

        # force this waveform's PN order to be 3 to avoid crashes
        if self.approx == lalsim.EccentricTD:
            self.phaseO = 3

    # From Pankow/master
    # See also: https://git.ligo.org/lscsoft/bilby_pipe/-/merge_requests/371/diffs
    try:
        _LAL_DICT_PARAMS = {"Lambda1": "lambda1", "Lambda2": "lambda2", "ampO": "ampO", "phaseO": "phaseO"}
        _LAL_DICT_PTYPE = {"Lambda1": lal.DictInsertREAL8Value, "Lambda2": lal.DictInsertREAL8Value, "ampO": lal.DictInsertINT4Value, "phaseO": lal.DictInsertINT4Value}
    except:
        print(" lalsimutils: Warning: Running with non-master version of lal ! ")
    def to_lal_dict(self):
        extra_params = lal.CreateDict()
        for k, p in ChooseWaveformParams._LAL_DICT_PARAMS.items():
            typfunc = ChooseWaveformParams._LAL_DICT_PTYPE[k]
            typfunc(extra_params, k, getattr(self, p))
        # Properly add tidal parammeters
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(extra_params, self.lambda1)
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(extra_params, self.lambda2)
        return extra_params

    def to_lal_dict_extended(self,extra_args_dict=None):
        """
        to_lal_dict_extended:
            Extended implementation, to address massive proliferation of options to control waveform version
            See  https://git.ligo.org/lscsoft/bilby_pipe/-/merge_requests/371/diffs
        """
        extra_params = self.to_lal_dict()
        if extra_args_dict:
            extra_keys = list(set(list(extra_args_dict)) - set(ChooseWaveformParams._LAL_DICT_PARAMS))  # list of new keys I need to find in lalsmulation
            for key in extra_keys:
                func = getattr(
                    lalsim, "SimInspiralWaveformParamsInsert" + key, 0
                    )
                if func != 0:
                    func(extra_params, extra_args_dict[key])
        return extra_params

    def manual_copy(self):
        P=self.copy()
        waveFlags = self.waveFlags
        if waveFlags:
            waveFlagsNew = lalsim.SimInspiralCreateWaveformFlags()
            lalsim.SimInspiralSetSpinOrder(waveFlagsNew, lalsim.SimInspiralGetSpinOrder(waveFlags))
            lalsim.SimInspiralSetTidalOrder(waveFlagsNew, lalsim.SimInspiralGetTidalOrder(waveFlags))
            P.waveFlags = waveFlagsNew
        return P

    def swap_components(self):
        s1x,s1y,s1z = self.s1x,self.s1y,self.s1z
        s2x,s2y,s2z = self.s2x,self.s2y,self.s2z
        m1 =self.m1
        m2 = self.m2
        self.s1x,self.s1y,self.s1z  = s2x,s2y,s2z 
        self.s2x,self.s2y,self.s2z  = s1x,s1y,s1z
        self.m1 = m2
        self.m2  = m1
        lam1,lam2 =self.lambda1,self.lambda2
        self.lambda2=lam1
        self.lambda1=lam2
        self.phiref = self.phiref+np.pi

        

    def assign_param(self,p,val):
        """
        assign_param
        Syntatic sugar to assign parameters to values.
        Provides ability to specify a binary by its values of 
            - mchirp, eta
            - system frame parameters
        VERY HELPFUL if you want to change just one parameter at a time (e.g., for Fisher )
        """
        if p == 'mtot':
            # change implemented at fixed chi1, chi2, eta
            q = self.m2/self.m1
            self.m1,self.m2 = np.array( [1./(1+q), q/(1.+q)])*val
            return self
        if p == 'q':
            # change implemented at fixed Mtot (NOT mc)
            mtot = self.m2+self.m1
            self.m1,self.m2 = np.array( [1./(1+val), val/(1.+val)])*mtot
            return self
        if p == 'log_mc':
            # change implemented at fixed chi1, chi2, eta
            eta = symRatio(self.m1,self.m2)
            self.m1,self.m2 = m1m2(10**val,eta)
            return self
        if p == 'mc' or p=='mc_ecc':
            # change implemented at fixed chi1, chi2, eta (and ecc)
            eta = symRatio(self.m1,self.m2)
            self.m1,self.m2 = m1m2(val,eta)
            return self
        if p == 'eta':
            # change implemented at fixed chi1, chi2, mc
            mc = mchirp(self.m1,self.m2)
            self.m1,self.m2 = m1m2(mc,val)
            return self
        if p == 'delta':
            # change implemented at fixed chi1, chi2, M
            M = self.m1 + self.m2
            self.m1 = M*(1+val)/2
            self.m2 = M*(1-val)/2
            return self
        if p == 'delta_mc':
            # change implemented at fixed chi1, chi2, *mc*
            eta_here = 0.25*(1 - val*val)
            self.assign_param('eta', eta_here)
            return self
        if p == 'chiz_plus':
            # Designed to give the benefits of sampling in chi_eff, without introducing a transformation/prior that depends on mass
            # Fixes chiz_minus by construction
            czm = (self.s1z-self.s2z)/2.
            czp = val
            self.s1z = (czp+czm)
            self.s2z = (czp-czm)
            return self
        if p == 'chiz_minus':
            # Designed to give the benefits of sampling in chi_eff, without introducing a transformation/prior that depends on mass
            # Fixes chiz_plus by construction
            czm =  val
            czp = (self.s1z+self.s2z)/2.
            self.s1z = (czp+czm)
            self.s2z = (czp-czm)
            return self
        if p == 's1z_bar':
            self.s1z = val
            return self
        if p == 's2z_bar':
            self.s2z = val
            return self
        if p == 'chi1_perp_bar':
#            chi1_perp = np.sqrt(self.s1x**2+self.s2y**2)
            phi1 = np.arctan2(self.s1y, self.s1x)
            chi1_perp_new = val*np.sqrt(1-self.s1z**2)  # R=Rbar*(1-z^2)^0.5 
            self.s1x = chi1_perp_new*np.cos(phi1)
            self.s1y = chi1_perp_new*np.sin(phi1)
            return self
        if p == 'chi1_perp_u':
#            chi1_perp = np.sqrt(self.s1x**2+self.s2y**2)
            Rb = np.power(val, 1./p_R)
            phi1 = np.arctan2(self.s1y, self.s1x)
            chi1_perp_new = Rb*np.sqrt(1-self.s1z**2)  # R=Rbar*(1-z^2)^0.5 
            self.s1x = chi1_perp_new*np.cos(phi1)
            self.s1y = chi1_perp_new*np.sin(phi1)
            return self
        if p == 'chi2_perp_bar':
#            chi1_perp = np.sqrt(self.s1x**2+self.s2y**2)
            phi2 = np.arctan2(self.s2y, self.s2x)
            chi2_perp_new = val*np.sqrt(1-self.s2z**2)  # R=Rbar*(1-z^2)^0.5 
            self.s2x = chi2_perp_new*np.cos(phi2)
            self.s2y = chi2_perp_new*np.sin(phi2)
            return self
        if p == 'chi2_perp_u':
#            chi1_perp = np.sqrt(self.s1x**2+self.s2y**2)
            phi2 = np.arctan2(self.s2y, self.s2x)
            Rb = np.power(val, 1./p_R)
            chi2_perp_new = Rb*np.sqrt(1-self.s2z**2)  # R=Rbar*(1-z^2)^0.5 
            self.s2x = chi2_perp_new*np.cos(phi2)
            self.s2y = chi2_perp_new*np.sin(phi2)
            return self
        if p == 'lambda_plus':
            # Designed to give the benefits of sampling in chi_eff, without introducing a transformation/prior that depends on mass
            # Fixes chiz_minus by construction
            czm = (self.lambda1-self.lambda2)/2.
            czp = val
            self.lambda1 = (czp+czm)
            self.lambda2 = (czp-czm)
            return self
        if p == 'lambda_minus':
            # Designed to give the benefits of sampling in chi_eff, without introducing a transformation/prior that depends on mass
            # Fixes chiz_plus by construction
            czm =  val
            czp = (self.lambda1+self.lambda2)/2.
            self.lambda1 = (czp+czm)
            self.lambda2 = (czp-czm)
            return self
        if p == 'chi1':
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi1VecMag = np.sqrt(np.dot(chi1Vec,chi1Vec))
            if chi1VecMag < 1e-5:
                Lref = self.OrbitalAngularMomentumAtReferenceOverM2()
                Lhat = Lref/np.sqrt(np.dot(Lref,Lref))
                self.s1x,self.s1y,self.s1z = val*Lhat
            else:
                self.s1x,self.s1y,self.s1z = val* chi1Vec/chi1VecMag
            return self
        if p == 'chi2':
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            chi2VecMag = np.sqrt(np.dot(chi2Vec,chi2Vec))
            if chi2VecMag < 1e-5:
                Lref = self.OrbitalAngularMomentumAtReferenceOverM2()
                Lhat = Lref/np.sqrt(np.dot(Lref,Lref))
                self.s2x,self.s2y,self.s2z = val*Lhat
            else:
                self.s2x,self.s2y,self.s2z = val* chi2Vec/chi2VecMag
            return self
        if p == 'thetaJN':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            self.init_via_system_frame(thetaJN=val,phiJL=phiJL,theta1=theta1,theta2=theta2,phi12=phi12,chi1=chi1,chi2=chi2,psiJ=psiJ)
            return self
        if p == 'phiJL':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            self.init_via_system_frame(thetaJN=thetaJN,phiJL=val,theta1=theta1,theta2=theta2,phi12=phi12,chi1=chi1,chi2=chi2,psiJ=psiJ)
            return self
        # if p == 'chi1':
        #     chi1_vec_now = np.array([self.s1x,self.s1y,self.s1z])
        #     chi1_now = np.sqrt(np.dot(chi1_vec_now,chi1_vec_now))
        #     if chi1_now < 1e-5:
        #         self.s1z = val  # assume aligned
        #         return self
        #     self.s1x,self.s1y,self.s1z = chi1_vec_now * val/chi1_now
        #     return self
        if p == 'theta1_Jfix':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            self.init_via_system_frame(thetaJN=thetaJN,phiJL=phiJL,theta1=val,theta2=theta2,phi12=phi12,chi1=chi1,chi2=chi2,psiJ=psiJ)
            return self
        if p == 'theta1':
            # Do it MANUALLY, assuming the L frame! 
            # Implementation avoids calling 'system_frame' transformations needlessly
            chiperp_vec_now = np.array([self.s1x,self.s1y])
            chiperp_now = np.sqrt(np.dot(chiperp_vec_now,chiperp_vec_now))
            chi_now = np.sqrt(self.s1z**2 + chiperp_now**2)
            if chiperp_now/chi_now < 1e-9: # aligned case - what do we do?
                self.s1y=0
                self.s1x = chi_now * np.sin(val)
                self.s1z = chi_now * np.cos(val)
                return self
            self.s1x = chi_now*np.sin(val) * self.s1x/chiperp_now
            self.s1y = chi_now*np.sin(val) * self.s1y/chiperp_now
            self.s1z = chi_now*np.cos(val)
            return self
        if p == 'cos_theta1':
           self.assign_param('theta1',np.arccos(val))
           return self
        if p == 'phi1':
            # if self.fref == 0:
            #     print(" Changing geometry requires a reference frequency ")
            #     sys.exit(1)
            # Do it MANUALLY, assuming the L frame! 
            chiperp_vec_now = np.array([self.s1x,self.s1y])
            chiperp_now = np.sqrt(np.dot(chiperp_vec_now,chiperp_vec_now))
            self.s1x = chiperp_now*np.cos(val)
            self.s1y = chiperp_now*np.sin(val)
            return self
        if p == 'theta2_Jfix':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            self.init_via_system_frame(thetaJN=thetaJN,phiJL=phiJL,theta1=theta1,theta2=val,phi12=phi12,chi1=chi1,chi2=chi2,psiJ=psiJ)
            return self
        if p == 'theta2':
            # Do it MANUALLY, assuming the L frame! 
            # Implementation avoids calling 'system_frame' transformations needlessly
            chiperp_vec_now = np.array([self.s2x,self.s2y])
            chiperp_now = np.sqrt(np.dot(chiperp_vec_now,chiperp_vec_now))
            chi_now = np.sqrt(self.s2z**2 + chiperp_now**2)
            if chiperp_now/chi_now < 1e-9: # aligned case
                self.s2y=0
                self.s2x = chi_now * np.sin(val)
                self.s2z = chi_now * np.cos(val)
                return self
            self.s2x = chi_now*np.sin(val) * self.s2x/chiperp_now
            self.s2y = chi_now*np.sin(val) * self.s2y/chiperp_now
            self.s2z = chi_now*np.cos(val)
            return self
        if p == 'cos_theta2':
           self.assign_param('theta2',np.arccos(val))
           return self
        if p == 'phi2':
            # if self.fref == 0:
            #     print(" Changing geometry requires a reference frequency ")
            #     sys.exit(1)
            # Do it MANUALLY, assuming the L frame! 
            chiperp_vec_now = np.array([self.s2x,self.s2y])
            chiperp_now = np.sqrt(np.dot(chiperp_vec_now,chiperp_vec_now))
            self.s2x = chiperp_now*np.cos(val)
            self.s2y = chiperp_now*np.sin(val)
            return self
        if p == 'psiJ':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            self.init_via_system_frame(thetaJN=thetaJN,phiJL=phiJL,theta1=theta1,theta2=theta2,phi12=phi12,chi1=chi1,chi2=chi2,psiJ=val)
            return self
        if p == 'beta':
            # Documentation: *changing* beta is designed for a single-spin binary at present
            # Based on expressions in this paper
            #    http://adsabs.harvard.edu/abs/2012PhRvD..86f4020B
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            if chi2 > 1e-5:
                print(" Changing beta only supported for single spin ")
                sys.exit(2)
            if np.abs(val)<1e-4:
                theta1=0
                self.init_via_system_frame(thetaJN=val,phiJL=phiJL,theta1=theta1,theta2=theta2,phi12=phi12,chi1=chi1,chi2=chi2,psiJ=psiJ)
                return self
            #kappa = np.cos(theta1) # target we want to determine
            SoverL = chi1 * self.VelocityAtFrequency(self.fref) * self.m1/self.m2  # S/L = m1 chi v/m2
            # sanity check this value of beta is *possible*
            if  val and np.cos(val)**2 < 1-SoverL**2:
                print(" This value of beta cannot be attained, because SoverL= ", SoverL, " so beta max = ", np.arccos(np.sqrt(1-SoverL**2)))
                sys.exit(3)
            x=np.cos(val)
            kappa = (- np.sin(val)**2 + x * np.sqrt( SoverL**2 - np.sin(val)**2))/SoverL
            # Note that if gamma < 1, there are two roots for each value of beta
            # def solveme(x):
            #     return (1+ x *SoverL)/np.sqrt(1+2*x*SoverL+SoverL**2) -val # using a root find instead of algebraic for readability
            # kappa = optimize.newton(solveme,0.01)
            # PROBLEM: Only implemented for radiation gauge, disable if in non-radiation gauge
            if spin_convention == "radiation":
                theta1 = np.arccos(kappa)
                self.init_via_system_frame(thetaJN=val,phiJL=phiJL,theta1=theta1,theta2=theta2,phi12=phi12,chi1=chi1,chi2=chi2,psiJ=psiJ)
            else:
                print(" beta assignment not implemented in non-radiation gauge ")
                sys.exit(4)
            return self
        # tidal parameters
        if p == 'LambdaTilde':
            Lt, dLt   = tidal_lambda_tilde(self.m1, self.m2, self.lambda1, self.lambda2)
            Lt = val
            self.lambda1, self.lambda2 = tidal_lambda_from_tilde(self.m1, self.m2, Lt, dLt)
            return self
        if p == 'DeltaLambdaTilde':
            Lt, dLt   = tidal_lambda_tilde(self.m1, self.m2, self.lambda1, self.lambda2)
            dLt = val
            self.lambda1, self.lambda2 = tidal_lambda_from_tilde(self.m1, self.m2, Lt, dLt)
            return self
        if p == 'chieff_aligned':
            chieff = self.extract_param('xi')
            chieff_new  = val
            mtot = self.extract_param('mtot')
            m1 = self.extract_param('m1')
            m2 = self.extract_param('m2')
            if np.abs(m1/m2 -1) > 1e-5:
            # only works for aligned system. Note aligned convention changes from iteration to iteration -- will eventually be aligned with z, but..
            # This assumes chiminus is a good and useful parameter.  
            # This parameterization was designed to be more stable for extreme mass ratio systems, where the usual chi- becomes highly degenerate
            # Note that by construction, in the limit of large mass ratio, chi1 -> xi and chi2 -> - chiminus, so the parameterization is not degenerate
            # Note we have chosen an exchange-antisymmetric combination (!), changing from prior use
            # Note that this transformation can as a side effect change spins to be > 1
# {(m1 c1 +     m2 c2 ) == (m1 + m2) \[Xi], (m2 c1 - m1 c2)/(m1 +      m2) == \[Chi]Diff}
# {c1, c2} /. Solve[%, {c1, c2}][[1]] // Simplify
# Collect[%[[2]], {\[Xi], \[Chi]Diff}]
# % - \[Xi] // Simplify // FullSimplify
                chi1 = self.s1z
                chi2 = self.s2z
                chiminus =  (m2*chi1 - m1*chi2)/(m1 + m2)  # change to be consistent with the 'chiMinus' use elsewhere, avoid extremes
                # Solve equations for M xi, M chiminus.  
                chi1_new = (m1+m2)/(m1**2+m2**2) * (m1*chieff_new + m2*chiminus)
                chi2_new = (m1+m2)/(m1**2+m2**2) * (m2*chieff_new + m1*chiminus)
#                chi1_new = chieff_new + (m1 - m2)*m2*(chieff_new + chiminus)/(m1**2 + m2**2)
#                chi2_new = chieff_new - (m1 - m2)*m1*(chieff_new + chiminus)/(m1**2 + m2**2)
                self.s1z = chi1_new
                self.s2z = chi2_new
                #self.assign_param('chi1', chi1_new)
                #self.assign_param('chi2', chi2_new)
            else:
                # for equal mass, require them to have the same value
                self.assign_param('chi1', chieff_new)
                self.assign_param('chi2', chieff_new)                
            return self
        # Soichiro's coordinates : mu1, mu2, q_mu, chi2z_mu
        if p == 'mu1':
            # Sometimes we use fits where we are NOT in solar mass units, so be careful!
            fac_scale = 1
            if self.m1 > 1e10:
                fac_scale = lal.MSUN_SI
            mc = mchirp(self.m1/fac_scale,self.m2/fac_scale)
            mu1,mu2,mu3 = tools.Mcqchi1chi2Tomu1mu2mu3(mc, self.m2/self.m1, self.s1z, self.s2z)
            mcNew,q,chi1z,chi2z = tools.mu1mu2qchi2ToMcqchi1chi2(val, mu2,self.m2/self.m1, self.s2z) # q,chi2z is fixed
            self.s1z = chi1z
            # now change mc at fixed q
            self.assign_param('mc',mcNew*fac_scale)
            return self
        if p == 'mu2':
            fac_scale=1
            if self.m1 > 1e10:
                fac_scale = lal.MSUN_SI
            mc = mchirp(self.m1/fac_scale,self.m2/fac_scale)
            mu1,mu2,mu3 = tools.Mcqchi1chi2Tomu1mu2mu3(mc, self.m2/self.m1, self.s1z, self.s2z)
            mcNew,q,chi1z,chi2z = tools.mu1mu2qchi2ToMcqchi1chi2(mu1, val,self.m2/self.m1, self.s2z) # q,chi2z is fixed
            self.s1z = chi1z
            # now change mc at fixed q
            self.assign_param('mc',mcNew*fac_scale)
            return self
        if p == 'chi1z_mu':
            raise("Not implemented yet")
        if p == 'eccentricity_squared':
            self.eccentricity = np.sqrt(val)  # value is eccentricity squared
            return self
        # assign an attribute
        if hasattr(self,p):
            setattr(self,p,val)
            return self
        print(" No attribute ", p, " in ", dir(self))
        print(" Is in valid_params? ", p in valid_params)
        sys.exit(1)
    def extract_param(self,p):
        """
        assign_param
        Syntatic sugar to extract parameters to values.
        Necessary to make full use of assign_param on 
            - mchirp, eta
            - system frame parameters
        VERY HELPFUL if you want to change just one parameter at a time (e.g., for Fisher )
        """
        if p == 'mtot':
            return (self.m2+self.m1)
        if p == 'q':
            return self.m2/self.m1
        if p == 'delta' or p=='delta_mc':  # Same access routine
            return (self.m1-self.m2)/(self.m1+self.m2)
        if p == 'mc':
            return mchirp(self.m1,self.m2)
        if p == 'mc_ecc':
            # defined Favata et al 2108.05861  see Eq. 1.1
            # This is defined at periapstron; check to make sure your e's are defined corretly!
            return mchirp(self.m1,self.m2)/np.power( 1 - 157*self.eccentricity**2/24., 3./5.)
        if p == 'log_mc':
            return np.log10(mchirp(self.m1,self.m2))
        if p == 'eta':
            return symRatio(self.m1,self.m2)
        if p == 'chi1':
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            return np.sqrt(np.dot(chi1Vec,chi1Vec))
        if p == 'chi2':
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            return np.sqrt(np.dot(chi2Vec,chi2Vec))
        if p == 'chi1_perp':
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD convention for spins!
            return np.sqrt( np.dot(chi1Vec,chi1Vec) -  np.dot(Lhat, chi1Vec)**2 )  # L frame !
        if p == 'chi1_perp_bar':
            # not supporting old L convention
            chi1_perp = np.sqrt( self.s1x**2 + self.s1y**2)
            return  chi1_perp/np.sqrt(1-self.s1z**2)   # R = Rbar*sqrt(1-z**2)
        if p == 'chi1_perp_u':
            # not supporting old L convention
            chi1_perp = np.sqrt( self.s1x**2 + self.s1y**2)
            return  np.power(chi1_perp/np.sqrt(1-self.s1z**2), p_R)
        if p == 'chi2_perp':
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD convention for spins!
            return np.sqrt( np.dot(chi2Vec,chi2Vec) -  np.dot(Lhat, chi2Vec)**2 )  # L frame !
        if p == 'chi2_perp_bar':
            # not supporting old non-L convention
            chi2_perp = np.sqrt( self.s2x**2 + self.s2y**2)
            return  chi2_perp/np.sqrt(1-self.s2z**2)
        if p == 'chi2_perp_u':
            # not supporting old non-L convention
            chi2_perp = np.sqrt( self.s2x**2 + self.s2y**2)
            return  np.power(chi2_perp/np.sqrt(1-self.s2z**2), p_R)
        if p == 's1z_bar':
            return self.s1z
        if p == 's2z_bar':
            return self.s2z

        if p == 'xi' or p == 'chieff_aligned':
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            Lhat = None
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD convention for spins!
            xi = np.dot(Lhat, (self.m1*chi1Vec + self.m2* chi2Vec))/(self.m1+self.m2)   # see also 'Xi', defined below
            return xi
        if p == 'chiMinus':
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            Lhat = None
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD convention for spins!
            xi = np.dot(Lhat, (self.m1*chi1Vec - self.m2* chi2Vec))/(self.m1+self.m2)   # see also 'Xi', defined below
            return xi
        if p == 'chiz_plus':
            # Designed to give the benefits of sampling in chi_eff, without introducing a transformation/prior that depends on mass
            return (self.s1z+self.s2z)/2.
        if p == 'chiz_minus':
            # Designed to give the benefits of sampling in chi_eff, without introducing a transformation/prior that depends on mass
            return (self.s1z-self.s2z)/2.
        if p == 'shu':
            # https://arxiv.org/pdf/1801.08162.pdf
            # Eq. 27
            # Shu/M^2 =  xi - 1/2 L.(S1/q + q S2)/M^2 = xi  - 1/2 L.(m1 m2 chi1 + m1 m2 chi2) = xi - 1/2 eta(chi1+chi2).L
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            Lhat = None
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD convention for spins!
            xi = np.dot(Lhat, (self.m1*chi1Vec + self.m2* chi2Vec))/(self.m1+self.m2)   # see also 'Xi', defined below
            shu = xi - 0.5*np.dot(Lhat, chi1Vec+chi2Vec) * (self.m1*self.m2)/ (self.m1+self.m2)**2
            return shu
        # Soichiro's coordinates : mu1, mu2, q_mu, chi2z_mu
        if p == 'mu1':
            fac_scale = 1
            if self.m1 > 1e10:
                fac_scale = lal.MSUN_SI
            mc = mchirp(self.m1,self.m2)/fac_scale
            mu1,mu2,mu3 = tools.Mcqchi1chi2Tomu1mu2mu3(mc, self.m2/self.m1, self.s1z, self.s2z)
            return mu1
        if p == 'mu2':
            fac_scale = 1
            if self.m1 > 1e10:
                fac_scale = lal.MSUN_SI
            mc = mchirp(self.m1,self.m2)/fac_scale
            mu1,mu2,mu3 = tools.Mcqchi1chi2Tomu1mu2mu3(mc, self.m2/self.m1, self.s1z, self.s2z)
            return mu2
        if p == 'q_mu':   # trivial, more important what is treated as constant
            return self.m2/self.m1  
        if p == 'chi2z_mu':
            return self.s2z
        if p == 'lambda_plus':
            # Designed to give the benefits of sampling in chi_eff, without introducing a transformation/prior that depends on mass
            return (self.lambda1+self.lambda2)/2.
        if p == 'lambda_minus':
            # Designed to give the benefits of sampling in chi_eff, without introducing a transformation/prior that depends on mass
            return (self.lambda1-self.lambda2)/2.
        if p == 'chiMinusAlt':
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            Lhat = None
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD convention for spins!
            xi = np.dot(Lhat, (self.m1*chi1Vec - self.m2* chi2Vec))/(self.m1- self.m2)   # see also 'Xi', defined below
            return xi
        if p == 'thetaJN':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return thetaJN
        if p == 'phiJL':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return phiJL
        if p == 'theta1_Jfix':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return theta1
        if p == 'theta2_Jfix':
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return theta2
        if p == 'theta1':
            chiperp_vec_now = np.array([self.s1x,self.s1y])
            chiperp_now = np.sqrt(np.dot(chiperp_vec_now,chiperp_vec_now))
            chi_now = np.sqrt(self.s1z**2 + chiperp_now**2)
            return np.arccos(self.s1z/chi_now)
        if p == 'theta2':
            chiperp_vec_now = np.array([self.s2x,self.s2y])
            chiperp_now = np.sqrt(np.dot(chiperp_vec_now,chiperp_vec_now))
            chi_now = np.sqrt(self.s2z**2 + chiperp_now**2)
            return np.arccos(self.s2z/chi_now)
        if p == 'cos_theta1':
            return np.cos(self.extract_param('theta1'))
        if p == 'cos_theta2':
            return np.cos(self.extract_param('theta2'))
        if p == 'cos_theta1_alt': # alternate implementation, using lalsuite call. Not needed for theta1, theta2
            if self.fref == 0:
                print(" Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return np.cos(theta1)
        if p == 'cos_theta2_alt': # alternate implementation. Not needed for theta1, theta2
            if self.fref == 0:
                print( " Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return np.cos(theta2)
        if p == 'phi1':  # Assumes L frame!
            zfac =self.s1x + 1j*self.s1y
            phi1 =  np.angle(zfac)
            return phi1
        if p == 'phi2':  # Assumes L frame!
            zfac =self.s2x + 1j*self.s2y
            phi2 =  np.angle(zfac)
            return phi2
        if p == 'psiJ':
            if self.fref == 0:
                print( " Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return psiJ
        if p == 'sin_phiJL':
            if self.fref == 0:
                print( " Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return np.sin(phiJL)
        if p == 'cos_phiJL':
            if self.fref == 0:
                print( " Changing geometry requires a reference frequency ")
                sys.exit(1)
            thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame()
            return np.cos(phiJL)
        if p == 'beta':
            if self.fref == 0:
                print( " Changing geometry requires a reference frequency ")
                sys.exit(1)
            Jref = self.TotalAngularMomentumAtReferenceOverM2()
            Jhat = Jref/np.sqrt(np.dot(Jref, Jref))
            Lref = self.OrbitalAngularMomentumAtReferenceOverM2()
            Lhat = Lref/np.sqrt(np.dot(Lref,Lref))
            return np.arccos(np.dot(Lhat,Jhat))   # holds in general
        if p == 'cos_beta':
            if self.fref == 0:
                print( " Changing geometry requires a reference frequency ")
                sys.exit(1)
            Jref = self.TotalAngularMomentumAtReferenceOverM2()
            Jhat = Jref/np.sqrt(np.dot(Jref, Jref))
            Lref = self.OrbitalAngularMomentumAtReferenceOverM2()
            Lhat = Lref/np.sqrt(np.dot(Lref,Lref))
            return np.dot(Lhat,Jhat)   # holds in general
        if p == 'sin_beta':
            if self.fref == 0:
                print( " Changing geometry requires a reference frequency ")
                sys.exit(1)
            Jref = self.TotalAngularMomentumAtReferenceOverM2()
            Jhat = Jref/np.sqrt(np.dot(Jref, Jref))
            Lref = self.OrbitalAngularMomentumAtReferenceOverM2()
            Lhat = Lref/np.sqrt(np.dot(Lref,Lref))
            return np.sqrt(1- np.dot(Lhat,Jhat)**2)   # holds in general
        # Other spin parameters of use in generalised fits
        if p == 'SoverM2':   # SCALAR
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            S = (chi1Vec*self.m1**2+chi2Vec*self.m2**2)/(self.m1+self.m2)**2
            return np.sqrt(np.dot(S,S))
        if p == 'SOverM2_vec':  
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            S = (chi1Vec*self.m1**2+chi2Vec*self.m2**2)/(self.m1+self.m2)**2
            return S
        if p == 'SOverM2_perp':  
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            S = (chi1Vec*self.m1**2+chi2Vec*self.m2**2)/(self.m1+self.m2)**2
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD conventi
            return  np.sqrt(np.dot(S,S) - np.dot(S,Lhat)**2  )
        if p == 'DeltaOverM2_vec':  
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            Delta = -1*(chi1Vec*self.m1-chi2Vec*self.m2)/(self.m1+self.m2)
            return Delta  # VECTOR
        if p == 'DeltaOverM2_perp':  
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            Delta = -1*(chi1Vec*self.m1-chi2Vec*self.m2)/(self.m1+self.m2)
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD conventi
            return  np.sqrt(np.dot(Delta,Delta) - np.dot(Delta,Lhat)**2  )
        if p == 'DeltaOverM2_L':  
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            Delta = -1*(chi1Vec*self.m1-chi2Vec*self.m2)/(self.m1+self.m2)
            if spin_convention == "L":
                Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
            else:
                Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD conventi
            return  np.dot(Delta,Lhat)
        if p == 'S0_vec':  
            chi1Vec = np.array([self.s1x,self.s1y,self.s1z])
            chi2Vec = np.array([self.s2x,self.s2y,self.s2z])
            S0 = (chi1Vec*self.m1+chi2Vec*self.m2)/(self.m1+self.m2)
            return S0  # VECTOR
        if p == 'chi_p':
            # see e.g.,https://journals.aps.org/prd/pdf/10.1103/PhysRevD.91.024043 Eq. 3.3, 3.4
            # Reviewed implementation  (note horrible shift in names for A1, A2 !)
            #   https://git.ligo.org/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/bayespputils.py#L3783
            mtot = self.extract_param('mtot')
            m1 = self.extract_param('m1')
            m2 = self.extract_param('m2')
            chi1 = np.array([self.s1x, self.s1y, self.s1z])
            chi2 = np.array([self.s2x, self.s2y, self.s2z])
            q = m2/m1  # note convention
            A1 = (2+ 3.*q/2); A2 = (2+3./(2*q))
            S1p = (m1**2 * chi1)[:2]
            S2p = (m2**2 * chi2)[:2]
            Sp = np.max([np.linalg.norm( A1*S1p), np.linalg.norm(A2*S2p)])
            return Sp/(A1*m1**2)  # divide by term for *larger* BH
        if p == 'chi_pavg':
            if (abs(self.s1x) < 1e-4 and abs(self.s1y) < 1e-4 and abs(self.s2x) < 1e-4 and abs(self.s2y) < 1e-4):
                chipavg = 0.0
            elif (abs(self.s1x) < 1e-4 and abs(self.s1y) < 1e-4 and abs(self.s1z) < 1e-4) or (abs(self.s2x) < 1e-4 and abs(self.s2y) < 1e-4 and abs(self.s2z) < 1e-4):
                chipavg = self.extract_param('chi_p')
            else:
                try:
                    # implementation of the averaged precession parameter from https://arxiv.org/pdf/2011.11948.pdf
                    # CH 21
                    ##########################################
                    G_SI = 6.67e-11 #SI units [m^3 kg^-1 s^-2]
                    c_SI = 2.99e8 #SI units [m s^-1]
                    SM = 1.98847e30 #Solar Mass in [kg]
                    conv = SM*G_SI/c_SI**3 #converts [SM] to [s kg^-1]
                    if self.extract_param('m1') > 1e29:
                        m1 = conv*self.extract_param('m1')/SM 
                        m2 = conv*self.extract_param('m2')/SM
                    else:
                        m1 = conv*self.extract_param('m1')
                        m2 = conv*self.extract_param('m2')
                    q = m2/m1 #mass ratio
                    thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2,psiJ = self.extract_system_frame() #inheriting values from system
                    deltaphi = phi12
                    #implementation
                    if self.fref == 0:
                        fref = 20 #default fixed frequency value
                    else:
                        fref = self.fref #user spec
                    def ftor_PN(f, q, chi1, chi2, theta1, theta2, deltaphi):
                        '''Convert GW frequency to PN orbital separation conversion'''
                        om = np.pi * f 
                        M_sec = m1 + m2
                        mom = M_sec * om                
                        eta = m1*m2
                        ct1 = np.cos(theta1)
                        ct2 = np.cos(theta2)
                        ct12 = np.sin(theta1) * np.sin(theta2) * np.cos(deltaphi) + ct1 * ct2
                        # Eq. 4.13, Kidder 1995. gr-qc/9506022
                        r = (mom)**(-2./3.)*(1. \
                                        - (1./3.)*(3.-eta)*mom**(2./3.)  \
                                        - (1./3.)* ( chi1*ct1*(2.*m1**2.+3.*eta) + chi2*ct2*(2.*m2**2.+3.*eta))*mom \
                                        + ( eta*(19./4. + eta/9.) -eta*chi1*chi2/2. * (ct12 - 3.*ct1*ct2 ))*mom**(4./3.)\
                                        )
                        return r
                    def omegatilde(q):
                        '''Ratio between the spin frequency, leading order term. Eq (13)'''
                        return q*(4*q+3)/(4+3*q)
                    def chip_terms(q,chi1,chi2,theta1,theta2):
                        '''Two chip terms'''
                        term1 = chi1*np.sin(theta1)
                        term2 = omegatilde(q) * chi2*np.sin(theta2)
                        return term1,term2
                    def chip_generalized(q,chi1,chi2,theta1,theta2,deltaphi):
                        '''Generalized definition of chip. Eq (15)'''
                        term1, term2 = chip_terms(q,chi1,chi2,theta1,theta2)
                        return (term1**2 + term2**2 + 2*term1*term2*np.cos(deltaphi))**0.5
                    @np.vectorize
                    def chip_averaged(q,chi1,chi2,theta1,theta2,deltaphi,r=None,fref=None):
                        '''Averaged definition of chip. Eq (15) and Appendix A'''
                        # Convert frequency to separation, if necessary
                        if r is None and fref is None: raise ValueError
                        elif r is not None and fref is not None: raise ValueError
                        if r is None:
                            # Eq A1
                            r = ftor_PN(fref, q, chi1, chi2, theta1, theta2, deltaphi)
                        #Compute constants of motion
                        L = (r**0.5)*q/(1+q)**2
                        S1 = chi1/(1.+q)**2
                        S2 = (q**2)*chi2/(1.+q)**2
                        # Eq 5
                        chieff = (chi1*np.cos(theta1)+q*chi2*np.cos(theta2))/(1+q)
                        # Eq A2
                        J = (L**2 + S1**2 + S2**2 + 2*L*(S1*np.cos(theta1) + S2*np.cos(theta2)) + 2*S1*S2*(np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi) + np.cos(theta1)*np.cos(theta2)))**0.5
                        # Solve dSdt=0. Details in arXiv:1506.03492
                        Sminus,Splus=precession.Sb_limits(chieff,J,q,S1,S2,r)
                        def integrand_numerator(S):
                            '''chip(S)/dSdt(S)'''
                            #Eq A3
                            theta1ofS = np.arccos( (1/(2*(1-q)*S1)) * ( (J**2-L**2-S**2)/L - 2*q*chieff/(1+q) ) )
                            #Eq A4
                            theta2ofS = np.arccos( (q/(2*(1-q)*S2)) * ( -(J**2-L**2-S**2)/L + 2*chieff/(1+q) ) )
                            # Eq A5
                            deltaphiofS = np.arccos( ( S**2-S1**2-S2**2 - 2*S1*S2*np.cos(theta1ofS)*np.cos(theta2ofS) ) / (2*S1*S2*np.sin(theta1ofS)*np.sin(theta2ofS)) )
                            # Eq A6 (prefactor cancels out)
                            dSdtofS = np.sin(theta1ofS)*np.sin(theta2ofS)*np.sin(deltaphiofS)/S
                            # Eq (15)
                            chipofS = chip_generalized(q,chi1,chi2,theta1ofS,theta2ofS,deltaphiofS)
                            return chipofS/dSdtofS
                        numerator = scipy.integrate.quad(integrand_numerator, Sminus, Splus)[0]
                        def integrand_denominator(S):
                            '''1/dSdt(S)'''
                            #Eq A3
                            theta1ofS = np.arccos( (1/(2*(1-q)*S1)) * ( (J**2-L**2-S**2)/L - 2*q*chieff/(1+q) ) )
                            #Eq A4
                            theta2ofS = np.arccos( (q/(2*(1-q)*S2)) * ( -(J**2-L**2-S**2)/L + 2*chieff/(1+q) ) )
                            # Eq A5
                            deltaphiofS = np.arccos( ( S**2-S1**2-S2**2 - 2*S1*S2*np.cos(theta1ofS)*np.cos(theta2ofS) ) / (2*S1*S2*np.sin(theta1ofS)*np.sin(theta2ofS)) )
                            # Eq A6 (prefactor cancels out)
                            dSdtofS = np.sin(theta1ofS)*np.sin(theta2ofS)*np.sin(deltaphiofS)/S
                            return 1/dSdtofS
                        denominator = scipy.integrate.quad(integrand_denominator, Sminus, Splus)[0]
                        return numerator/denominator
                    chipavg = chip_averaged(q,chi1,chi2,theta1,theta2,deltaphi,fref=fref)
                except ZeroDivisionError:
                    chipavg = self.extract_param('chi_p')
            return chipavg
        if p == 'LambdaTilde':
            Lt, dLt   = tidal_lambda_tilde(self.m1, self.m2, self.lambda1, self.lambda2)
            return Lt
        if p == 'DeltaLambdaTilde':
            Lt, dLt   = tidal_lambda_tilde(self.m1, self.m2, self.lambda1, self.lambda2)
            return dLt
        if p == 'eccentricity_squared':
            return self.eccentricity**2
        if p == 'ecc_cos_meanPerAno':
            return self.eccentricity*np.cos(self.meanPerAno)
        if p == 'ecc_sin_meanPerAno':
            return self.eccentricity*np.sin(self.meanPerAno)
        if 'product(' in p:
            # Drop first and last characters
            a=p.replace(' ', '') # drop spaces
            a = a[:len(a)-1] # drop last
            a = a[8:]
            terms = a.split(',')
            vals = list(map(self.extract_param, terms)) # recurse to parse lower-level quantities
            return np.prod(vals)
        if 'inverse(' in p:
            # Drop first and last characters
            a=p.replace(' ', '') # drop spaces
            a = a[:len(a)-1] # drop last
            a = a[8:]
            terms = a.split(',')
            vals = list(map(self.extract_param, terms)) # recurse to parse lower-level quantities
            return np.prod(vals)
        # assign an attribute
        if hasattr(self,p):
            return getattr(self,p)
        print( " No attribute ", p, " in ", dir(self))
        sys.exit(0)


    def randomize(self,zero_spin_Q=False,aligned_spin_Q=False,volumetric_spin_prior_Q=False,default_inclination=None,default_phase=None,default_polarization=None,dMax=500.,dMin=1.,sMax=1):
        mMin = 2.   # min component mass (Msun)
        mMax = 100.  # max component mass (Msun)
        sMin = 0.   # min spin magnitude
#        sMax = 1.   # max spin magnitude
#        dMin = float(dMin)   # min distance (Mpc)
#        dMax = float(dMax) # max distance (Mpc)
        self.m1 = np.random.uniform(mMin,mMax)
        self.m2 = np.random.uniform(mMin,mMax)  # 
        self.m1, self.m2 = [np.max([self.m1,self.m2]), np.min([self.m1,self.m2])]
        self.m1 *= lsu_MSUN
        self.m2 *= lsu_MSUN
#        self.approx = lalsim.SpinTaylorT4  # need a 2-spin approximant by default!
        if default_inclination is not None:
            print( "Setting default inclination")
            self.incl = float(default_inclination)
        else:
            self.incl = np.random.uniform(0, np.pi)
        if default_phase is not None:
            print( "Setting default phase")
            self.phiref = float(default_phase)
        else:
            self.phiref = np.random.uniform(0, 2*np.pi)
        if default_polarization is not None:
            print( "Setting default polarization")
            self.psi = float(default_polarization)
        else:
            self.psi = np.random.uniform(0, 2*np.pi)  # match PE range
        if not zero_spin_Q and not aligned_spin_Q:
            s1mag = np.random.uniform(sMin,sMax)
            if volumetric_spin_prior_Q:
                s1mag = np.power(np.random.uniform(sMin**3,sMax**3),1./3.)
            s1theta = np.arccos(np.random.uniform(-1,1))
            s1phi = np.random.uniform(0,2*np.pi)
            s2mag = np.random.uniform(sMin,sMax)
            if volumetric_spin_prior_Q:
                s2mag = np.power(np.random.uniform(sMin**3,sMax**3),1./3.)
            s2theta = np.arccos(np.random.uniform(-1,1))
            s2phi = np.random.uniform(0,2*np.pi)
            self.s1x = s1mag * sin(s1theta) * cos(s1phi)
            self.s1y = s1mag * sin(s1theta) * sin(s1phi)
            self.s1z = s1mag * cos(s1theta)
            self.s2x = s2mag * sin(s2theta) * cos(s2phi)
            self.s2y = s2mag * sin(s2theta) * sin(s2phi)
            self.s2z = s2mag * cos(s2theta)
        if aligned_spin_Q:
            s1mag = np.random.uniform(-sMax,sMax)
#            s1theta = self.incl
#            s1phi = 0.
            s2mag = np.random.uniform(-sMax,sMax)
#            s2theta = self.incl
#            s2phi = 0.
            self.s1x = 0 # s1mag * sin(s1theta) * cos(s1phi)
            self.s1y = 0 # s1mag * sin(s1theta) * sin(s1phi)
            self.s1z = s1mag # s1mag * cos(s1theta)
            self.s2x = 0#s2mag * sin(s2theta) * cos(s2phi)
            self.s2y = 0 #s2mag * sin(s2theta) * sin(s2phi)
            self.s2z = s2mag #  s2mag * cos(s2theta)
        if np.isnan(s1mag):
            print( " catastrophe ")
            sys.exit(0)
        self.radec=True
        dist =  dMax*np.power(np.random.uniform( np.power(dMin/dMax,3),1), 1./3)  # rough, but it should work
        self.dist = dist*1e6 * lsu_PC
        self.lambda1 = 0.
        self.lambda2 = 0.
        self.theta = np.pi/2- np.arccos(np.random.uniform(-1,1))  #np.random.uniform(-np.pi/2,np.pi/2) # declination. Uniform in cos, but note range
        self.phi = np.random.uniform(0,2*np.pi) # right ascension
        self.psi = np.random.uniform(0,np.pi) # polarization angle
        self.deltaF = None

    def init_via_system_frame(self,thetaJN=0., phiJL=0., theta1=0., theta2=0., phi12=0., chi1=0., chi2=0.,psiJ=0.):
        """
        Populate spin directions (and psi_L) using system frame parameters.   Note this is NOT using \beta
        Example:
        P.init_via_system_frame(thetaJN=0.1, phiJL=0.1, theta1=0.1, theta2=0.1, phi12=0.1, chi1=1., chi2=1., psiJ=0.)
        """
        # Create basic parameters
#        self.incl, self.s1x,self.s1y, self.s1z, self.s2x, self.s2y, self.s2z = lalsim.SimInspiralTransformPrecessingInitialConditions(np.float(thetaJN), np.float(phiJL), np.float(theta1),np.float(theta2), np.float(phi12), np.float(chi1), chi2, self.m1, self.m2, self.fref)
        f_to_use = self.fref
        if self.fref==0:
            f_to_use = self.fmin
        try:
            self.incl, self.s1x,self.s1y, self.s1z, self.s2x, self.s2y, self.s2z = lalsim.SimInspiralTransformPrecessingNewInitialConditions(float(thetaJN), float(phiJL), float(theta1),float(theta2), float(phi12), float(chi1), chi2, self.m1, self.m2, f_to_use,self.phiref)
        except:
            # New format for this function
            self.incl, self.s1x,self.s1y, self.s1z, self.s2x, self.s2y, self.s2z = lalsim.SimInspiralTransformPrecessingNewInitialConditions(float(thetaJN), float(phiJL), float(theta1),float(theta2), float(phi12), float(chi1), chi2, self.m1, self.m2, f_to_use)
        # Define psiL via the deficit angle between Jhat in the radiation frame and the psiJ we want to achieve 
        Jref = self.TotalAngularMomentumAtReferenceOverM2()
        Jhat = Jref/np.sqrt(np.dot(Jref, Jref))
        self.psi= np.mod(psiJ  -np.arctan2(Jhat[1],Jhat[0]), 2*np.pi)   # define on [0, 2pi]
        return True

    def extract_system_frame(self,verbose=False):
        """
        Extract system frame angles. 
        Returned in precisely the format needed for use in init_via_system_frame.
        P.init_via_system_frame(P.extract_system_frame())  should leave the state unchanged.
        CHECK DONE IN : LIGO-Development-UpdatedSpinningWaveforms/KISTI-MCMC/PrecessingFisher/Python/overlap_versus_systemframe
        PROBLEM: Polarization angle isn't stable (oddly?)
        """
        M = self.m1+self.m2
        S1 = (self.m1/M)*(self.m1/M) * np.array([self.s1x,self.s1y, self.s1z])
        S2 = self.m2*self.m2 * np.array([self.s2x,self.s2y, self.s2z])/(M*M)
        Jref = self.TotalAngularMomentumAtReferenceOverM2()
        Jhat = Jref/np.sqrt(np.dot(Jref, Jref))
        Lref = self.OrbitalAngularMomentumAtReferenceOverM2()
        Lhat = Lref/np.sqrt(np.dot(Lref,Lref))
        S1hat = S1/np.sqrt(np.dot(S1,S1))
        S2hat = S2/np.sqrt(np.dot(S2,S2))


        # extract frame vectors
        frmJ = VectorToFrame(Jhat)
        hatX, hatY, hatZ = frmJ  
      
        # extract the polar angle of J, KEEPING IN MIND that the J reported above does NOT include psiL!
        psiJ = self.psi + np.arctan2(Jhat[1], Jhat[0])


        if hasattr(lalsim,'SimInspiralTransformPrecessingWvf2PE'):
            # Use predefined / default tool
            #   - this tool assumes the 'L' frame
            # See e.g., patch to LI https://git.ligo.org/lscsoft/lalsuite/commit/1f963908caa4f038532114840088b91f9b73e6ce
            try:
               thetaJN,phiJL,theta1,theta2,phi12,chi1,chi2=lalsim.SimInspiralTransformPrecessingWvf2PE(self.incl,self.s1x, self.s1y, self.s1z,self.s2x, self.s2y,self.s2z, self.m1, self.m2, float(np.max([self.fref,self.fmin])), self.phiref)
            except:
               print("Failure to convert coordinates in lalsimutils")
               self.print_params()
               sys.exit(1)  # exit, throw error
            return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, psiJ  # psiJ is not provided by the above routine alas


        # extract spin magnitudes
        chi1 = np.sqrt(np.dot(S1,S1)) * (M/self.m1)**2
        chi2 = np.sqrt(np.dot(S2,S2)) * (M/self.m2)**2

        # Extract angles relative to line of sight, assumed z axis in the radiation frame in which S1, S2 are stored
        # But theta1,theta2 (system frame angles) are relative to *Lhat*
        thetaJN= 0
        # If aligned, trivial
        if np.dot(Lhat, Jhat) > 1-1e-5:
            thetaJN = self.incl
        else:
            if spin_convention == 'radiation':
                thetaJN = np.arccos(Jhat[2])
            else:
                nhat_obs = np.array([ np.sin(self.incl)*np.cos(-self.phiref), np.sin(self.incl)*np.sin(-self.phiref), np.cos(self.incl)])  # not confirmed
                thetaJN = np.arccos(np.dot(Jhat, nhat_obs))
        if any(S1):
            theta1 = np.arccos( np.dot(S1hat,Lhat))
            if np.isnan(theta1):
                theta1 = 0
        else:
            theta1 =0.
            S1hat = Lhat
        if any(S2):
            theta2 = np.arccos( np.dot(S2hat,Lhat))
            if np.isnan(theta2):
                theta2 = 0
        else:
            theta2 = 0.
            S2hat = Lhat


        # extract polar angle of L around J
        beta = np.arccos(np.dot(Jhat, Lhat))  # not used here, but we can compute it
        expIalpha = np.dot((hatX+1j*hatY), Lhat)
        phiJL =alpha = np.real(np.log(expIalpha)/1j)
#        phiJL = phiJL +  np.pi  # phase convention, to be consistent with Evan's code used above in init_via_system_frame
        if np.isnan(phiJL):
            phiJL=0

        # compute relative angle of spins, in J frame
        # Must correctly account for treatment of case that S2 parallel to L
        expIphi1 = np.dot((hatX+1j*hatY), S1hat)
        expIphi2 = np.dot((hatX+1j*hatY), S2hat)
        if np.abs(expIphi2)< 1e-5:
            phi12 = -np.angle(expIphi1)
        else:
            phi12 = np.angle(expIphi2)- np.angle(expIphi1) # np.float(np.real(np.log(expIphi2/expIphi1)/1j))   # convert from 1-elemetn array
        if np.isnan(phi12):
            phi12 = 0

        return thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, psiJ

    def copy(self):
        """
        Create a deep copy, so copy and original can be changed separately
        This does NOT work reliably (segfault side effects possible)
        """
        return copy.deepcopy(self)

    def print_params(self,show_system_frame=False):
        """
        Print all key-value pairs belonging in the class instance
        """
        print( "This ChooseWaveformParams has the following parameter values:")
        print( "m1 =", self.m1 / lsu_MSUN, "(Msun)")
        print( "m2 =", self.m2 / lsu_MSUN, "(Msun)")
        print( "s1x =", self.s1x)
        print( "s1y =", self.s1y)
        print( "s1z =", self.s1z)
        print( "s2x =", self.s2x)
        print( "s2y =", self.s2y)
        print( "s2z =", self.s2z)
        S1vec = np.array([self.s1x,self.s1y,self.s1z])*self.m1*self.m1
        S2vec = np.array([self.s2x,self.s2y,self.s2z])*self.m2*self.m2
        qval = self.m2/self.m1
        print(   " : Vector spin products")
        print(   " : |s1|, |s2| = ", np.sqrt(vecDot([self.s1x,self.s1y,self.s1z],[self.s1x,self.s1y,self.s1z])), np.sqrt(vecDot([self.s2x,self.s2y,self.s2z],[self.s2x,self.s2y,self.s2z])))
        print(   " : s1.s2 = ",  vecDot([self.s1x,self.s1y,self.s1z],[self.s2x,self.s2y,self.s2z]))
        if spin_convention == "L":
            Lhat = np.array([0,0,1]) # CRITICAL to work with modern PE output. Argh. Must swap convention elsewhere
        else:
            Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar anogle!   Uses OLD convention for spins!
        print(   " : hat(L). s1 x s2 =  ",  vecDot( Lhat, vecCross([self.s1x,self.s1y,self.s1z],[self.s2x,self.s2y,self.s2z])))
        print(   " : hat(L).(S1(1+q)+S2(1+1/q)) = ", vecDot( Lhat, S1vec*(1+qval)  + S2vec*(1+1./qval) )/(self.m1+self.m2)/(self.m1+self.m2))
        if show_system_frame:
            thePrefix = ""
            thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, psiJ = self.extract_system_frame()
            print( thePrefix, " :+ theta_JN = ", thetaJN)
            print( thePrefix, " :+ psiJ = ", psiJ)
            print( thePrefix, " :+ phiJL=alphaJL = ", phiJL)
            print( thePrefix, " :+ chi1 = ", chi1)
            print( thePrefix, " :+ chi2 = ", chi2)
            print( thePrefix, " :+ theta1 = ", theta1)
            print( thePrefix, " :+ theta2 = ", theta2)
            print( thePrefix, " :+ phi12 = ", phi12)
            print( thePrefix, " :+ beta = ", self.extract_param('beta'))
        print( "lambda1 =", self.lambda1)
        print( "lambda2 =", self.lambda2)
        print("EOB Parameters:")
        print("a6c = ",self.a6c)
        print( "inclination =", self.incl)
        print( "distance =", self.dist / 1.e+6 / lsu_PC, "(Mpc)")
        print( "reference orbital phase =", self.phiref)
        print( "polarization angle =", self.psi)
        print( "eccentricity = ", self.eccentricity)
        print( "meanPerAno = ", self.meanPerAno)
        print("E0 / M = ", self.E0)
        print("j_hyp = ", self.p_phi0)
        print( "time of coalescence =", float(self.tref),  " [GPS sec: ",  int(self.tref), ",  GPS ns ", (self.tref - int(self.tref))*1e9, "]")
        print( "detector is:", self.detector)
        if self.radec==False:
            print( "Sky position relative to overhead detector is:")
            print( "zenith angle =", self.theta, "(radians)")
            print( "azimuth angle =", self.phi, "(radians)")
        if self.radec==True:
            print( "Sky position relative to geocenter is:")
            print( "declination =", self.theta, "(radians)")
            print( "right ascension =", self.phi, "(radians)")
        if self.radec==True:
            print( " -- derived parameters (detection-relevant) -- ")
            print( "   + 2(phi+psi) ", np.fmod(2*(self.psi+self.phiref),2*np.pi))
            print( "   + 2(phi-psi) ", np.fmod(2*(self.phiref- self.psi),2*np.pi))
            for ifo in ['H1', 'L1']:
                detector = lalsim.DetectorPrefixToLALDetector(ifo)
                print( "   +Arrival time at ", ifo, " = ", self.tref - np.round(float(self.tref))+ lal.TimeDelayFromEarthCenter(detector.location, self.phi, self.theta, self.tref), " versus int second")
        print( "starting frequency is =", self.fmin)
        print( "reference frequency is =", self.fref)
        print( "Max frequency is =", self.fmax)
        print( "time step =", self.deltaT, "(s) <==>", 1./self.deltaT,\
                "(Hz) sample rate")
        print( "freq. bin size is =", self.deltaF, "(Hz)")
        if isinstance(self.approx,str):
           print( "approximant is =", self.approx)
        else:
           print( "approximant is =", lalsim.GetStringFromApproximant(self.approx))
        print( "phase order =", self.phaseO)
        print( "amplitude order =", self.ampO)
        if self.waveFlags:
            thePrefix=""
            print( thePrefix, " :  Spin order " , lalsim.SimInspiralGetSpinOrder(self.waveFlags))
            print( thePrefix, " :  Tidal order " , lalsim.SimInspiralGetTidalOrder(self.waveFlags))
        else:
            print( "waveFlags struct is = ", self.waveFlags)
        print( "nonGRparams struct is", self.nonGRparams)
        if self.taper==lsu_TAPER_NONE:
            print( "Tapering is set to LAL_SIM_INSPIRAL_TAPER_NONE")
        elif self.taper==lsu_TAPER_START:
            print( "Tapering is set to LAL_SIM_INSPIRAL_TAPER_START")
        elif self.taper==lsu_TAPER_END:
            print( "Tapering is set to LAL_SIM_INSPIRAL_TAPER_END")
        elif self.taper==lsu_TAPER_STARTEND:
            print( "Tapering is set to LAL_SIM_INSPIRAL_TAPER_STARTEND")
        else:
            print( "Warning! Invalid value for taper:", self.taper)

    def VelocityAtFrequency(self,f):  # in units of c
        m1 = self.m1* lsu_G / lsu_C**3
        m2 = self.m2*lsu_G / lsu_C**3
        return ( (m1+m2) * lsu_PI * f)**(1./3.)
    def FrequencyAtVelocity(self,v):
        m1 = self.m1* lsu_G / lsu_C**3
        m2 = self.m2*lsu_G / lsu_C**3
        return v**3/(lsu_PI*(m1+m2))
    def OrbitalAngularMomentumAtReference(self):   # in units of kg in SI
        v = self.VelocityAtFrequency(max(self.fref,self.fmin));
        Lhat = None
        if spin_convention == "L":
            Lhat = np.array([0,0,1])
        else:
            Lhat = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar angle!
        M = (self.m1+self.m2)
        eta = symRatio(self.m1,self.m2)   # dimensionless
        eta2 = eta*eta
        eta3= eta2*eta
        eta4=eta3*eta
        delta = np.sqrt(1 - 4 * eta)
        m1_prime = (1 + delta) / 2
        m2_prime = (1 - delta) / 2
        Sl = m1_prime ** 2 * self.s1z + m2_prime ** 2 * self.s2z
        Sigmal = self.s2z * m2_prime - self.s1z * m1_prime
        # Appendix G.2 of PRD 103, 104056 (2021).
        # in units of kg in SI. L at 1PN from Kidder 1995 Eq 2.9 or 2PN from Blanchet 1310.1528 Eq. 375 (zero spin)
        # code for spin-dependent corrections: checked against/from https://github.com/dingo-gw/dingo/blob/main/dingo/gw/waveform_generator/frame_utils.py
        return Lhat*M*M*eta/v * ( 1+ (1.5 + eta/6)*v*v +  (27./8 - 19*eta/8 +eta2/24.)*(v**4)  + (7*eta3/1296 + 31*eta2/24 + (41*np.pi**2/24 - 6889/144)*eta + 135/16)*v**6 
                                  + (-55*eta4/31104 -215*eta3/1728 + (356035 / 3456 - 2255 * np.pi ** 2 / 576)*eta2 + eta*(-64*np.log(16*v**2)/3 -6455*np.pi**2/1536 - 128*lal.GAMMA/3 + 98869 / 5760) + 2835/128)*v**8
                                  + (-35 * Sl / 6 - 5 * delta * Sigmal / 2) * v ** 3
                                  + ((-77 / 8 + 427 * eta / 72) * Sl + delta * (-21 / 8 + 35 * eta / 12) * Sigmal)* v ** 5
                                  )
    def OrbitalAngularMomentumAtReferenceOverM2(self):
        L = self.OrbitalAngularMomentumAtReference()
        return L/(self.m1+self.m2)/(self.m1+self.m2)
    def TotalAngularMomentumAtReference(self):    # does NOT correct for psi polar angle, per convention
        L = self.OrbitalAngularMomentumAtReference()
        S1 = self.m1*self.m1 * np.array([self.s1x,self.s1y, self.s1z])
        S2 = self.m2*self.m2 * np.array([self.s2x,self.s2y, self.s2z])
        return L+S1+S2
    def TotalAngularMomentumAtReferenceOverM2(self):
        J = self.TotalAngularMomentumAtReference()
        return J/(self.m1+self.m2)/(self.m1+self.m2)

    def Xi(self):
        L = self.OrbitalAngularMomentumAtReferenceOverM2()
        S1 = self.m1*self.m1 * np.array([self.s1x,self.s1y, self.s1z])
        S2 = self.m2*self.m2 * np.array([self.s2x,self.s2y, self.s2z])
        S0 = (S1*(1+self.m2/self.m1) + S2*(1+self.m1/self.m2))/(self.m1+self.m2)/(self.m1+self.m2)
        return vecDot(vecUnit(L), S0)

    def HardAlignedQ(self):
        """
        Test if L,S1,S2 all parallel to z
        """
        return self.incl == 0. and self.s1y ==0. and self.s1x==0. and self.s2x==0. and self.s2y==0.

    def SoftAlignedQ(self):
        """
        Test if L,S1,S2 all parallel to *one another*
        """
        if not(spin_convention == 'radiation'):
            return np.abs(self.s1x)+np.abs(self.s2x)+np.abs(self.s1y)+np.abs(self.s2y) < 1e-5
        Lvec = self.OrbitalAngularMomentumAtReference()
        Lvec = Lvec/np.sqrt(np.dot(Lvec,Lvec))
#        Lvec = np.array( [np.sin(self.incl),0,np.cos(self.incl)])  # does NOT correct for psi polar angle!
        S1 = np.array([self.s1x,self.s1y, self.s1z])
        S2 = np.array([self.s2x,self.s2y, self.s2z])
        if np.dot(S1,S1) < 1e-5:
            S1hat = Lvec
        else:
            S1hat = S1/np.sqrt(np.dot(S1,S1))
        if np.dot(S2,S2)<1e-5:
            S2hat = Lvec
        else:
            S2hat = S2/np.sqrt(np.dot(S2,S2))
        return np.abs(np.dot(Lvec, S1hat))>0.999 and np.abs(np.dot(Lvec,S2hat))>0.999

    def copy_sim_inspiral(self, row):
        """
        Fill this ChooseWaveformParams with the fields of a
        row of a SWIG wrapped lalmetaio.SimInspiral table

        NB: SimInspiral table does not contain deltaT, deltaF, fref, fmax,
        lambda1, lambda2, waveFlags, nonGRparams, or detector fields, but
        ChooseWaveformParams does have these fields.
        This function will not alter these fields, so their values will
        be whatever values the instance previously had.
        """
        self.phiref = row.coa_phase
        self.m1 = row.mass1 * lsu_MSUN
        self.m2 = row.mass2 * lsu_MSUN
        self.s1x = row.spin1x
        self.s1y = row.spin1y
        self.s1z = row.spin1z
        self.s2x = row.spin2x
        self.s2y = row.spin2y
        self.s2z = row.spin2z
        if hasattr(row, 'f_lower'):
            self.fmin = row.f_lower
        self.dist = row.distance * lsu_PC * 1.e6
        self.incl = row.inclination
        if hasattr(row, 'amp_order'):
            self.ampO = row.amp_order
        if hasattr(row, 'waveform'):
            if not (str(row.waveform).find("Taylor") == -1 ) or ("Eccentric" in row.waveform):  # Not meaningful to have an order for EOB, etc
                self.phaseO = lalsim.GetOrderFromString(str(row.waveform))
            else:
                self.phaseO = -1
            try:
                self.approx = lalsim.GetApproximantFromString(str(row.waveform))  # this is buggy for SEOB waveforms, adding irrelevant PN terms
            except:
                self.approx = row.waveform  # direct pass strings if failure, for gwsignal
            if row.waveform == 'SEOBNRv3': 
                self.approx = lalsim.SEOBNRv3
            if row.waveform == 'SEOBNRv2':
                self.approx = lalsim.SEOBNRv2
            if row.waveform ==  'SEOBNRv4T':
                self.approx = lalTEOBv4
            if row.waveform == 'SEOBNRv4HM'   and lalSEOBNRv4HM > 0 :
                self.approx = lalSEOBNRv4HM
            if rosDebugMessagesContainer[0]:
                print( " Loaded approximant ", self.approx,  " AKA ", lalsim.GetStringFromApproximant(self.approx), " from ", row.waveform)
        self.theta = row.latitude # Declination
        self.phi = row.longitude # Right ascension
        self.radec = True # Flag to interpret (theta,phi) as (DEC,RA)
        self.psi = row.polarization
        if 'time_geocent' in dir(row):
            self.tref = row.time_geocent
        else:
            self.tref = row.geocent_end_time
            if lalmetaio_old_style or hasattr(row, 'geocent_end_time_ns'):
                self.tref += 1e-9*row.geocent_end_time_ns
        if hasattr(row, 'taper'):
            self.taper = lalsim.GetTaperFromString(str(row.taper))
        # FAKED COLUMNS (nonstandard)
        self.lambda1 = row.alpha5
        self.lambda2 = row.alpha6
        self.E0 = row.psi3
        self.p_phi0 = row.beta
        self.a6c = row.psi0
        self.eccentricity=row.alpha4
        self.meanPerAno=row.alpha
        self.snr = row.alpha3   # lnL info
        # WARNING: alpha1, alpha2 used by ILE for weights!
        if hasattr(row, 'alpha'):
            if not(row.alpha4):
                self.eos_table_index = row.alpha
            else:
                if not(row.alpha):
                    self.eos_table_index = None
        else:
            self.eos_table_index=None
    

    def create_sim_inspiral(self):
        """
        Create a sim_inspiral table from P.  *One* element from it
        """
        global rosDebugMessagesContainer
        if rosDebugMessagesContainer[0]:
            print( " --- Creating XML row for the following ---- ")
            self.print_params()
        sim_valid_cols = [ "simulation_id", "inclination", "longitude", "latitude", "polarization", "geocent_end_time",  "coa_phase", "distance", "mass1", "mass2", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z"] # ,  "alpha1", "alpha2", "alpha3"
        if True: #lalmetaio_old_style:
            sim_valid_cols += [ "geocent_end_time_ns"]
        si_table = lsctables.New(lsctables.SimInspiralTable, sim_valid_cols)
        row = si_table.RowType()
        row.simulation_id = si_table.get_next_id()
        # Set all parameters to default value of zero
        for slot in row.__slots__: setattr(row, slot, 0.)
        # Copy parameters
        row.spin1x = self.s1x
        row.spin1y = self.s1y
        row.spin1z = self.s1z
        row.spin2x = self.s2x
        row.spin2y = self.s2y
        row.spin2z = self.s2z
        row.mass1 = self.m1/lsu_MSUN
        row.mass2 = self.m2/lsu_MSUN
        row.mchirp = mchirp(row.mass1,row.mass2)
        row.longitude = self.phi
        row.latitude   = self.theta
        row.inclination = self.incl
        row.polarization = self.psi
        row.coa_phase = self.phiref
        # New code for managing times in output: see https://git.ligo.org/kipp.cannon/python-igwn_ligolw/-/blob/master/igwn_ligolw/lsctables.py
        if 'time_geocent' in dir(row):
            row.time_geocent = float(self.tref)
        # http://stackoverflow.com/questions/6032781/pythonnumpy-why-does-numpy-log-throw-an-attribute-error-if-its-operand-is-too
        elif  hasattr(row, 'geocent_time_ns'):   # old school approach, will not work now
            row.geocent_end_time = np.floor( float(self.tref))
            row.geocent_end_time_ns = np.floor( float(1e9*(self.tref - row.geocent_end_time)))
        else:
            row.geocent_end_time = float(self.tref)
        row.distance = self.dist/(1e6*lsu_PC)
        row.amp_order = self.ampO
        # PROBLEM: This line is NOT ROBUST, because of type conversions
        row.waveform = lalsim.GetStringFromApproximant(self.approx)
        if  ("Taylor" in row.waveform) or ("Eccentric" in row.waveform):   # we only have PN orders embedded in Taylor?
            row.waveform =row.waveform+lsu_StringFromPNOrder(self.phaseO)
        row.taper = "TAPER_NONE"
        row.f_lower =self.fmin
        # NONSTANDARD
        row.alpha5 = self.lambda1
        row.alpha6 = self.lambda2
        row.psi0 = self.a6c
        row.psi3 = self.E0
        row.beta = self.p_phi0
        row.alpha4 = self.eccentricity
        row.alpha = self.meanPerAno
        if self.eos_table_index and not self.eccentricity:
            row.alpha = self.eos_table_index
        if self.snr:
            row.alpha3 = self.snr
        # Debug: 
        if rosDebugMessagesContainer[0]:
            print( " Constructing the following XML table ")
            si_table.append(row)
            si_table.write()
        return row


    def copy_lsctables_sim_inspiral(self, row):
        """
        Fill this ChooseWaveformParams with the fields of a
        row of an lsctables.SimInspiral table
        (i.e. SimInspiral table in the format as read from a file)

        NB: SimInspiral table does not contain deltaT, deltaF, fref, fmax,
        lambda1, lambda2, waveFlags, nonGRparams, or detector fields, but
        ChooseWaveformParams does have these fields.
        This function will not alter these fields, so their values will
        be whatever values the instance previously had.

        Adapted from code by Chris Pankow
        """
        # Convert from lsctables.SimInspiral --> lalmetaio.SimInspiral
        swigrow = lalmetaio.SimInspiralTable()
        for simattr in lsctables.SimInspiralTable.validcolumns.keys():
            if simattr in ['process_id','simulation_id','process::process_id','process:process_id']:
                try:
                    setattr(swigrow, simattr,0)
                except:
                    True  # we don't really care if this doesn't happen
            elif simattr in ["waveform", "source", "numrel_data", "taper","process_id"]:
                # unicode -> char* doesn't work
                setattr( swigrow, simattr, str(getattr(row, simattr)) )
            elif not(lalmetaio_old_style) and ('end_time_ns' in simattr ):
                basename = simattr.replace('_ns', '')
                val = float(getattr(swigrow, basename))
                dt = float(getattr(row, simattr))
                setattr( swigrow, basename, (val+1e-9*dt))
            else:
                try:
                    setattr( swigrow, simattr, getattr(row, simattr) )
                except:
                    raise Exception(" Failed to set  attribute {}".format(simattr))
        # Call the function to read lalmetaio.SimInspiral format
        self.copy_sim_inspiral(swigrow)

    def scale_to_snr(self,new_SNR,psd, ifo_list,analyticPSD_Q=True):
        """
        scale_to_snr
          - evaluates network SNR in the ifo list provided (assuming *constant* psd for all..may change)
          - uses network SNR to rescale the distance of the source, so the SNR is now  new_SNR
          - returns current_SNR, for sanity
        """
        deltaF=findDeltaF(self)
        det_orig = self.detector
        IP = Overlap(fLow=self.fmin, fNyq=1./self.deltaT/2., deltaF=deltaF, psd=psd, full_output=True,analyticPSD_Q=analyticPSD_Q)

        rho_ifo= {}
        current_SNR_squared =0
        for det in ifo_list:
            self.detector = det
            self.radec = True
            h=hoff(self)
            rho_ifo[det] = IP.norm(h)
            current_SNR_squared +=rho_ifo[det]*rho_ifo[det]
        current_SNR = np.sqrt(current_SNR_squared)

        self.detector = det_orig
        self.dist = (current_SNR/new_SNR)*self.dist
        return current_SNR


def xml_to_ChooseWaveformParams_array(fname, minrow=None, maxrow=None,
        deltaT=1./4096., fref=0., lambda1=0., lambda2=0., waveFlags=None,
        nonGRparams=None, detector="H1", deltaF=None, fmax=0.):
    """
    Function to read an xml file 'fname' containing a SimInspiralTable,
    convert rows of the SimInspiralTable into ChooseWaveformParams instances
    and return an array of the ChooseWaveformParam instances

    Can optionally give 'minrow' and 'maxrow' to convert only rows
    in the range (starting from zero) [minrow, maxrow). If these arguments
    are not given, this function will convert the whole SimInspiral table.

    The rest of the optional arguments are the fields in ChooseWaveformParams
    that are not present in SimInspiral tables. Any of these arguments not given
    values will use the standard default values of ChooseWaveformParams.
    """
    
  
    xmldoc = utils.load_filename( fname ,contenthandler = cthdler )
    try:
        # Read SimInspiralTable from the xml file, set row bounds
        sim_insp = lsctables.SimInspiralTable.get_table(xmldoc) #table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
        length = len(sim_insp)
        if not minrow and not maxrow:
            minrow = 0
            maxrow = length
        else:
            assert minrow >= 0
            assert minrow <= maxrow
            assert maxrow <= length
        rng = range(minrow,maxrow)
        # Create a ChooseWaveformParams for each requested row
        Ps = [ChooseWaveformParams(deltaT=deltaT, fref=fref, lambda1=lambda1,
            lambda2=lambda2, waveFlags=waveFlags, nonGRparams=nonGRparams,                                   
            detector=detector, deltaF=deltaF, fmax=fmax) for i in rng]
        # Copy the information from requested rows to the ChooseWaveformParams
        [Ps[i-minrow].copy_lsctables_sim_inspiral(sim_insp[i]) for i in rng]
        # set the approximants correctly -- this is NOT straightforward because of conversions
    except ValueError:
        print( "No SimInspiral table found in xml file",file=sys.stderr)
    return Ps

#
# end-to-end XML i/o as sim_inspiral
#
def ChooseWaveformParams_array_to_xml(P_list, fname="injections", minrow=None, maxrow=None,
        deltaT=1./4096., fref=0., waveFlags=None,
        nonGRparams=None, detector="H1", deltaF=None, fMax=0.):
    """
    Standard XML storage for parameters.
    Note that lambda values are NOT stored in xml table entries --- I have a special hack to do this
    """
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    sim_table = lsctables.New(lsctables.SimInspiralTable)
    xmldoc.childNodes[0].appendChild(sim_table)
    indx =0
    for P in P_list:
        row= P.create_sim_inspiral()
        row.process_id = indx # ilwd.ilwdchar("process:process_id:{0}".format(indx))
        row.simulation_id = indx # ilwd.ilwdchar("sim_inspiral:simulation_id:{0}".format(indx))
        indx+=1
        sim_table.append(row)
    if rosDebugMessagesContainer[0]:
            print( " Preparing to write the followingXML ")
            sim_table.write()


    fname_out = fname
    if not(".xml.gz" in fname):
        fname_out = fname+".xml.gz"
    utils.write_filename(xmldoc, fname_out, compress="gz")

    return True

hdf_params = ['m1', 'm2', \
   's1x',   's1y', 's1z', 's2x', 's2y', 's2z', \
    'dist', 'incl', 'phiref', 'theta', 'phi', 'tref', 'psi', \
    'lambda1', 'lambda2', 'fref', 'fmin', \
    'lnL', 'p', 'ps']
import h5py
def ChooseWaveformParams_array_to_hdf5(P_list, fname="injections", 
        deltaT=1./4096., fref=0., waveFlags=None,
        nonGRparams=None, detector="H1", deltaF=None, fMax=0.):
    """
    HDF5 storage for parameters.
    Compare to lalinference HDF5 i/o https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/io/hdf5.py

    TERRIBLE CODE: Assumes hardcoded order, does not embed metadata with field names
    SHOULD RESTRUCTURE to load into record-array like structure
    """
    f = h5py.File(fname+".hdf5","w")

    arr = np.zeros( (len(P_list),len(hdf_params)) )
    indx = 0
    for P in P_list:
        pindex = 0
        for param in hdf_params:  # Don't store these other quantities
            if param in   ['lnL', 'p', 'ps']:
                continue
            val= P.extract_param(param); fac=1   # getattr is probably faster
            if param in ['m1','m2']:
                fac = lal.MSUN_SI
            arr[indx][pindex] = val/fac
            pindex += 1
        indx += 1

    dset = f.create_dataset("waveform_parameters", (len(P_list),len(hdf_params)), dtype='f', data=arr)  # lalinference_o2 for now            
    f.close()
    return True

def hdf5_to_ChooseWaveformParams_array(fname="injections", 
        deltaT=1./4096., fref=0., waveFlags=None,
        nonGRparams=None, detector="H1", deltaF=None, fMax=0.):
    """
    HDF5 storage for parameters.
    Compare to lalinference HDF5 i/o https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/io/hdf5.py

    TERRIBLE CODE: Assumes hardcoded order, does not embed metadata with field names
    SHOULD RESTRUCTURE to load into record-array like structure
    """
    f = h5py.File(fname+".hdf5","r")


    dset = f["waveform_parameters"]
    P_list = []
    for indx in np.arange(len(dset)):
        P = ChooseWaveformParams()
        for pindex in np.arange(len(hdf_params)-3):
            param = hdf_params[pindex]
            fac = 1;
            if param in ['m1','m2']:
                fac = lal.MSUN_SI
            P.assign_param(param, dset[indx,pindex]*fac)
        P_list.append(P)

    return P_list



#
# Classes for computing inner products of waveforms
#
class InnerProduct(object):
    """
    Base class for inner products
    """
    def __init__(self, fLow=10., fMax=None, fNyq=2048., deltaF=1./8.,
            psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower, analyticPSD_Q=True,
            inv_spec_trunc_Q=False, T_spec=0., waveform_is_psi4=False):
        self.fLow = fLow # min limit of integration
        self.fMax = fMax # max limit of integration
        self.fNyq = fNyq # max freq. in arrays whose IP will be computed
        self.deltaF = deltaF
        self.deltaT = 1./2./self.fNyq
        self.len1side = int(fNyq/deltaF)+1 # length of Hermitian arrays
        self.len2side = 2*(self.len1side-1) # length of non-Hermitian arrays
        self.weights = np.zeros(self.len1side)
        self.weights2side = np.zeros(self.len2side)
        if self.fMax is None:
            self.fMax = self.fNyq
        assert self.fMax <= self.fNyq
        self.minIdx = int(round(self.fLow/deltaF))
        self.maxIdx = int(round(self.fMax/deltaF))
        # Fill 1-sided (Herm.) weights from psd
        if analyticPSD_Q is True:
            for i in range(self.minIdx,self.maxIdx): # set weights = 1/Sn(f)
                extra_weight=1.0
                if waveform_is_psi4:
                    extra_weight=1.0/(2*np.pi*i*deltaF)/(2*np.pi*i*deltaF)
                self.weights[i] = 1./psd(i*deltaF)*extra_weight
        elif analyticPSD_Q is False:
            if isinstance(psd, lal.REAL8FrequencySeries):
                assert psd.f0 == 0. # don't want heterodyned psd
                assert abs(psd.deltaF - self.deltaF) <= TOL_DF
                fPSD = (psd.data.length - 1) * psd.deltaF # -1 b/c start at f=0
                assert self.fMax <= fPSD
                ivals = np.arange(self.minIdx,self.maxIdx)
                ivals_test_ok = psd.data.data[ivals]>0
                ivals_ok = ivals[ivals_test_ok] 
                extra_weight=np.ones(len(self.weights))
                if waveform_is_psi4:
                    extra_weight[ivals_ok] = 1./(2*np.pi*ivals[ivals_ok]*deltaF)**2
                self.weights[ivals_ok] = 1./psd.data.data[ivals_ok] * extra_weight[ivals_ok]
                # for i in range(self.minIdx,self.maxIdx):
                #     if psd.data.data[i] != 0.:
                #         extra_weight=1.0
                #         if waveform_is_psi4:
                #             extra_weight=1.0/(2*np.pi*i*deltaF)/(2*np.pi*i*deltaF)
                #         self.weights[i] = 1./psd.data.data[i]*extra_weight
            else: # if we get here psd must be an array
                fPSD = (len(psd) - 1) * self.deltaF # -1 b/c start at f=0
                assert self.fMax <= fPSD
                # ivals = np.arange(self.minIdx,self.maxIdx)
                # ivals_ok = psd[ivals]>0
                # extra_weight=np.ones(len(self.weights))
                # if waveform_is_psi4:
                #     extra_weight[ivals_ok] = 1./(2*np.pi*ivals[ivals_ok]*deltaF)**2
                # self.weights[ivals_ok] = 1./psd[ivals_ok] * extra_weight[ivals_ok]
                for i in range(self.minIdx,self.maxIdx):
                    if psd[i] != 0.:
                        extra_weight=1.0
                        if waveform_is_psi4:
                            extra_weight=1.0/(2*np.pi*i*deltaF)/(2*np.pi*i*deltaF)
                        self.weights[i] = 1./psd[i]*extra_weight
        else:
            raise ValueError("analyticPSD_Q must be either True or False")

        # Do inverse spectrum truncation if requested
        if inv_spec_trunc_Q is True and T_spec != 0.:
            N_spec = int(T_spec / self.deltaT ) # number of non-zero TD pts
            # Ensure you will have some uncorrupted region in IP time series
            assert N_spec < self.len2side / 2
            # Create workspace arrays
            WFD = lal.CreateCOMPLEX16FrequencySeries('FD root inv. spec.',
                    lal.LIGOTimeGPS(0.), 0., self.deltaF,
                    lsu_DimensionlessUnit, self.len1side)
            WTD = lal.CreateREAL8TimeSeries('TD root inv. spec.',
                    lal.LIGOTimeGPS(0.), 0., self.deltaT,
                    lsu_DimensionlessUnit, self.len2side)
            fwdplan = lal.CreateForwardREAL8FFTPlan(self.len2side, 0)
            revplan = lal.CreateReverseREAL8FFTPlan(self.len2side, 0)
            WFD.data.data[:] = np.sqrt(self.weights) # W_FD is 1/sqrt(S_n(f))
            WFD.data.data[0] = WFD.data.data[-1] = 0. # zero 0, f_Nyq bins
            lal.REAL8FreqTimeFFT(WTD, WFD, revplan) # IFFT to TD
            for i in range(int(N_spec/2), self.len2side - int(N_spec/2)):
                WTD.data.data[i] = 0. # Zero all but T_spec/2 ends of W_TD
            lal.REAL8TimeFreqFFT(WFD, WTD, fwdplan) # FFT back to FD
            WFD.data.data[0] = WFD.data.data[-1] = 0. # zero 0, f_Nyq bins
            # Square to get trunc. inv. PSD
            self.weights = np.abs(WFD.data.data*WFD.data.data)

        # Create 2-sided (non-Herm.) weights from 1-sided (Herm.) weights
        # They should be packed monotonically, e.g.
        # W(-N/2 df), ..., W(-df) W(0), W(df), ..., W( (N/2-1) df)
        # In particular,freqs = +-i*df are in N/2+-i bins of array
        self.weights2side[:len(self.weights)] = self.weights[::-1]
        self.weights2side[len(self.weights)-1:] = self.weights[0:-1]

    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        raise Exception("This is the base InnerProduct class! Use a subclass")

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        raise Exception("This is the base InnerProduct class! Use a subclass")


class RealIP(InnerProduct):
    r"""
    Real-valued inner product. self.ip(h1,h2) computes

             fNyq
    4 Re \int      h1(f) h2*(f) / Sn(f) df
             fLow

    And similarly for self.norm(h1)

    DOES NOT maximize over time or phase
    """
    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h1.data.length == self.len1side
        assert h2.data.length == self.len1side
        assert abs(h1.deltaF-h2.deltaF) <= TOL_DF\
                and abs(h1.deltaF-self.deltaF) <= TOL_DF
        val = np.sum(np.conj(h1.data.data)*h2.data.data*self.weights)
        val = 4. * self.deltaF * np.real(val)
        return val

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length == self.len1side
        assert abs(h.deltaF-self.deltaF) <= TOL_DF
        val = np.sum(np.conj(h.data.data)*h.data.data*self.weights)
        val = np.sqrt( 4. * self.deltaF * np.abs(val) )
        return val


class HermitianComplexIP(InnerProduct):
    r"""
    Complex-valued inner product. self.ip(h1,h2) computes

          fNyq
    4 \int      h1(f) h2*(f) / Sn(f) df
          fLow

    And similarly for self.norm(h1)

    N.B. Assumes h1, h2 are Hermitian - i.e. they store only positive freqs.
         with negative freqs. given by h(-f) = h*(f)
    DOES NOT maximize over time or phase
    """
    def ip(self, h1, h2):
        """
        Compute inner product between two COMPLEX16Frequency Series
        """
        assert h1.data.length == self.len1side
        assert h2.data.length == self.len1side
        assert abs(h1.deltaF-h2.deltaF) <= TOL_DF\
                and abs(h1.deltaF-self.deltaF) <= TOL_DF
        val = np.sum(np.conj(h1.data.data)*h2.data.data*self.weights)
        val *= 4. * self.deltaF
        return val

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length == self.len1side
        assert abs(h.deltaF-self.deltaF) <= TOL_DF
        val = np.sum(np.conj(h.data.data)*h.data.data*self.weights)
        val = np.sqrt( 4. * self.deltaF * np.abs(val) )
        return val


class ComplexIP(InnerProduct):
    r"""
    Complex-valued inner product. self.ip(h1,h2) computes

          fNyq
    2 \int      h1(f) h2*(f) / Sn(f) df
          -fNyq

    And similarly for self.norm(h1)

    N.B. DOES NOT assume h1, h2 are Hermitian - they should contain negative
         and positive freqs. packed as
    [ -N/2 * df, ..., -df, 0, df, ..., (N/2-1) * df ]
    DOES NOT maximize over time or phase
    """
    def ip(self, h1, h2,include_epoch_differences=False):
        """
        Compute inner product between two COMPLEX16Frequency Series
        Accounts for time shfit
        """
        assert h1.data.length==h2.data.length==self.len2side
        assert abs(h1.deltaF-h2.deltaF) <= TOL_DF\
                and abs(h1.deltaF-self.deltaF) <= TOL_DF
        val = 0.
        factor_shift = np.ones( len(h1.data.data))
        if include_epoch_differences:
            fvals = evaluate_fvals(h1)
            factor_shift = np.exp(-1j* (float(h1.epoch) - float(h2.epoch))*fvals*2*np.pi)  # exp( i omega( t_2 - t_1) )
        val = np.sum( np.conj(h1.data.data)*h2.data.data*factor_shift*self.weights2side )
        val *= 2. * self.deltaF
        return val

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length==self.len2side
        assert abs(h.deltaF-self.deltaF) <= TOL_DF
        length = h.data.length
        val = 0.
        val = np.sum( np.conj(h.data.data)*h.data.data*self.weights2side )
        val = np.sqrt( 2. * self.deltaF * np.abs(val) )
        return val

class Overlap(InnerProduct):
    r"""
    Inner product maximized over time and phase. self.ip(h1,h2) computes:

                  fNyq
    max 4 Abs \int      h1*(f,tc) h2(f) / Sn(f) df
     tc           fLow

    h1, h2 must be COMPLEX16FrequencySeries defined in [0, fNyq]
    (with the negative frequencies implicitly given by Hermitianity)

    If self.full_output==False: returns
        The maximized (real-valued, > 0) overlap
    If self.full_output==True: returns
        The maximized overlap
        The entire COMPLEX16TimeSeries of overlaps for each possible time shift
        The index of the above time series at which the maximum occurs
        The phase rotation which maximizes the real-valued overlap
    """
    def __init__(self, fLow=10., fMax=None, fNyq=2048., deltaF=1./8.,
            psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower, analyticPSD_Q=True,
            inv_spec_trunc_Q=False, T_spec=0., full_output=False):
        super(Overlap, self).__init__(fLow, fMax, fNyq, deltaF, psd,
                analyticPSD_Q, inv_spec_trunc_Q, T_spec) # Call base constructor
        self.full_output = full_output
        self.deltaT = 1./self.deltaF/self.len2side
        self.revplan = lal.CreateReverseCOMPLEX16FFTPlan(self.len2side, 0)
        self.intgd = lal.CreateCOMPLEX16FrequencySeries("SNR integrand", 
                lal.LIGOTimeGPS(0.), 0., self.deltaF, lsu_HertzUnit,
                self.len2side)
        self.ovlp = lal.CreateCOMPLEX16TimeSeries("Complex overlap", 
                lal.LIGOTimeGPS(0.), 0., self.deltaT, lsu_DimensionlessUnit,
                self.len2side)

    def ip(self, h1, h2):
        """
        Compute inner product between two Hermitian COMPLEX16Frequency Series
        """
        assert h1.data.length==h2.data.length==self.len1side
        assert abs(h1.deltaF-h2.deltaF) <= TOL_DF\
                and abs(h1.deltaF-self.deltaF) <= TOL_DF
        # Tabulate the SNR integrand
        # Set negative freqs. of integrand to zero
        self.intgd.data.data[:self.len1side] = np.zeros(self.len1side)
        # Fill positive freqs with inner product integrand
        temp = 4.*np.conj(h1.data.data) * h2.data.data * self.weights
        self.intgd.data.data[self.len1side-1:] = temp[:-1]
        # Reverse FFT to get overlap for all possible reference times
        lal.COMPLEX16FreqTimeFFT(self.ovlp, self.intgd, self.revplan)
        rhoSeries = np.abs(self.ovlp.data.data)
        rho = rhoSeries.max()
        if self.full_output==False:
            # Return overlap maximized over time, phase
            return rho
        else:
            # Return max overlap, full overlap time series and other info
            rhoIdx = rhoSeries.argmax()
            rhoPhase = np.angle(self.ovlp.data.data[rhoIdx])
            # N.B. Copy rho(t) to a new TimeSeries, so we don't return a
            # reference to the TimeSeries belonging to the class (self.ovlp),
            # which will be overwritten if its ip() method is called again later
            rhoTS = lal.CreateCOMPLEX16TimeSeries("Complex overlap",
                lal.LIGOTimeGPS(0.), 0., self.deltaT, lsu_DimensionlessUnit,
                self.len2side)
            rhoTS.data.data[:] = self.ovlp.data.data[:]
            return rho, rhoTS, rhoIdx, rhoPhase

    def norm(self, h):
        """
        Compute norm of a COMPLEX16Frequency Series
        """
        assert h.data.length == self.len1side
        assert abs(h.deltaF-self.deltaF) <= TOL_DF
        val = 0.
        val = np.sum( np.conj(h.data.data)*h.data.data *self.weights)
        val = np.sqrt( 4. * self.deltaF * np.abs(val) )
        return val

    def wrap_times(self):
        """
        Return a vector of wrap-around time offsets, i.e.
        [ 0, dt, 2 dt, ..., N dt, -(N-1) dt, -(N-1) dt, ..., -2 dt, -dt ]

        This is useful in conjunction with the 'full_output' option to plot
        the overlap vs timeshift. e.g. do:

        IP = Overlap(full_output=True)
        t = IP.wrap_times()
        rho, ovlp, rhoIdx, rhoPhase = IP.ip(h1, h2)
        plot(t, abs(ovlp))
        """
        tShift = np.arange(self.len2side) * self.deltaT
        for i in range(self.len1side,self.len2side):
            tShift[i] -= self.len2side * self.deltaT
        return tShift

class ComplexOverlap(InnerProduct):
    r"""
    Inner product maximized over time and polarization angle. 
    This inner product does not assume Hermitianity and is therefore
    valid for waveforms that are complex in the TD, e.g. h+(t) + 1j hx(t).
    self.IP(h1,h2) computes:

                  fNyq
    max 2 Abs \int      h1*(f,tc) h2(f) / Sn(f) df
     tc          -fNyq

    h1, h2 must be COMPLEX16FrequencySeries defined in [-fNyq, fNyq-deltaF]
    At least one of which should be non-Hermitian for the maximization
    over phase to work properly.

    If self.full_output==False: returns
        The maximized overlap
    If self.full_output==True: returns
        The maximized overlap
        The entire COMPLEX16TimeSeries of overlaps for each possible time shift
        The index of the above time series at which the maximum occurs
        The phase rotation which maximizes the real-valued overlap
    """
    def __init__(self, fLow=10., fMax=None, fNyq=2048., deltaF=1./8.,
            psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower, analyticPSD_Q=True,
            inv_spec_trunc_Q=False, T_spec=0., full_output=False,interpolate_max=False, waveform_is_psi4=False):
        super(ComplexOverlap, self).__init__(fLow, fMax, fNyq, deltaF, psd,
                analyticPSD_Q, inv_spec_trunc_Q, T_spec, waveform_is_psi4) # Call base constructor
        self.full_output=full_output
        self.interpolate_max=interpolate_max
        self.deltaT = 1./self.deltaF/self.len2side
        # Create FFT plan and workspace vectors
        self.revplan=lal.CreateReverseCOMPLEX16FFTPlan(self.len2side, 0)
        self.intgd = lal.CreateCOMPLEX16FrequencySeries("SNR integrand", 
                lal.LIGOTimeGPS(0.), 0., self.deltaF,
                lsu_HertzUnit, self.len2side)
        self.ovlp = lal.CreateCOMPLEX16TimeSeries("Complex overlap", 
                lal.LIGOTimeGPS(0.), 0., self.deltaT, lsu_DimensionlessUnit,
                self.len2side)

    def ip(self, h1, h2, **kwargs):
        """
        Compute inner product between two non-Hermitian COMPLEX16FrequencySeries
        """
        assert h1.data.length==h2.data.length==self.len2side
        assert abs(h1.deltaF-h2.deltaF) <= TOL_DF\
                and abs(h1.deltaF-self.deltaF) <= TOL_DF
        # Tabulate the SNR integrand
        self.intgd.data.data = 2*np.conj(h1.data.data)\
                *h2.data.data*self.weights2side
        # Reverse FFT to get overlap for all possible reference times
        lal.COMPLEX16FreqTimeFFT(self.ovlp, self.intgd, self.revplan)
        rhoSeries = np.abs(self.ovlp.data.data)
        rho = rhoSeries.max()
        if self.interpolate_max:
            # see: spokes.py and util_ManualOverlapGrid.py
            rhoIdx = rhoSeries.argmax()
            datReduced = rhoSeries[rhoIdx-2:rhoIdx+2]
            try:
                z =np.polyfit(np.arange(len(datReduced)),datReduced,2)
                if z[0]<0:
                    return z[2] - z[1]*z[1]/4/z[2]
            except:
                print( " Duration error ", datReduced, " skipping interpolation in time to best point ")
            # Otherwise, act as normally
        if self.full_output==False:
            # Return overlap maximized over time, phase
            return rho
        else:
            # Return max overlap, full overlap time series and other info
            rhoIdx = rhoSeries.argmax()
            rhoPhase = np.angle(self.ovlp.data.data[rhoIdx])
            # N.B. Copy rho(t) to a new TimeSeries, so we don't return a
            # reference to the TimeSeries belonging to the class (self.ovlp),
            # which will be overwritten if its ip() method is called again later
            rhoTS = lal.CreateCOMPLEX16TimeSeries("Complex overlap",
                lal.LIGOTimeGPS(0.), 0., self.deltaT, lsu_DimensionlessUnit,
                self.len2side)
            rhoTS.data.data[:] = self.ovlp.data.data[:]
            return rho, rhoTS, rhoIdx, rhoPhase

    def norm(self, h):
        """
        Compute norm of a non-Hermitian COMPLEX16FrequencySeries
        """
        assert h.data.length==self.len2side
        assert abs(h.deltaF-self.deltaF) <= TOL_DF
        val = np.sum( np.conj(h.data.data)*h.data.data *self.weights2side)
        val = np.sqrt( 2. * self.deltaF * np.abs(val) )
        return val

    def wrap_times(self):
        """
        Return a vector of wrap-around time offsets, i.e.
        [ 0, dt, 2 dt, ..., N dt, -(N-1) dt, -(N-1) dt, ..., -2 dt, -dt ]

        This is useful in conjunction with the 'full_output' option to plot
        the overlap vs timeshift. e.g. do:

        IP = ComplexOverlap(full_output=True)
        t = IP.wrap_times()
        rho, ovlp, rhoIdx, rhoPhase = IP.ip(h1, h2)
        plot(t, abs(ovlp))
        """
        tShift = np.arange(self.len2side) * self.deltaT
        for i in range(self.len1side,self.len2side):
            tShift[i] -= self.len2side * self.deltaT
        return tShift


def CreateCompatibleComplexOverlap(hlmf,**kwargs):
    """
    CreateCompatibleComplexOverlap: accepts dictionary or single instance of COMPLEX16FrequencySeries
    """
    if isinstance(hlmf, dict):
        modes = hlmf.keys()
        hbase = hlmf[list(modes)[0]]
    else:
        hbase =hlmf
    deltaF = hbase.deltaF
    fNyq = hbase.deltaF*hbase.data.length/2 # np.max(evaluate_fvals(hbase))
    if rosDebugMessagesContainer[0]:
        print( kwargs)
        print( "dF, fNyq, npts = ",deltaF, fNyq, len(hbase.data.data))
    IP = ComplexOverlap(fNyq=fNyq, deltaF=deltaF, **kwargs)
    return IP

def CreateCompatibleComplexIP(hlmf,**kwargs):
    """
    Creates complex IP (no maximization)
    """
    if isinstance(hlmf, dict):
        modes = hlmf.keys()
        hbase = hlmf[list(modes)[0]]
    else:
        hbase =hlmf
    deltaF = hbase.deltaF
    fNyq = hbase.deltaF*hbase.data.length/2 # np.max(evaluate_fvals(hbase))
    if rosDebugMessagesContainer[0]:
        print( kwargs)
        print( "dF, fNyq, npts = ",deltaF, fNyq, len(hbase.data.data))
    IP = ComplexIP(fNyq=fNyq, deltaF=deltaF, **kwargs)
    return IP



#
# Antenna pattern functions
#
def Fplus(theta, phi, psi):
    """
    Antenna pattern as a function of polar coordinates measured from
    directly overhead a right angle interferometer and polarization angle
    """
    return 0.5*(1. + cos(theta)*cos(theta))*cos(2.*phi)*cos(2.*psi)\
            - cos(theta)*sin(2.*phi)*sin(2.*psi)

def Fcross(theta, phi, psi):
    """
    Antenna pattern as a function of polar coordinates measured from
    directly overhead a right angle interferometer and polarization angle
    """
    return 0.5*(1. + cos(theta)*cos(theta))*cos(2.*phi)*sin(2.*psi)\
            + cos(theta)*sin(2.*phi)*cos(2.*psi)

#
# Mass parameter conversion functions - note they assume m1 >= m2
#
def mass1(Mc, eta):
    """Compute larger component mass from Mc, eta"""
    return 0.5*Mc*eta**(-3./5.)*(1. + np.sqrt(1 - 4.*eta))

def mass2(Mc, eta):
    """Compute smaller component mass from Mc, eta"""
    return 0.5*Mc*eta**(-3./5.)*(1. - np.sqrt(1 - 4.*eta))

def mchirp(m1, m2):
    """Compute chirp mass from component masses"""
    return (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)

def symRatio(m1, m2):
    """Compute symmetric mass ratio from component masses"""
    return m1*m2/(m1+m2)/(m1+m2)

def m1m2(Mc, eta):
    """Compute component masses from Mc, eta. Returns m1 >= m2"""
    etaV = np.array(1-4*eta,dtype=float) 
    if isinstance(eta, float):
        if etaV < 0:
            etaV = 0
            etaV_sqrt =0
        else:
            etaV_sqrt = np.sqrt(etaV)
    else:
        indx_ok = etaV>=0
        etaV_sqrt = np.zeros(len(etaV),dtype=float)
        etaV_sqrt[indx_ok] = np.sqrt(etaV[indx_ok])
        etaV_sqrt[np.logical_not(indx_ok)] = 0 # set negative cases to 0, so no sqrt problems
    m1 = 0.5*Mc*eta**(-3./5.)*(1. + etaV_sqrt)
    m2 = 0.5*Mc*eta**(-3./5.)*(1. - etaV_sqrt)
    return m1, m2

def eta_crit(Mc, m2_min):
    sol = scipy.optimize.root(lambda etv: m1m2(Mc, etv)[1] - m2_min, 0.23)
    return sol.x[0]

def Mceta(m1, m2):
    """Compute chirp mass and symmetric mass ratio from component masses"""
    Mc = (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)
    eta = m1*m2/(m1+m2)/(m1+m2)
    return Mc, eta

#
# Tidal parameter conversion functions
#
def tidal_lambda_tilde(mass1, mass2, lambda1, lambda2):
    """
    'Effective' lambda parameters.
    Lackey et al https://arxiv.org/pdf/1402.5156.pdf, Eq. (5,6).
    """
    mt = mass1 + mass2
    eta = mass1 * mass2 / mt**2
    q = np.sqrt(1 - 4*eta)
    lt1, lt2 = lambda1, lambda2 # lambda1 / mass1**5, lambda2 / mass2**5  # Code is already dimensionless
    lt_sym = lt1 + lt2
    lt_asym = lt1 - lt2
#    if mass1 < mass2:
#        q*=-1
    q*= np.sign(mass1-mass2)

    lam_til = (1 + 7*eta - 31*eta**2) * lt_sym + q * (1 + 9*eta - 11*eta**2) * lt_asym
    dlam_til = q * (1 - 13272*eta/1319 + 8944*eta**2/1319) * lt_sym + (1 - 15910*eta/1319 + 32850*eta**2/1319 + 3380*eta**3/1319) * lt_asym
    dlam_til *= 0.5
    lam_til *= 8. / 13
    return lam_til, dlam_til

def tidal_lambda_from_tilde(mass1, mass2, lam_til, dlam_til):
    """
    Determine physical lambda parameters from effective parameters.
    """
    mt = mass1 + mass2
    eta = mass1 * mass2 / mt**2
    # q = np.sqrt(1 - 4*eta)

    # a = (8./13) * (1 + 7*eta - 31*eta**2)
    # b = (8./13) * q * (1 + 9*eta - 11*eta**2)
    # c = 0.5 * q * (1 - 13272*eta/1319 + 8944*eta**2/1319)
    # d = 0.5 * (1 - 15910*eta/1319 + 32850*eta**2/1319 + 3380*eta**3/1319)

    # lambda1 = 0.5 * ((c - d) * lam_til - (a - b) * dlam_til)/(b*c - a*d)
    # lambda2 = 0.5 * ((c + d) * lam_til - (a + b) * dlam_til)/(a*d - b*c)
    lambda1,lambda2 = lam1_lam2_of_pe_params(eta, lam_til, dlam_til)

    return lambda1, lambda2

#
# Bernuzzi's tidal conversion functions
#
###
### Bernuzzi's conversion routines
###
try:
    from scipy.special import factorial2
except:  
    from scipy.misc import factorial2
def lamtilde_of_eta_lam1_lam2(eta, lam1, lam2):
    r"""
    $\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
    Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
    Lambda_2 is for the secondary star m_2.
    """
    return (8.0/13.0)*((1.0+7.0*eta-31.0*eta**2)*(lam1+lam2) + np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)*(lam1-lam2))
    
def deltalamtilde_of_eta_lam1_lam2(eta, lam1, lam2):
    r"""
    This is the definition found in Les Wade's paper.
    Les has factored out the quantity \sqrt(1-4\eta). It is different from Marc Favata's paper.
    $\delta\tilde\Lambda(\eta, \Lambda_1, \Lambda_2)$.
    Lambda_1 is assumed to correspond to the more massive (primary) star m_1.
    Lambda_2 is for the secondary star m_2.
    """
    return (1.0/2.0)*(
        np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)*(lam1+lam2)
        + (1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)*(lam1-lam2)
    )
    
def lam1_lam2_of_pe_params(eta, lamt, dlamt):
    """
    lam1 is for the the primary mass m_1.
    lam2 is for the the secondary mass m_2.
    m_1 >= m2.
    """
    a = (8.0/13.0)*(1.0+7.0*eta-31.0*eta**2)
    b = (8.0/13.0)*np.sqrt(1.0-4.0*eta)*(1.0+9.0*eta-11.0*eta**2)
    c = (1.0/2.0)*np.sqrt(1.0-4.0*eta)*(1.0 - 13272.0*eta/1319.0 + 8944.0*eta**2/1319.0)
    d = (1.0/2.0)*(1.0 - 15910.0*eta/1319.0 + 32850.0*eta**2/1319.0 + 3380.0*eta**3/1319.0)
    den = (a+b)*(c-d) - (a-b)*(c+d)
    lam1 = ( (c-d)*lamt - (a-b)*dlamt )/den
    lam2 = (-(c+d)*lamt + (a+b)*dlamt )/den
    # Adjust lam1 and lam2 if lam1 becomes negative
    # lam2 should be adjusted such that lamt is held fixed
#    if lam1<0:
#        lam1 = 0
#        lam2 = lamt / (a-b)
    return lam1, lam2

def Yagi13_fitcoefs(ell):
    r"""
    Coefficients of Yagi 2013 fits for multipolar
    $\bar{\lambda}_\ell = 2 k_\ell/(C^{2\ell+1} (2\ell-1)!!)$
    Tab.I (NS) http://arxiv.org/abs/1311.0872
    """
    if ell==3:
        c = [-1.15,1.18,2.51e-2,-1.31e-3,2.52e-5];
    elif ell==4:
        c = [-2.45,1.43,3.95e-2,-1.81e-3,2.8e-5];
    else:
        c = [];
    return c;

def Yagi13_fit_barlamdel(barlam2, ell):
    r"""
    Yagi 2013 fits for multipolar
    $\bar{\lambda}_\ell$ = 2 k_\ell/(C^{2\ell+1} (2\ell-1)!!)$
    Eq.(10),(61); Tab.I; Fig.8 http://arxiv.org/abs/1311.0872
    """
    lnx = np.log(barlam2);
    coefs = Yagi13_fitcoefs(ell);
    lny = np.polyval(coefs[::-1], lnx);
    return np.exp(lny)

def barlamdel_to_kappal(q, barlaml, ell):
    r"""
    $\kappa^{A,B}_\ell(\bar{\lambda}_\ell)$
    Assume $q=M_A/M_B>=1$
    """
    XA = q/(1.+q);
    XB = 1. - XA;
    blamfact = factorial2(2*ell-1) * barlaml;
    p = 2*ell + 1;
    kappaAl = blamfact * XA**p / q; 
    kappaBl = blamfact * XB**p * q; 
    return  kappaAl, kappaBl


#
# Other utility functions
#

def unwind_phase(phase,thresh=5.):
    """
    Unwind an array of values of a periodic variable so that it does not jump
    discontinuously when it hits the periodic boundary, but changes smoothly
    outside the periodic range.

    Note: 'thresh', which determines if a discontinuous jump occurs, should be
    somewhat less than the periodic interval. Empirically, 5 is usually a safe
    value of thresh for a variable with period 2 pi.

    Fast method: take element-by-element differences, use mod 2 pi, and then add
    """
    cnt = 0 # count number of times phase wraps around branch cut
    length = len(phase)
    unwound = np.zeros(length)
    delta = np.zeros(length)

    unwound[0] =phase[0]
    delta = np.mod(phase[1:] - phase[:-1]+np.pi,2*np.pi)-np.pi                 # d(n)= p(n+1)-p(n) : the step forward item. The modulus is positive, so use an offset. The phase delta should be ~ 0 for each step
    unwound[1:] =unwound[0]+np.cumsum(delta)            # d(n)+d(n-1)=p(n)
#    print delta, unwound

    # unwound[0] = phase[0]
    # for i in range(1,length):
    #     if phase[i-1] - phase[i] > thresh: # phase wrapped forward
    #         cnt += 1
    #     elif phase[i] - phase[i-1] > thresh: # phase wrapped backward
    #         cnt -= 1
    #     unwound[i] = phase[i] + cnt * 2. * np.pi
    return unwound
# def unwind_phase(phase,thresh=5.):
#     """
#     Unwind an array of values of a periodic variable so that it does not jump
#     discontinuously when it hits the periodic boundary, but changes smoothly
#     outside the periodic range.

#     Note: 'thresh', which determines if a discontinuous jump occurs, should be
#     somewhat less than the periodic interval. Empirically, 5 is usually a safe
#     value of thresh for a variable with period 2 pi.
#     """
#     cnt = 0 # count number of times phase wraps around branch cut
#     length = len(phase)
#     unwound = np.zeros(length)
#     unwound[0] = phase[0]
#     for i in range(1,length):
#         if phase[i-1] - phase[i] > thresh: # phase wrapped forward
#             cnt += 1
#         elif phase[i] - phase[i-1] > thresh: # phase wrapped backward
#             cnt += 1
#         unwound[i] = phase[i] + cnt * 2. * np.pi
#     return unwound

def nextPow2(length):
    """
    Find next power of 2 <= length
    """
    return int(2**np.ceil(np.log2(length)))

def findDeltaF(P):
    """
    Given ChooseWaveformParams P, generate the TD waveform,
    round the length to the next power of 2,
    and find the frequency bin size corresponding to this length.
    This is useful b/c deltaF is needed to define an inner product
    which is needed for norm_hoft and norm_hoff functions
    """
    h = hoft(P)
    return 1./(nextPow2(h.data.length) * P.deltaT)

def estimateWaveformDuration(P,LmaxEff=2):
    """
    Input:  P
    Output:estimated duration (in s) based on Newtonian inspiral from P.fmin to infinite frequency
    """
    fM  = P.fmin*(P.m1+P.m2)*lsu_G / lsu_C**3
    fM *= 2./LmaxEff  # if we use higher modes, lower the effective frequency, so HM start in band
    eta = symRatio(P.m1,P.m2)
    Msec = (P.m1+P.m2)*lsu_G / lsu_C**3
    return Msec*5./256. / eta* np.power((lsu_PI*fM),-8./3.)
def estimateDeltaF(P,LmaxEff=2):
    """
    Input:  P
    Output:estimated duration (in s) based on Newtonian inspiral from P.fmin to infinite frequency
    """
    T = estimateWaveformDuration(P,LmaxEff=2)+0.1  # buffer for merger
    return 1./(P.deltaT*nextPow2(T/P.deltaT))
    

def sanitize_eta(eta, tol=1.e-10, exception='error'):
    """
    If 'eta' is slightly outside the physically allowed range for
    symmetric mass ratio, push it back in. If 'eta' is further
    outside the physically allowed range, throw an error
    or return a special value.
    Explicitly:
        - If 'eta' is in [tol, 0.25], return eta.
        - If 'eta' is in [0, tol], return tol.
        - If 'eta' in is (0.25, 0.25+tol], return 0.25
        - If 'eta' < 0 OR eta > 0.25+tol,
            - if exception=='error' raise a ValueError
            - if exception is anything else, return exception
    """
    MIN = 0.
    MAX = 0.25
    if eta < MIN or eta > MAX+tol:
        if exception=='error':
            raise ValueError("Value of eta outside the physicaly-allowed range of symmetric mass ratio.")
        else:
            return exception
    elif eta < tol:
        return tol
    elif eta > MAX:
        return MAX
    else:
        return eta

#
# Utilities using Overlap based classes to calculate physical quantities
#
def singleIFOSNR(data, psd, fNyq, fmin=None, fmax=None):
    """
    Calculate single IFO SNR using inner product class.
    """
    assert data.deltaF == psd.deltaF
    IP = ComplexIP(fLow=fmin, fNyq=fNyq, deltaF=psd.deltaF, psd=psd, fMax=fmax, analyticPSD_Q=isinstance(psd, types.FunctionType))
    return IP.norm(data)

#
# Functions to generate waveforms
#
def hoft(P, Fp=None, Fc=None,**kwargs):
    """
    Generate a TD waveform from ChooseWaveformParams P
    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams.

    Returns a REAL8TimeSeries object
    """

    # special sauce for EOB, because it is so finicky regarding
    if P.approx == lalsim.EOBNRv2HM and P.m1 == P.m2:
#        print " Using ridiculous tweak for equal-mass line EOB"
        P.m2 = P.m1*(1-1e-6)
    extra_waveform_args = {}
    if 'extra_waveform_args' in kwargs:
        extra_waveform_args.update(kwargs['extra_waveform_args'])
    extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)
    if P.approx==lalsim.TEOBResumS and has_external_teobresum and info_use_ext:
        Lmax=4
        modes_used = []
        distance_s = P.dist/lal.C_SI
        m_total_s = MsunInSec*(P.m1+P.m2)/lal.MSUN_SI
        k_coprecessing_frame_check = [1, 0, 4, 8]
        for l in np.arange(2,Lmax+1,1):
            for m in np.arange(0,l+1,1):
                if m !=0:
                    modes_used.append((l,m))
        print(modes_used)
        k = modes_to_k(modes_used)
        k_coprecessing_frame = []
        for count,value in enumerate(k):
            if value in k_coprecessing_frame_check:
                k_coprecessing_frame.append(value)
        print(" k inertial modes: ", k, "k coprecessing frames: ", k_coprecessing_frame)
        if kwargs.get('force_22_mode', False):
            print('Forcing ONLY the 22 modes')
            k = [1]
        M1=P.m1/lal.MSUN_SI
        M2=P.m2/lal.MSUN_SI
        nu=M1*M2/((M1+M2)**2)
        ecc_ano=eccentric_anomaly_from_mean(P.meanPerAno,P.eccentricity)
        true_ano=true_anomaly_from_eccentric(ecc_ano,P.eccentricity)
        if P.E0 == 0.0:
            print("Using TEOBResumSDALI/GIOTTO standard call")
            pars = {
                'M'                  : M1+M2,
                'q'                  : M1/M2,
                'LambdaAl2'          : P.lambda1,
                'LambdaBl2'          : P.lambda2,
                'chi1x'              : P.s1x,
                'chi1y'              : P.s1y,
                'chi1z'              : P.s1z,
                'chi2x'              : P.s2x,
                'chi2y'              : P.s2y,
                'chi2z'              : P.s2z,
                'ecc'                : P.eccentricity,
                'inclination'        : P.incl,
                'coalescence_angle'  : np.pi / 2 - P.phiref,
                'srate_interp'       : 1./P.deltaT,
                'initial_frequency'  : P.fmin,
                'distance'           : P.dist/(lal.PC_SI*1e6),
                'anomaly'            : true_ano,
#                'a6c'                : P.a6c,
                'domain'             : 0,
                'arg_out'            : "yes",
                'interp_uniform_grid': "yes",
                'use_geometric_units': "no",
                # spin_flx can be "EOB" or "PN" prescription. Currently suggested to use EOB as standard (use PN for GIOTTO like prescription)
                'spin_flx'           : "EOB",
                'spin_interp_domain' : 0,
                # This can be "QNMs" or "constant". Currently it looks like without QNMs enabled gives basically only prior for precessing events (even for low mass events)
                'ringdown_eulerangles': "QNMs",
                'use_mode_lm'        : k_coprecessing_frame,
                #                'output_lm'          : k,
#                'df'                 : P.deltaF,
                'output_hpc'         : "no"
            }
            if (np.abs(P.s1x) >  1e-4 or P.s1y!=0.0 or P.s2x!=0.0 or P.s2y!=0.0):
                pars.update({'use_mode_lm_inertial': k})
            if P.a6c < 1000:
                pars.update({'a6c' : P.a6c})
        else:
            print("Using hyperbolic call")
            hyp_wav = True # convenient way to know if the waveform is hyperbolic
            
            pars = {
                'M'                  : M1+M2,
                'q'                  : M1/M2,
                'H_hyp'              : P.E0, # energy at initial separation
                'j_hyp'              : P.p_phi0, # angular momentum at initial separation
                'r_hyp'              : 6000.0, #hardcoded separation: may need to increase with certain systems; may want to make a option
                'npc'                : "no",
                'nqc_coefs_hlm'      : "none",
                'nqc_coefs_flx'      : "none",
                'ode_tmax'           : 3e4, # controls how long end of waveform is; may need to change for some systems
                'LambdaAl2'          : P.lambda1,
                'LambdaBl2'          : P.lambda2,
                'chi1x'              : P.s1x,
                'chi1y'              : P.s1y,
                'chi1z'              : P.s1z,
                'chi2x'              : P.s2x,
                'chi2y'              : P.s2y,
                'chi2z'              : P.s2z,
                'inclination'        : P.incl,
                'coalescence_angle'  : np.pi / 2 - P.phiref,
                'srate_interp'       : 1./P.deltaT,
                'initial_frequency'  : P.fmin,
                'distance'           : P.dist/(lal.PC_SI*1e6),
                'domain'             : 0,
                'arg_out'            : "yes",
                'interp_uniform_grid': "yes",
                'use_geometric_units': "no",
                # spin_flx can be "EOB" or "PN" prescription. Currently suggested to use EOB as standard (use PN for GIOTTO like prescription)
                'spin_flx'           : "EOB",
                'spin_interp_domain' : 0,
                # This can be "QNMs" or "constant". Currently it looks like without QNMs enabled gives basically only prior for precessing events (even for low mass events)
                'ringdown_eulerangles': "QNMs",
                'use_mode_lm'        : k_coprecessing_frame,
                #                'output_lm'          : k,
#                'df'                 : P.deltaF,
                'output_hpc'         : "no"
            }
            if (np.abs(P.s1x) >  1e-4 or P.s1y!=0.0 or P.s2x!=0.0 or P.s2y!=0.0):
                pars.update({'use_mode_lm_inertial': k})

        print("Starting EOBRun_module")
        print(pars)
        t, hptmp, hctmp, hlmtmp, dyn = EOBRun_module.EOBRunPy(pars)
        print("EOBRun_module done")
        hpepoch = -P.deltaT*np.argmax(np.abs(hptmp)**2+np.abs(hctmp)**2)
        hplen = len(hptmp)
        hp = {}
        hc = {}
        hp = lal.CreateREAL8TimeSeries("hoft", hpepoch, 0,
                                       P.deltaT, lsu_DimensionlessUnit, hplen)
        hc = lal.CreateREAL8TimeSeries("hoft", hpepoch, 0,
                                       P.deltaT, lsu_DimensionlessUnit, hplen)
        hp.data.data = hptmp
        hc.data.data = hctmp



    else:
        hp, hc = lalsim.SimInspiralTD( \
                                       P.m1, P.m2, \
                                       P.s1x, P.s1y, P.s1z, \
                                       P.s2x, P.s2y, P.s2z, \
                                       P.dist, P.incl, P.phiref,  \
                                       0 , P.eccentricity, P.meanPerAno, \
                                       P.deltaT, P.fmin, P.fref, \
                                       extra_params, P.approx)
        
# O2 branch
#    hp, hc = lalsim.SimInspiralTD( \
#            P.m1, P.m2, \
#            P.s1x, P.s1y, P.s1z, \
#            P.s2x, P.s2y, P.s2z, \
#            P.dist, P.incl, P.phiref,  \
#            P.psi, P.eccentricity, P.meanPerAno, \
#            P.deltaT, P.fmin, P.fref, \
#            extra_params, P.approx)

    if Fp!=None and Fc!=None:
        hp.data.data *= Fp
        hc.data.data *= Fc
        hp = lal.AddREAL8TimeSeries(hp, hc)
        ht = hp
    elif P.radec==False:
        fp = Fplus(P.theta, P.phi, P.psi)
        fc = Fcross(P.theta, P.phi, P.psi)
        hp.data.data *= fp
        hc.data.data *= fc
        hp = lal.AddREAL8TimeSeries(hp, hc)
        ht = hp
    else:
        hp.epoch = hp.epoch + P.tref
        hc.epoch = hc.epoch + P.tref
        ht = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, 
                P.phi, P.theta, P.psi, 
                lalsim.DetectorPrefixToLALDetector(str(P.detector)))
    if P.taper != lsu_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(ht.data, P.taper)
    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen >= ht.data.length
        ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    return ht

def hoff(P, Fp=None, Fc=None, fwdplan=None):
    """
    Generate a FD waveform from ChooseWaveformParams P.
    Will return a COMPLEX16FrequencySeries object.

    If P.approx is a FD approximant, hoff_FD is called.
    This path calls SimInspiralChooseFDWaveform
        fwdplan must be None for FD approximants.

    If P.approx is a TD approximant, hoff_TD is called.
    This path calls ChooseTDWaveform and performs an FFT.
        The TD waveform will be zero-padded so it's Fourier transform has
        frequency bins of size P.deltaT.
        If P.deltaF == None, the TD waveform will be zero-padded
        to the next power of 2.
    """
    # For FD approximants, use the ChooseFDWaveform path = hoff_FD
    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
        # Raise exception if unused arguments were specified
        if fwdplan is not None:
            raise ValueError('FFT plan fwdplan given with FD approximant.\nFD approximants cannot use this.')
        hf = hoff_FD(P, Fp, Fc)

    # For TD approximants, do ChooseTDWaveform + FFT path = hoff_TD
    else:
        hf = hoff_TD(P, Fp, Fc, fwdplan)

    return hf

def hoff_TD(P, Fp=None, Fc=None, fwdplan=None):
    """
    Generate a FD waveform from ChooseWaveformParams P
    by creating a TD waveform, zero-padding and
    then Fourier transforming with FFTW3 forward FFT plan fwdplan

    If P.deltaF==None, just pad up to next power of 2
    If P.deltaF = 1/X, will generate a TD waveform, zero-pad to length X seconds
        and then FFT. Will throw an error if waveform is longer than X seconds

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, you may to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)

    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams

    Returns a COMPLEX16FrequencySeries object
    """
    ht = hoft(P, Fp, Fc)

    if P.deltaF == None: # h(t) was not zero-padded, so do it now
        TDlen = nextPow2(ht.data.length)
        ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == ht.data.length
    
    if fwdplan==None:
        fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    FDlen = int(TDlen/2+1)
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lsu_HertzUnit, 
            FDlen)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    return hf

def hoff_FD(P, Fp=None, Fc=None,**kwargs):
    """
    Generate a FD waveform for a FD approximant.
    Note that P.deltaF (which is None by default) must be set
    """
    if P.deltaF is None:
        raise ValueError('None given for freq. bin size P.deltaF')
    extra_waveform_args = {}
    if 'extra_waveform_args' in kwargs:
        extra_waveform_args.update(kwargs['extra_waveform_args'])
    extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)

    hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(P.phiref, P.deltaF,
             P.m1, P.m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z,
             P.dist, P.incl, P.phiref, P.psi,
             P.eccentricity, P.meanPerAno, P.deltaF, 
             P.fmin, P.fmax, P.fref, 
             extra_params, P.approx)
#            P.m1, P.m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin,
#            P.fmax, P.fref, P.dist, P.incl, P.lambda1, P.lambda2, P.waveFlags,
#            P.nonGRparams, P.ampO, P.phaseO, P.approx)
    if Fp is not None and Fc is not None:
        hptilde.data.data *= Fp
        hctilde.data.data *= Fc
        hptilde = lal.AddCOMPLEX16FrequencySeries(hptilde, hctilde)
        htilde = hptilde
    elif P.radec==False:
        fp = Fplus(P.theta, P.phi, P.psi)
        fc = Fcross(P.theta, P.phi, P.psi)
        hptilde.data.data *= fp
        hctilde.data.data *= fc
        hptilde = lal.AddCOMPLEX16FrequencySeries(hptilde, hctilde)
        htilde = hptilde
    else:
        raise ValueError('Must use P.radec=False for FD approximant (for now)')
    # N.B. TaylorF2(RedSpin)(Tidal)  stop filling the output array at ISCO.
    # The Hermitian inner product classes now expect the arrays to be a
    # power of two plus one. Therefore, we zero-pad the output
    # so it will work with lalsimutils inner products
    FDlen = int(1./P.deltaF/P.deltaT/2.+1)
    if htilde.data.length != FDlen:
        htilde = lal.ResizeCOMPLEX16FrequencySeries(htilde, 0, FDlen)
    return htilde

def norm_hoff(P, IP, Fp=None, Fc=None, fwdplan=None):
    """
    Generate a normalized FD waveform from ChooseWaveformParams P.
    Will return a COMPLEX16FrequencySeries object.

    If P.approx is a FD approximant, norm_hoff_FD is called.
    This path calls SimInspiralChooseFDWaveform
        fwdplan must be None for FD approximants.

    If P.approx is a TD approximant, norm_hoff_TD is called.
    This path calls ChooseTDWaveform and performs an FFT.
        The TD waveform will be zero-padded so it's Fourier transform has
        frequency bins of size P.deltaT.
        If P.deltaF == None, the TD waveform will be zero-padded
        to the next power of 2.
    """
    # For FD approximants, use the ChooseFDWaveform path = hoff_FD
    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
        # Raise exception if unused arguments were specified
        if fwdplan is not None:
            raise ValueError('FFT plan fwdplan given with FD approximant.\nFD approximants cannot use this.')
        hf = norm_hoff_FD(P, IP, Fp, Fc)

    # For TD approximants, do ChooseTDWaveform + FFT path = hoff_TD
    else:
        hf = norm_hoff_TD(P, IP, Fp, Fc, fwdplan)

    return hf

def norm_hoff_TD(P, IP, Fp=None, Fc=None, fwdplan=None):
    """
    Generate a waveform from ChooseWaveformParams P normalized according
    to inner product IP by creating a TD waveform, zero-padding and
    then Fourier transforming with FFTW3 forward FFT plan fwdplan.
    Returns a COMPLEX16FrequencySeries object.

    If P.deltaF==None, just pad up to next power of 2
    If P.deltaF = 1/X, will generate a TD waveform, zero-pad to length X seconds
        and then FFT. Will throw an error if waveform is longer than X seconds

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, you may to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)

    You may pass in antenna patterns Fp, Fc. If none are provided, they will
    be computed from the information in ChooseWaveformParams.

    N.B. IP and the waveform generated from P must have the same deltaF and 
        the waveform must extend to at least the highest frequency of IP's PSD.
    """
    hf = hoff_TD(P, Fp, Fc, fwdplan)
    norm = IP.norm(hf)
    hf.data.data /= norm
    return hf

def norm_hoff_FD(P, IP, Fp=None, Fc=None):
    """
    Generate a FD waveform for a FD approximant normalized according to IP.
    Note that P.deltaF (which is None by default) must be set.
    IP and the waveform generated from P must have the same deltaF and 
        the waveform must extend to at least the highest frequency of IP's PSD.
    """
    if P.deltaF is None:
        raise ValueError('None given for freq. bin size P.deltaF')

    htilde = hoff_FD(P, Fp, Fc)
    norm = IP.norm(htilde)
    htilde.data.data /= norm
    return htilde

def non_herm_hoff(P):
    """
    Generate a FD waveform with two-sided spectrum. i.e. not assuming
    the Hermitian symmetry applies
    """
    htR = hoft(P) # Generate real-valued TD waveform
    if P.deltaF == None: # h(t) was not zero-padded, so do it now
        TDlen = nextPow2(htR.data.length)
        htR = lal.ResizeREAL8TimeSeries(htR, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == htR.data.length
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(htR.data.length,0)
    htC = lal.CreateCOMPLEX16TimeSeries("hoft", htR.epoch, htR.f0,
            htR.deltaT, htR.sampleUnits, htR.data.length)
    # copy h(t) into a COMPLEX16 array which happens to be purely real
    for i in range(htR.data.length):
        htC.data.data[i] = htR.data.data[i]
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
            htR.epoch, htR.f0, 1./htR.deltaT/htR.data.length, lsu_HertzUnit, 
            htR.data.length)
    lal.COMPLEX16TimeFreqFFT(hf, htC, fwdplan)
    return hf



## Version protection
#   Pull out arguments of the ModesFromPolarizations function
#argist_FromPolarizations=lalsim.SimInspiralTDModesFromPolarizations.__doc__.split('->')[0].replace('SimInspiralTDModesFromPolarizations','').replace('REAL8','').replace('Dict','').replace('Approximant','').replace('(','').replace(')','').split(',')


def hlmoft(P, Lmax=2,nr_polarization_convention=False, fixed_tapering=False, silent=True, fd_standoff_factor=0.964,no_condition=False,fd_L_frame=False,fd_centering_factor=0.9,fd_alignment_postevent_time=None,**kwargs ):
    """
    Generate the TD h_lm -2-spin-weighted spherical harmonic modes of a GW
    with parameters P. Returns a SphHarmTimeSeries, a linked-list of modes with
    a COMPLEX16TimeSeries and indices l and m for each node.

    The linked list will contain all modes with l <= Lmax
    and all values of m for these l.

    nr_polarization_convention : apply a factor -1 to all modes provided by lalsuite

    fd_standoff_factor: for ChooseFDWaveform, reduce starting frequency P.fmin by this factor internally when generating.
          FD waveform calculation implicitly assumes fmin is in the inspiral regime (if not you are making bad life choices).
          Normal default value of 0.964 is based on changing inspiral duration by 10% (1.1^(3/8) ~ 1.036). Any constant factor will change
          the inspiral duration by a factor (1./fd_standoff_factor)^(8/3) or so.

          WARNING: internal_hlm_generator will CHANGE this factor to 0.9, to better match complex_hoft.  I'm keeping the default of this function here to match the nominal documentaiton of SimInspiralTD, while internal_hlm_generator has a different factor, chosen to match its actual output.
    

    no_condition: disable conditioning (for ChooseFDModes specifically). For diagnostic plots on impact of/need for conditioning.
    fd_L_frame: rotate hlmoft to L frame from J frame for return values from ChooseFDModes (XO4/XPHM). 
    """
    assert Lmax >= 2

    # Check that masses are not nan!
    assert (not np.isnan(P.m1)) and (not np.isnan(P.m2)), " masses are NaN "
    phiref_shift_convention =0


    # includes the 'release' version
    extra_waveform_args = {}
    fmin_backstop = 10 # artificial P.fmin to use for setting tapering scales and other things, so P.fmin can be reserved for passing as '0' when requested in crazy ways for NRSur, etc
    if 'fmin_backstop' in kwargs:
        fmin_backstop = kwargs['fmin_backstop']
    if 'extra_waveform_args' in kwargs:
        extra_waveform_args.update(kwargs['extra_waveform_args'])
    extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)

    sign_factor = 1
    if nr_polarization_convention or (P.approx==lalsim.SpinTaylorT1 or P.approx==lalsim.SpinTaylorT2 or P.approx==lalsim.SpinTaylorT3 or P.approx==lalsim.SpinTaylorT4):
        sign_factor = -1
    if (P.approx == lalIMRPhenomHM or P.approx == lalIMRPhenomXHM or P.approx == lalIMRPhenomXPHM or P.approx == lalIMRPhenomXO4a or P.approx == lalSEOBNRv4HM_ROM and is_ChooseFDModes_present):
       is_precessing=True
       if np.sqrt(P.s1x**2 + P.s1y**2 + P.s2x**2+P.s2y**2)<1e-10:  # only perform if really precessing, otherwise skip. Really only for XP variants
           is_precessing=False
       if P.fref==0 and (P.approx == lalIMRPhenomXPHM):
          P.fref=P.fmin
#       extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)
       fNyq = 0.5/P.deltaT
       TDlen = int(1./(P.deltaT*P.deltaF))
       if fd_alignment_postevent_time:
           if fd_alignment_postevent_time < TDlen*P.deltaT/2:
               fd_centering_factor = 1-fd_alignment_postevent_time/(TDlen*P.deltaT)  # align so there is a time fd_alignment_time_postevent
           else:
               print(" Warning: fd alignment postevent time requested incompatible with short duration ",file=sys.stderr)
       fNyq_offset = fNyq - P.deltaF
       # Argh: https://git.ligo.org/waveforms/reviews/imrphenomxhm-amplitude-recalibration/-/wikis/home#review-statement
#       if P.approx == lalIMRPhenomXPHM and 'release' in kwargs:
#           lalsim.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(extra_params, kwargs['release'])
       hlms_struct = lalsim.SimInspiralChooseFDModes(P.m1, P.m2, \
                                                     P.s1x, P.s1y, P.s1z, \
                                                     P.s2x, P.s2y, P.s2z, \
                                                     P.deltaF, P.fmin*fd_standoff_factor, fNyq, P.fref, P.phiref, P.dist, P.incl, extra_params, P.approx)
       hlmsT = {}
       hlms = {}
       hlmsdict = SphHarmFrequencySeries_to_dict(hlms_struct,Lmax)

       # FD conditioning using fd_standoff_factor, HIGH PASS FILTER to eliminate LOW FREQUENCIES
       # Note this conditioning is based on the L=2 mode start frequency -- we should use different conditioning for each coprecessing mode
       # but that is very difficult to do because modes of different 'm' and thus typical frequency generally mix
       # Note also that unless the segment length is large, this is often surprisingly few frequency bins for tapering
       if not(no_condition):
           our_fvals = evaluate_fvals(hlmsdict[(2,2)])
           vectaper_symmetric  = np.ones(len(our_fvals))
           indx_below = np.logical_and(np.abs(np.abs(our_fvals)<P.fmin), np.abs(our_fvals)>=P.fmin*fd_standoff_factor)
           vectaper_symmetric[indx_below] = 0.5 + 0.5*np.cos(np.pi* (np.abs(our_fvals[indx_below])/P.fmin - 1)/(1-fd_standoff_factor))
           indx_within = np.abs(our_fvals)< P.fmin*fd_standoff_factor
           for mode in hlmsdict:
               hlmsdict[mode].data.data*=vectaper_symmetric
               hlmsdict[mode].data.data[indx_within]=0
       
       # Base taper, based on 1% of waveform length
       ntaper = int(0.01*TDlen)  # fixed 1% of waveform length, at start
       ntaper = np.max([ntaper, int(1./(P.fmin*P.deltaT))])  # require at least one waveform cycle of tapering; should never happen
       vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))

       # Taper any wraparound that has happened after the peak
       #    - Based on SimInspiralTDFromTD  https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c
       tchirp = lalsim.SimInspiralChirpTimeBound(P.fmin, P.m1,P.m2, P.s1z, P.s2z)
       s = lalsim.SimInspiralFinalBlackHoleSpinBound(P.s1z,P.s2z)
       t_merge = lalsim.SimInspiralMergeTimeBound(P.m1,P.m2) + lalsim.SimInspiralRingdownTimeBound(P.m1+P.m2,s)
       t_extra = 3./P.fmin
       indx_crit = int(TDlen*(fd_centering_factor)) + int((t_merge+t_extra)/P.deltaT)  # t_extra is a lot of time generally if fmin is low
       if indx_crit + ntaper > TDlen:
           indx_crit = TDlen - ntaper

       for mode in hlmsdict:
          hlmsdict[mode] = lal.ResizeCOMPLEX16FrequencySeries(hlmsdict[mode],0, TDlen)
          hlmsT[mode] = DataInverseFourier(hlmsdict[mode])
          # Phase factors: see crazy conventions in https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c
          if True: #P.approx == lalIMRPhenomXHM or P.approx == lalIMRPhenomHM:
              hlmsT[mode].data.data = -1*hlmsT[mode].data.data   # shifts polarization angle
          # Assume fourier transform is CENTERED.  This makes sure any ringdown dies out. In practice, too pessimistic .. better to be asymmetric
          factor_centering = fd_centering_factor   # not clear it works right if we shift it
          hlmsT[mode].data.data = np.roll(hlmsT[mode].data.data,-int(hlmsT[mode].data.length*(1-factor_centering)))  
          hlmsT[mode].epoch = -(hlmsT[mode].data.length*hlmsT[mode].deltaT)*(factor_centering)
          if not(no_condition):
              # Taper at the start of the segment
              hlmsT[mode].data.data[:ntaper]*=vectaper
              # Taper anything at the *end* of the segment. This could be  important when interacting with real data, which when wrapped is strongly discontinuous
              #     - this generally will NOT taper everything, because some of the turn-on generally extends well past this
              if TDlen-ntaper < indx_crit:
                  # just multiply the very end by a small amount of tapering
                  hlmsT[mode].data.data[TDlen-ntaper:]*=vectaper[::-1]
              else:
                  # zero out a a lot of time at the end, and taper time as we approach this critical extra time
                  hlmsT[mode].data.data[indx_crit:indx_crit+ntaper] *= vectaper[::-1]
                  hlmsT[mode].data.data[indx_crit+ntaper:] = 0

       # Rotate if requested, and if P.fref is specified
       # see discussion in https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c
       if is_precessing and fd_L_frame and P.fref:
             alpha,beta,gamma =extract_JL_angles(P,return_inverse=True)
             # alpha0, beta==theta_JN already identified above. Missing polarization rotation factor as well
             # zeta_pol will not be used since it is inclination-dependent and therefore depends on extrinsic choices! Also normally calling with P.incl ==0
             _, _, _, theta_JN, alpha0, misc_phi, zeta_pol = lalsim.SimIMRPhenomXPCalculateModelParametersFromSourceFrame(P.m1,P.m2, P.fref, P.phiref, P.incl, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, extra_params)
             # Get remaining psiJ quantity (should add to extract_JL_angles)
             thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, psiJ  = P.extract_system_frame()
             # theta_JN is not useful, for rotations will be close to P.incl in most cases
             alpha+= np.pi # empirically validated sign for XPHM, comparing precessing radiation to SEOBv4PHM
#             print(alpha, beta, gamma, zeta_pol)
#             print(alpha0, thetaJN, np.pi - phiJL,psiJ)
             hlmsT_alt = rotate_hlm_static(hlmsT, -gamma - np.pi/2, -beta,-alpha ,extra_polarization=psiJ)  
             hlmsT = hlmsT_alt
        # phase shift ChooseFDModes
#       for mode in hlmsT:
 #          hlmsT[mode].data.data *= np.exp(-1j*mode[1]*np.pi/2) # phase correction factor, empirically to match phase convention for precession for example. Note this should always be done

       if P.deltaF is not None:
          if not silent:
              print(hlmsT[mode].data.length,TDlen, " and in seconds ", hlmsT[mode].data.length*hlmsT[mode].deltaT,TDlen*P.deltaT, " with deltaT={} {}".format(hlmsT[mode].deltaT,P.deltaT))
          for mode in hlmsT:
             hlms[mode] = lal.ResizeCOMPLEX16TimeSeries(hlmsT[mode],0,TDlen)
       return hlms

    if (P.approx == lalsim.SEOBNRv2 or P.approx == lalsim.SEOBNRv1 or P.approx == lalSEOBv4 or P.approx==lalsim.SEOBNRv4_opt or P.approx == lalsim.EOBNRv2 or P.approx == lalTEOBv2 or P.approx==lalTEOBv4 or P.approx == lalSEOBNRv4HM):
        hlm_out = hlmoft_SEOB_dict(P,Lmax=Lmax)
        if True: #not (P.taper == lsu_TAPER_NONE): #True: #P.taper:
            ntaper = int(0.01*hlm_out[(2,2)].data.length)  # fixed 1% of waveform length, at start
            ntaper = np.max([ntaper, int(1./(P.fmin*P.deltaT))])  # require at least one waveform cycle of tapering; should never happen
            vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
            for key in hlm_out.keys():
                hlm_out[key].data.data *= sign_factor
                # Apply a naive filter to the start. Ideally, use an earlier frequency to start with
                hlm_out[key].data.data[:ntaper]*=vectaper
        return hlm_out
    elif P.approx == lalsim.SEOBNRv3 or P.approx == lalsim.SEOBNRv3_opt:
        hlm_out = hlmoft_SEOBv3_dict(P)
        if not hlm_out:
            print( " Failed generation: SEOBNRv3 ")
            sys.exit(0)
        if True: #P.taper:
            ntaper = int(0.01*hlm_out[(2,2)].data.length)  # fixed 1% of waveform length, at start
            ntaper = np.max([ntaper, int(1./(P.fmin*P.deltaT))])  # require at least one waveform cycle of tapering; should never happen
            vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
            for key in hlm_out.keys():
                # Apply a naive filter to the start. Ideally, use an earlier frequency to start with
                hlm_out[key].data.data *= sign_factor
                hlm_out[key].data.data[:ntaper]*=vectaper
        return hlm_out

    if ('IMRPhenomP' in  lalsim.GetStringFromApproximant(P.approx) or P.approx == lalIMRPhenomXP or (check_FD_pending(P.approx)) ): # and not (P.SoftAlignedQ()):
        print("Passing model through hlmoft_IMRPv2_dict")
        hlms = hlmoft_IMRPv2_dict(P)
        if not (P.deltaF is None):
            TDlen = int(1./P.deltaF * 1./P.deltaT)
            for mode in hlms:
                if TDlen > hlms[mode].data.length:
                    hlms[mode] = lal.ResizeCOMPLEX16TimeSeries(hlms[mode],0,TDlen)
                if TDlen < hlms[mode].data.length:  # we have generated too long a signal!...truncate from LEFT. Danger!
                    hlms[mode] = lal.ResizeCOMPLEX16TimeSeries(hlms[mode],hlms[mode].data.length-TDlen,TDlen)
        if True: #P.taper:
            ntaper = int(0.01*hlms[(2,2)].data.length)  # fixed 1% of waveform length, at start, be consistent with other methods
            ntaper = np.max([ntaper, int(1./(P.fmin*P.deltaT))])  # require at least one waveform cycle of tapering; should never happen
            vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
            for key in hlms.keys():
                # Apply a naive filter to the start. Ideally, use an earlier frequency to start with
                hlms[key].data.data *= sign_factor
                hlms[key].data.data[:ntaper]*=vectaper

        return hlms

    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
        print("Passing model through hlmoft_FromFD_dict")
        hlms = hlmoft_FromFD_dict(P,Lmax=Lmax)
    elif (P.approx == lalsim.TaylorT1 or P.approx==lalsim.TaylorT2 or P.approx==lalsim.TaylorT3 or P.approx==lalsim.TaylorT4 or P.approx == lalsim.EOBNRv2HM or P.approx==lalsim.EOBNRv2 or P.approx==lalsim.SpinTaylorT1 or P.approx==lalsim.SpinTaylorT2 or P.approx==lalsim.SpinTaylorT3 or P.approx==lalsim.SpinTaylorT4 or P.approx == lalSEOBNRv4P or P.approx == lalSEOBNRv4PHM or P.approx == lalNRSur7dq4 or P.approx == lalNRSur7dq2 or P.approx==lalNRHybSur3dq8 or P.approx == lalIMRPhenomTPHM) or (P.approx ==lalsim.TEOBResumS and not(has_external_teobresum) and not(info_use_resum_polarizations)):
        # approximant likst: see https://git.ligo.org/lscsoft/lalsuite/blob/master/lalsimulation/lib/LALSimInspiral.c#2541
        extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)
        # prevent segmentation fault when hitting nyquist frequency violations
        if (P.approx == lalSEOBNRv4PHM or P.approx == lalSEOBNRv4P) and P.approx >0:
            try:
                # fails on error, throws EDOM. This *should* be done internally by ChooseTDModes, but is not working
                test = lalsim.EOBCheckNyquistFrequency(P.m1/lal.MSUN_SI,P.m2/lal.MSUN_SI, np.array([P.s1x,P.s1y, P.s1z]), np.array([P.s2x,P.s2y, P.s2z]), Lmax,P.approx, P.deltaT)
            except Exception as e:
                raise NameError(" Nyquist frequency error for v4P/v4PHM, check srate")
        # extra phase factor of pi/2 added to fix consistency issue with our reconstruction code and other convention; easily demonstrated with precessing binaries, and also in docs
        phiref_shift_convention =np.pi/2
        hlms = lalsim.SimInspiralChooseTDModes(P.phiref, P.deltaT, P.m1, P.m2, \
	    P.s1x, P.s1y, P.s1z, \
	    P.s2x, P.s2y, P.s2z, \
            P.fmin, P.fref, P.dist, extra_params, \
             Lmax, P.approx)
    elif P.approx ==lalsim.TEOBResumS and has_external_teobresum and not(info_use_resum_polarizations):  # don't call external if fallback to polarizations
        print("Using TEOBResumS hlms (only using (2,2); (2,1); (3,3); (4,4) in coprecessing frame)")
        modes_used = []
        modes_used_check = []
        distance_s = P.dist/lal.C_SI
        m_total_s = MsunInSec*(P.m1+P.m2)/lal.MSUN_SI
        k_coprecessing_frame_check = [1, 0, 4, 8]
        for l in np.arange(2,Lmax+1,1):
            for m in np.arange(0,l+1,1):
                if m !=0:
                    modes_used.append((l,m))
        print(modes_used)
        k = modes_to_k(modes_used)
        k_coprecessing_frame = []
        for count,value in enumerate(k):
            if value in k_coprecessing_frame_check:
                k_coprecessing_frame.append(value)
        print(" k inertial modes: ", k, "k coprecessing frames: ", k_coprecessing_frame)
        if kwargs.get('force_22_mode', False):
            print('Forcing ONLY the 22 modes')
            k = [1]
        M1=P.m1/lal.MSUN_SI
        M2=P.m2/lal.MSUN_SI
        nu=M1*M2/((M1+M2)**2)
        ecc_ano=eccentric_anomaly_from_mean(P.meanPerAno,P.eccentricity)
        true_ano=true_anomaly_from_eccentric(ecc_ano,P.eccentricity)
        if P.E0 == 0.0:
            print("Using TEOBResumSDALI/GIOTTO standard call")
            pars = {
                'M'                  : M1+M2,
                'q'                  : M1/M2,
                'LambdaAl2'          : P.lambda1,
                'LambdaBl2'          : P.lambda2,
                'chi1x'              : P.s1x,
                'chi1y'              : P.s1y,
                'chi1z'              : P.s1z,
                'chi2x'              : P.s2x,
                'chi2y'              : P.s2y,
                'chi2z'              : P.s2z,
                'ecc'                : P.eccentricity,
                'inclination'        : P.incl,
                'coalescence_angle'  : np.pi / 2 - P.phiref,
                'srate_interp'       : 1./P.deltaT,
                'initial_frequency'  : P.fmin,
                'distance'           : P.dist/(lal.PC_SI*1e6),
                'anomaly'            : true_ano,
#                'a6c'                : P.a6c,
                'domain'             : 0,
                'arg_out'            : "yes",
                'interp_uniform_grid': "yes",
                'use_geometric_units': "no",
                # spin_flx can be "EOB" or "PN" prescription. Currently suggested to use EOB as standard (use PN for GIOTTO like prescription)
                'spin_flx'           : "EOB",
                'spin_interp_domain' : 0,
                # This can be "QNMs" or "constant". Currently it looks like without QNMs enabled gives basically only prior for precessing events (even for low mass events)
                'ringdown_eulerangles': "QNMs",
                'use_mode_lm'        : k_coprecessing_frame,
                #                'output_lm'          : k,
#                'df'                 : P.deltaF,
                'output_hpc'         : "no"
            }
            if (np.abs(P.s1x) >  1e-4 or P.s1y!=0.0 or P.s2x!=0.0 or P.s2y!=0.0):
                pars.update({'use_mode_lm_inertial': k})
            if P.a6c < 1000:
                pars.update({'a6c' : P.a6c})
        else:
            print("Using hyperbolic call")
            hyp_wav = True # convenient way to know if the waveform is hyperbolic
            
            pars = {
                'M'                  : M1+M2,
                'q'                  : M1/M2,
                'H_hyp'              : P.E0, # energy at initial separation
                'j_hyp'              : P.p_phi0, # angular momentum at initial separation
                'r_hyp'              : 6000.0, #hardcoded separation: may need to increase with certain systems; may want to make a option
                'npc'                : "no",
                'nqc_coefs_hlm'      : "none",
                'nqc_coefs_flx'      : "none",
                'ode_tmax'           : 3e4, # controls how long end of waveform is; may need to change for some systems
                'LambdaAl2'          : P.lambda1,
                'LambdaBl2'          : P.lambda2,
                'chi1x'              : P.s1x,
                'chi1y'              : P.s1y,
                'chi1z'              : P.s1z,
                'chi2x'              : P.s2x,
                'chi2y'              : P.s2y,
                'chi2z'              : P.s2z,
                'inclination'        : P.incl,
                'coalescence_angle'  : np.pi / 2 - P.phiref,
                'srate_interp'       : 1./P.deltaT,
                'initial_frequency'  : P.fmin,
                'distance'           : P.dist/(lal.PC_SI*1e6),
                'domain'             : 0,
                'arg_out'            : "yes",
                'interp_uniform_grid': "yes",
                'use_geometric_units': "no",
                # spin_flx can be "EOB" or "PN" prescription. Currently suggested to use EOB as standard (use PN for GIOTTO like prescription)
                'spin_flx'           : "EOB",
                'spin_interp_domain' : 0,
                # This can be "QNMs" or "constant". Currently it looks like without QNMs enabled gives basically only prior for precessing events (even for low mass events)
                'ringdown_eulerangles': "QNMs",
                'use_mode_lm'        : k_coprecessing_frame,
                #                'output_lm'          : k,
#                'df'                 : P.deltaF,
                'output_hpc'         : "no"
            }
            if (np.abs(P.s1x) >  1e-4 or P.s1y!=0.0 or P.s2x!=0.0 or P.s2y!=0.0):
                pars.update({'use_mode_lm_inertial': k})
                
        # Run the WF generator
        print("Starting EOBRun_module")
        print(pars)
        t, hptmp, hctmp, hlmtmp, dym = EOBRun_module.EOBRunPy(pars)

        k_list_orig = hlmtmp.keys()
#        print(hlmtmp.keys())
        rho_net = np.zeros(len(t))
        for mode in hlmtmp:
            rho_net += np.abs(hlmtmp[mode][0])**2
        hlmepoch = -P.deltaT*np.argmax(rho_net)
#        hpepoch = -P.deltaT*np.argmax(np.abs(hptmp)**2+np.abs(hctmp)**2)
        hlmlen = len(hptmp)
        hlm = {}
        hlmtmp2 = {}
        if True:
            modes_used_new = []
            for k in hlmtmp.keys():
                if k=="0":
                    modes_used_new.append((2,1))
                    hlmtmp2[(2,1)]=np.array(hlmtmp[k])
                if k=="-0":
                    modes_used_new.append((2,-1))
                    hlmtmp2[(2,-1)]=np.array(hlmtmp[k])
                if k=="20":
                    modes_used_new.append((2,0))
                    hlmtmp2[(2,0)]=np.array(hlmtmp[k])
                if k=="1":
                    modes_used_new.append((2,2))
                    hlmtmp2[(2,2)]=np.array(hlmtmp[k])
                if k=="-1":
                    modes_used_new.append((2,-2))
                    hlmtmp2[(2,-2)]=np.array(hlmtmp[k])
                if k=="2":
                    modes_used_new.append((3,1))
                    hlmtmp2[(3,1)]=np.array(hlmtmp[k])
                if k=="-2":
                    modes_used_new.append((3,-1))
                    hlmtmp2[(3,-1)]=np.array(hlmtmp[k])
                if k=="3":
                    modes_used_new.append((3,2))
                    hlmtmp2[(3,2)]=np.array(hlmtmp[k])
                if k=="-3":
                    modes_used_new.append((3,-2))
                    hlmtmp2[(3,-2)]=np.array(hlmtmp[k])
                if k=="4":
                    modes_used_new.append((3,3))
                    hlmtmp2[(3,3)]=np.array(hlmtmp[k])
                if k=="-4":
                    modes_used_new.append((3,-3))
                    hlmtmp2[(3,-3)]=np.array(hlmtmp[k])
                if k=="30":
                    modes_used_new.append((3,0))
                    hlmtmp2[(3,0)]=np.array(hlmtmp[k])
                if k=="5":
                    modes_used_new.append((4,1))
                    hlmtmp2[(4,1)]=np.array(hlmtmp[k])
                if k=="-5":
                    modes_used_new.append((4,-1))
                    hlmtmp2[(4,-1)]=np.array(hlmtmp[k])
                if k=="6":
                    modes_used_new.append((4,2))
                    hlmtmp2[(4,2)]=np.array(hlmtmp[k])
                if k=="-6":
                    modes_used_new.append((4,-2))
                    hlmtmp2[(4,-2)]=np.array(hlmtmp[k])
                if k=="7":
                    modes_used_new.append((4,3))
                    hlmtmp2[(4,3)]=np.array(hlmtmp[k])
                if k=="-7":
                    modes_used_new.append((4,-3))
                    hlmtmp2[(4,-3)]=np.array(hlmtmp[k])
                if k=="8":
                    modes_used_new.append((4,4))
                    hlmtmp2[(4,4)]=np.array(hlmtmp[k])
                if k=="-8":
                    modes_used_new.append((4,-4))
                    hlmtmp2[(4,-4)]=np.array(hlmtmp[k])
                if k=="40":
                    modes_used_new.append((4,0))
                    hlmtmp2[(4,0)]=np.array(hlmtmp[k])
            modes_used=modes_used_new
            print(modes_used,hlmtmp,hlmtmp2)
#        for count,mode in enumerate(modes_used):
#            hlmtmp2[mode]=np.array(hlmtmp[str(count)])
        check_if_only_positive_m = False  
        # check : if TEOBResumS only returns modes with m>=0, it is assuming reflection symmetry!  So impose it
        mode_keys = np.array([[l,m] for l,m in hlmtmp2.keys()])[:,1]
        check_if_only_positive_m = not( (mode_keys < 0).any())
        for mode in modes_used:
            hlmtmp2[mode][0]*=(m_total_s/distance_s)*nu
            hlm[mode] = lal.CreateCOMPLEX16TimeSeries("Complex hlm(t)", hlmepoch, 0,
                                                      P.deltaT, lsu_DimensionlessUnit, hlmlen)
            hlm[mode].data.data = (hlmtmp2[mode][0] * np.exp(-1j*(mode[1]*(np.pi/2.)+hlmtmp2[mode][1])))
            if not (P.deltaF is None):
                TDlen = int(1./P.deltaF * 1./P.deltaT)
                print("TDlen: ", TDlen, "data length: ", hlm[mode].data.length)
                if TDlen < hlm[mode].data.length:
#                    print("TDlen < hlm[mode].data.length: need to increase segment length; Instead Truncating from left!")
#                    sys.exit()
                    hlm[mode] = lal.ResizeCOMPLEX16TimeSeries(hlm[mode],hlm[mode].data.length-TDlen,TDlen)
                elif TDlen >= hlm[mode].data.length:
                    hlm[mode] = lal.ResizeCOMPLEX16TimeSeries(hlm[mode],0,TDlen)
            if check_if_only_positive_m or (np.abs(P.s1x) <  1e-4 and P.s2x == 0.0 and P.s1y == 0.0 and P.s2y == 0.0):
                print("Conjugating modes")
                mode_conj = (mode[0],-mode[1])
                if not mode_conj in hlm:
                    hC = hlm[mode]
                    hC2 = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hC.epoch, hC.f0,
                                                        hC.deltaT, lsu_DimensionlessUnit, hC.data.length)
                    hC2.data.data = (-1.)**mode[0] * np.conj(hC.data.data) # h(l,-m) = (-1)^ell hlm^* for reflection symmetry
                    hlm[mode_conj] = hC2

        # Create a taper, matching exactly what is used in hoft
        hp = lal.CreateREAL8TimeSeries('junk',
                    lal.LIGOTimeGPS(0.), 1., P.deltaT,
                    lsu_DimensionlessUnit, len(hlm[(2,2)].data.data ) )
        hp.data.data = np.ones(len(hp.data.data))
        lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
        # apply taper to all modes
        for mode in hlm:
            hlm[mode].data.data*= hp.data.data

        return hlm
    else: # (P.approx == lalSEOBv4 or P.approx == lalsim.SEOBNRv2 or P.approx == lalsim.SEOBNRv1 or  P.approx == lalsim.EOBNRv2 
        extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)
        # Format about to change: should not have several of these parameters
        hlms = lalsim.SimInspiralTDModesFromPolarizations( \
            P.m1, P.m2, \
            P.s1x, P.s1y, P.s1z, \
            P.s2x, P.s2y, P.s2z, \
            P.dist, P.phiref,  \
            P.psi, P.eccentricity, P.meanPerAno, \
            P.deltaT, P.fmin, P.fref, \
            extra_params, P.approx)


    # FIXME: Add ability to taper
    # COMMENT: Add ability to generate hlmoft at a nonzero GPS time directly.
    #      USUALLY we will use the hlms in template-generation mode, so will want the event at zero GPS time

    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, 2, 2)
        # Consider modifing TD behavior to be consistent with FD behavior used to match LI
        if TDlen >= hxx.data.length:
            hlms = lalsim.ResizeSphHarmTimeSeries(hlms, 0, TDlen)

    hlm_dict = SphHarmTimeSeries_to_dict(hlms,Lmax)


    # positive m only check: should never happen
    # ...but if it does (for NRSur lalsuite interface), perform reflection.
    approx_string = P.approx
    if not(isinstance(approx_string,str)):
        approx_string = lalsim.GetStringFromApproximant(approx_string)
    if 'NRSur' in approx_string:
        # some NRSur are aligned only and return only m>=0 modes, so reflect IF NEEDED
        mode_keys = np.array([[l,m] for l,m in hlm_dict.keys()])[:,1]
        check_if_only_positive_m = not((mode_keys < 0).any())
        if check_if_only_positive_m:
            mode_keys2 = list(hlm_dict.keys())
            for mode  in mode_keys2:
                mode_conj = (mode[0],-mode[1])
                if not mode_conj in hlm_dict:  # sanity check
                    hC = hlm_dict[mode]
                    hC2 = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hC.epoch, hC.f0,
                                                        hC.deltaT, lsu_DimensionlessUnit, hC.data.length)
                    hC2.data.data = (-1.)**mode[0] * np.conj(hC.data.data) # h(l,-m) = (-1)^ell hlm^* for reflection symmetry
                    hlm_dict[mode_conj] = hC2

    for mode in hlm_dict:
        hlm_dict[mode].data.data *= sign_factor
        if phiref_shift_convention:
            hlm_dict[mode].data.data *= np.exp(1j*mode[1]*phiref_shift_convention)*sign_factor # double-count

        # Force waveform duration to fit inside target time!  (SimInspiralTD adds a lot of padding)
        if not (P.deltaF is None):  # lalsim.SimInspiralImplementedFDApproximants(P.approx)==1 and 
            TDlen = int(1./P.deltaF * 1./P.deltaT)
            if TDlen < hlm_dict[mode].data.length:  # we have generated too long a signal!...truncate from LEFT. Danger!
                    hlm_dict[mode] = lal.ResizeCOMPLEX16TimeSeries(hlm_dict[mode],hlm_dict[mode].data.length-TDlen,TDlen)

    # Tapering: applies to cases without direct return, like TDmodesFromPolarizations and ChooseTDModes
    if not(no_condition):
        fmin_effective = P.fmin
        if not(fmin_effective):
            fmin_effective  = fmin_backstop
        # at this point we know the waveform length!
        TDlen_here = hlm_dict[mode].data.length
        # Base taper, based on 1% of waveform length
        ntaper = int(0.01*TDlen_here)  # fixed 1% of waveform length, at start
        ntaper = np.max([ntaper, int(1./(fmin_effective*P.deltaT)) ])  # require at least one waveform cycle of tapering; should never happen
        vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
        # Taper at the start of the segment
        for mode in hlm_dict:
            hlm_dict[mode].data.data[:ntaper]*=vectaper


    return hlm_dict   # note data type is different than with SEOB; need to finish port to pure dictionary

def hlmoft_FromFD_dict(P,Lmax=2):
    """
    Uses Chris Pankow's interface in lalsuite
    Do not redshift the source
    """
#    hlm_struct = lalsim.SimInspiralTDModesFromPolarizations(P.deltaT, P.m1, P.m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 0., P.lambda1, P.lambda2, P.waveFlags, None, P.ampO, P.phaseO, P.approx)
    extra_params = P.to_lal_dict()
        # Format about to change: should not have several of these parameters
    hlms = lalsim.SimInspiralTDModesFromPolarizations( \
            P.m1, P.m2, \
            P.s1x, P.s1y, P.s1z, \
            P.s2x, P.s2y, P.s2z, \
            P.dist, P.phiref,  \
            P.psi, P.eccentricity, P.meanPerAno, \
            P.deltaT, P.fmin, P.fref, \
            extra_params, P.approx)


    return hlms

def hlmoft_SEOBv3_dict(P,Lmax=2):
    """
    Generate the TD h_lm -2-spin-weighted spherical harmonic modes of a GW
    with parameters P. Returns a dictionary of modes.
    A hack for SEOBNRv3, because there's not a natural dictionary output

    Will NOT perform the -1 factor correction
    """

    ampFac = (P.m1 + P.m2)/lal.MSUN_SI * lal.MRSUN_SI / P.dist

    # inc is not consistent with the modern convention I will be reading in (spins aligned with L, hlm in the L frame)
    PrecEOBversion=300 # use opt
    hplus, hcross, dynHi, hlmPTS, hlmPTSHi, hIMRlmJTSHi, hLM, attachP = lalsim.SimIMRSpinEOBWaveformAll(0, P.deltaT, \
                                            P.m1, P.m2, P.fmin, P.dist, 0, \
                                            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, PrecEOBversion)
    hlm_dict = SphHarmTimeSeries_to_dict(hLM,2)
    # for j in range(5):
    #     m = hLM.m
    #     hlm_dict[(l,m)]  = hLM.mode
    #     hLM= hLM.next
    #my_epoch = - P.deltaT*np.argmax(np.abs(hlm_dict[(2,2)].data.data)**2 + np.abs(hlm_dict[(2,1)].data.data)**2 + np.abs(hlm_dict[(2,0)].data.data)**2 )  # find the event time in the data
    for key in hlm_dict:
        # Amplitude
        hlm_dict[key].data.data *= ampFac
        # epoch
        hlm_dict[key].epoch = hplus.epoch  # set the event as usual : t=0 corresponds to the time of the event

    return hlm_dict

def hlmoft_SEOB_dict(P,Lmax=2):
    r"""
    Generate the TD h_lm -2-spin-weighted spherical harmonic modes of a GW
    with parameters P. Returns a dictionary of modes.
    Just for SEOBNRv2 SEOBNRv1, and EOBNRv2.  Uses aligned-spin trick to get (2,2) and (2,-2) modes.
    A hack.
    Works for any aligned-spin time-domain waveform with only (2,\pm 2) modes though.
    """

    if P.approx == lalSEOBNRv4HM:
        extra_params = P.to_lal_dict()        
        nqcCoeffsInput=lal.CreateREAL8Vector(10)
        hlm_struct, dyn, dynHi = lalsim.SimIMRSpinAlignedEOBModes(P.deltaT, P.m1, P.m2, P.fmin, P.dist, P.s1z, P.s2z,41, 0., 0., 0.,0.,0.,0.,0.,0.,1.,1.,nqcCoeffsInput, 0)

        hlms = SphHarmTimeSeries_to_dict(hlm_struct,Lmax)
        mode_list_orig = list(hlms.keys())  # force a cast and therefore a copy - some python versions not playing safely here
        for mode in mode_list_orig:
            # Add zero padding if requested time period too short
            if not (P.deltaF is None):
                TDlen = int(1./P.deltaF * 1./P.deltaT)
                if TDlen > hlms[mode].data.length:
                    hlms[mode] = lal.ResizeCOMPLEX16TimeSeries(hlms[mode],0,TDlen)

            # Should only populate positive modes; create negative modes
            mode_conj = (mode[0],-mode[1])
            if not mode_conj in hlms:
                hC = hlms[mode]
                hC2 = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hC.epoch, hC.f0, 
                                                    hC.deltaT, lsu_DimensionlessUnit, hC.data.length)
                hC2.data.data = (-1.)**mode[0] * np.conj(hC.data.data) # h(l,-m) = (-1)^ell hlm^* for reflection symmetry
#                hT = hlms[mode].copy() # hopefully this works
#                hT.data.data = np.conj(hT.data.data)
                hlms[mode_conj] = hC2
        return hlms

    if not (P.approx == lalsim.SEOBNRv2 or P.approx==lalsim.SEOBNRv1 or P.approx == lalSEOBv4 or P.approx == lalsim.SEOBNRv4_opt or P.approx==lalsim.EOBNRv2 or P.approx == lalTEOBv2 or P.approx==lalTEOBv4):
        return None

    # Remember, we have a fiducial orientation for the h22. 
    # WARNING THIS MAY BE DESTRUCTIVE
#    P2 = P.manual_copy()
    P.phiref=0.
    P.psi=0.
    P.incl = 0
    hC = complex_hoft(P)  # pad as needed
    hC.epoch = hC.epoch - P.tref  # need to CORRECT the event time: hoft adds an epoch
    if rosDebugMessagesContainer[0]:
        print( " SEOB hlm trick: epoch of hC ", hC.epoch)
    fac = np.sqrt(5./np.pi)/2
    hC.data.data *=1./fac #lal.SpinWeightedSphericalHarmonic(0.,0., -2,2,2)

    # Copy...but I don't trust lalsuite copy
    hC2 = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hC.epoch, hC.f0, 
            hC.deltaT, lsu_DimensionlessUnit, hC.data.length)
    hC2.data.data =np.conj(hC.data.data)

    hlm_dict={}
    hlm_dict[(2,2)] = hC
    hlm_dict[(2,-2)] = hC2

    return hlm_dict


def hlmoft_IMRPv2_dict(P,sgn=-1):
    """
    Reconstruct hlm(t) from h(t,nhat) evaluated in 5 different nhat directions, specifically for IMRPhenomPv2
    Using SimInspiralTD evaluated at nhat=zhat  (zhat=Jhat), nhat=-zhat, and at three points in the equatorial plane.
    There is an analytic solution, but for code transparency I do the inverse numerically rather than code it in

    ISSUES
      - lots of instability in recovery depending on point set used (e.g., degeneracy in aligned spin case). Need to choose intelligently
      - duration of hlm is *longer* than deltaF (because of SimInspiralTD) in general.
    """

#<< RotationAndHarmonics`SpinWeightedHarmonics`
#expr = Sum[  SpinWeightedSphericalHarmonicY[-2, 2, m, th, ph] h[m], {m, -2, 2}]
#vecHC = Map[(expr /. {th -> #[[1]], ph -> #[[2]]}) &, { {0, 0}, {Pi,     0}, {Pi/2, 0}, {Pi/2, Pi/2}, {Pi/2, 3 Pi/2}}]
#mtx = Table[Coefficient[vecHC, h[m]], {m, -2, 2}] // Simplify
#Inverse[mtx] // Simplify

# exprHere =   expr /. {h[1] -> 0, h[-1] -> 0, h[0] -> 0, 
#     h[2] -> Exp[-I 2 \[CapitalPhi]], 
#     h[-2] -> Exp[I 2 \[CapitalPhi]]} // Simplify
# vals = Map[(exprHere /. {th -> #[[1]], ph -> -#[[2]]}) &, {{Pi, 
#     0}, {Pi/2, 0}, {Pi/2, 2 Pi/3}, {Pi/2, 4 Pi/3}, {0, 0}}]
# Transpose[imtx].vals // N // FullSimplify // Chop

    mtxAngularForward = np.zeros((5,5),dtype=np.complex64)
    # Organize points so the (2,-2) and (2,2) mode clearly have contributions only from one point
    # Problem with points in equatorial plane: pure real signal in aligned-spin limit! Degeneracy!
#    paramsForward =[ [np.pi,0], [np.pi/2,0], [np.pi/2,np.pi/2], [np.pi/2, 3*np.pi/2], [0,0]];
#    paramsForward =[ [np.pi,0], [np.pi/2,0], [np.pi/2, np.pi/3], [np.pi/2, 2*np.pi/3], [0,0]];
    paramsForward =[ [np.pi,0], [np.pi/3,0], [np.pi/2, 0], [2*np.pi/3, 0], [0,0]];
#    paramsForward =[ [np.pi,0], [np.pi/2,0], [np.pi/4, 0],[3*np.pi/4,np.pi], [0,0]];
#    paramsForward =[ [np.pi,0], [np.pi/4,np.pi/4], [np.pi/2, np.pi/2], [3*np.pi/4,3*np.pi/2],[0,np.pi]];
    mvals = [-2,-1,0,1,2]
    for indx in np.arange(len(paramsForward)):
        for indx2 in np.arange(5):
            m = mvals[indx2]
            th,ph = paramsForward[indx]
            mtxAngularForward[indx,indx2] = lal.SpinWeightedSphericalHarmonic(th,-ph,-2,2,m)  # note phase sign
    mtx = np.linalg.inv(mtxAngularForward)

#    print np.around(mtx,decimals=5)

    # Now generate solutions at these values
    P_copy = P.manual_copy()
    P_copy.tref =0  # we do not need or want this offset when constructing hlm
    P_copy.psi = 0 # we want to force a certain polarization
    # Force rouding of tiny transverse spins, to avoid numerical problems associated with the way the PhenomP code defines orientations
    P_copy.s1x = int(P.s1x*1e4)/1.e4
    P_copy.s2x = int(P.s2x*1e4)/1.e4
    P_copy.s1y = int(P.s1y*1e4)/1.e4
    P_copy.s2y = int(P.s2y*1e4)/1.e4

    hTC_list = []
    for indx in np.arange(len(paramsForward)):
        th,ph = paramsForward[indx]
#        P_copy.assign_param('thetaJN', th)  # to be CONSISTENT with h(t) produced by ChooseTD, need to use incl here (!!)
        P_copy.incl = th  # to be CONSISTENT with h(t) produced by ChooseTD, need to use incl here (!!)
        P_copy.phiref = ph
        # Note argument change!  Hack to recover reasonable hlm modes for (2,+/-1), (2,0)
        # Something is funny about how SimInspiralFD/etc applies all these coordinate transforms
        if P_copy.fref ==0:
            P_copy.fref = P_copy.fmin
#        if np.abs(np.cos(P_copy.incl))<1:
#            P_copy.phiref =  P_copy.phiref
#            P_copy.psi = P_copy.phiref # polarization angle is set by emission direction, physically

        hTC = complex_hoft_IMRPv2(P_copy)
        hTC_list.append(hTC)

    # Now construct the hlm from this sequence
    hlmT = {}
    for indx2 in np.arange(5):
        m = mvals[indx2]
        hlmT[(2,m)] = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hTC.epoch, hTC.f0, 
            hTC.deltaT, lsu_DimensionlessUnit, hTC.data.length)
        hlmT[(2,m)].epoch = float(hTC_list[0].epoch)
        hlmT[(2,m)].data.data *=0   # this is needed, memory is not reliably cleaned
        for indx in np.arange(len(paramsForward)):
            hlmT[(2,m)].data.data += mtx[indx2,indx] * hTC_list[indx].data.data
    
    return hlmT

def hlmoff(P, Lmax=2):
    """
    Generate the FD h_lm -2-spin-weighted spherical harmonic modes of a GW
    with parameters P. Returns a SphHarmTimeSeries, a linked-list of modes with
    a COMPLEX16TimeSeries and indices l and m for each node.

    The linked list will contain all modes with l <= Lmax
    and all values of m for these l.
    """

    hlms = hlmoft(P, Lmax)
    if isinstance(hlms,dict):
        hlmsF = {}
        for mode in hlms:
            hlmsF[mode] = DataFourier(hlms[mode])
        return hlmsF
    hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, 2, 2)
    if P.deltaF == None: # h_lm(t) was not zero-padded, so do it now
        TDlen = nextPow2(hxx.data.length)
        hlms = lalsim.ResizeSphHarmTimeSeries(hlms, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == hxx.data.length

    # FFT the hlms
    Hlms = lalsim.SphHarmFrequencySeriesFromSphHarmTimeSeries(hlms)

    return Hlms

def hlmoft_SpinTaylorManual_dict(P,Lmax=2):
    """
    Generate the TD h_lm -2-spin-weighted spherical harmonic modes of a GW
    with parameters P. Returns a dictionary of modes.
    Just for SpinTaylor, using manual interface provided by Riccardo and Jake.

    """

    dictParams = P.to_lal_dict()
    P.approx = approx = lalsim.GetApproximantFromString("SpinTaylorT4")
    modearray=lalsim.SimInspiralCreateModeArray()
    
    #Construct array with modes you want (based off Lmax given)
    modes_used = []
    for l in np.arange(2,Lmax+1,1):
        for m in np.arange(-l,l+1,1):
            modes_used.append([l,m])
            
    for mode in modes_used:
        lalsim.SimInspiralModeArrayActivateMode(modearray, mode[0], mode[1])
        lalsim.SimInspiralModeArrayActivateMode(modearray, mode[0], -mode[1])
        
        #Get hlms thanks to Riccardo
        hp, hx, V, Phi, S1x, S1y, S1z, S2x, S2y, S2z, LNx, LNy, LNz, E1x, E1y, E1z = lalsim.SimInspiralSpinTaylorDriver(P.phiref, P.deltaT, P.m1, P.m2, P.fmin, P.fref, P.dist, P.s1x, P.s1y,
                                                                                                                        P.s1z, P.s2x, P.s2y, P.s2z, 0, 0,1,1,0,0, dictParams, approx)
        hlm_struct = lalsim.SimInspiralSpinTaylorHlmModesFromOrbit(V, Phi, LNx, LNy, LNz, E1x, E1y, E1z,  S1x, S1y, S1z, S2x, S2y, S2z, P.m1, P.m2, P.dist, P.ampO, modearray)

    #add padding 
    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        hxx = lalsim.SphHarmTimeSeriesGetMode(hlm_struct,2,2)
        assert TDlen >= hxx.data.length
        hlm_struct = lalsim.ResizeSphHarmTimeSeries(hlm_struct,0,TDlen)

    return hlm_struct


def conj_hlmoff(P, Lmax=2):
    hlms = hlmoft(P, Lmax)
    if isinstance(hlms,dict):
        hlmsF = {}
        for mode in hlms:
            hlms[mode].data.data = np.conj(hlms[mode].data.data)
            hlmsF[mode] = DataFourier(hlms[mode])
        return hlmsF
    hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, 2, 2)
    if P.deltaF == None: # h_lm(t) was not zero-padded, so do it now
        TDlen = nextPow2(hxx.data.length)
        hlms = lalsim.ResizeSphHarmTimeSeries(hlms, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == hxx.data.length

    # Conjugate each mode before taking FFT
    for l in range(2, Lmax+1):
        for m in range(-l, l+1):
            hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, l, m)
            if hxx:
                hxx.data.data = np.conj(hxx.data.data)
    # FFT the hlms
    Hlms = lalsim.SphHarmFrequencySeriesFromSphHarmTimeSeries(hlms)

    return Hlms

def std_and_conj_hlmoff(P, Lmax=2,**kwargs):
    hlms = hlmoft(P, Lmax,**kwargs)
    if isinstance(hlms,dict):
        hlmsF = {}
        hlms_conj_F = {}
        for mode in hlms:
            hlmsF[mode] = DataFourier(hlms[mode])
            hlms[mode].data.data = np.conj(hlms[mode].data.data)
            hlms_conj_F[mode] = DataFourier(hlms[mode])
        return hlmsF, hlms_conj_F
    hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, 2, 2)
    if P.deltaF == None: # h_lm(t) was not zero-padded, so do it now
        TDlen = nextPow2(hxx.data.length)
        hlms = lalsim.ResizeSphHarmTimeSeries(hlms, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == hxx.data.length

    # Make into dictionary, and do what we did above
    hlmsT = {}
    for l in range(2, Lmax+1):
        for m in range(-l, l+1):
            hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, l, m)
            if hxx:
                hlmsT[mode]=hxx
    hlms=hlmsT
    #
    hlmsF = {}
    hlms_conj_F = {}
    for mode in hlms:
            hlmsF[mode] = DataFourier(hlms[mode])
            hlms[mode].data.data = np.conj(hlms[mode].data.data)
            hlms_conj_F[mode] = DataFourier(hlms[mode])
    return hlmsF, hlms_conj_F

def hoft_from_hlm(hlms,P, return_complex=False, extra_phase_shift=0):
    """
    hoft_from_hlm(hlm,P,Lmax):  return hoft like output given hlmoft input.
    Important if hoft is not accessible (e.g., not provided by lalsuite)
    """
    h22 = hlms[(2,2)]

    # Create complex strain object
    hT = lal.CreateCOMPLEX16TimeSeries("hT", h22.epoch, h22.f0,
            h22.deltaT, h22.sampleUnits, h22.data.length)
    hT.data.data*=0  # fill with zeros

    # create for loop over elements of the series to add it
    for mode in hlms:
        hlm = hlms[mode]
        hT.data.data += hlm.data.data * lal.SpinWeightedSphericalHarmonic(P.incl, extra_phase_shift- P.phiref,-2,mode[0],mode[1])

    if return_complex:
        hT.data.data *= np.exp(-2*1j*P.psi)
        return hT

    # now create real valued output based on detectors   
    if P.radec==False:
        fp = Fplus(P.theta, P.phi, P.psi)
        fc = Fcross(P.theta, P.phi, P.psi)
        fp = Fplus(P.theta, P.phi, P.psi)
        fc = Fcross(P.theta, P.phi, P.psi)

        h_real = lal.CreateREAL8TimeSeries("hT", h22.epoch, h22.f0,
                                           h22.deltaT, h22.sampleUnits, h22.data.length)
        h_real.data.data =  np.real(hT.data.data)*fp + np.imag(hT.data.data)*fc
    else:
        hp = lal.CreateREAL8TimeSeries("hT", h22.epoch, h22.f0,
            h22.deltaT, h22.sampleUnits, h22.data.length)
        hc = lal.CreateREAL8TimeSeries("hT", h22.epoch, h22.f0,
            h22.deltaT, h22.sampleUnits, h22.data.length)
        hp.data.data = np.real(hT.data.data)
        hc.data.data = np.imag(hT.data.data)
        hp.epoch = hp.epoch + P.tref
        hc.epoch = hc.epoch + P.tref
        h_real = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc, 
                P.phi, P.theta, P.psi, 
                lalsim.DetectorPrefixToLALDetector(str(P.detector)))


    return h_real

def SphHarmTimeSeries_to_dict(hlms, Lmax):
    """
    Convert a SphHarmTimeSeries SWIG-wrapped linked list into a dictionary.

    The keys are tuples of integers of the form (l,m)
    and a key-value pair will be created for each (l,m) with
    2 <= l <= Lmax, |m| <= l for which
    lalsimulation.SphHarmTimeSeriesGetMode(hlms, l, m)
    returns a non-null pointer.
    """
    if isinstance(hlms, dict):
        return hlms
    hlm_dict = {}
    for l in range(2, Lmax+1):
        for m in range(-l, l+1):
            hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, l, m)
            if hxx is not None:
                hlm_dict[(l,m)] = hxx

    return hlm_dict

def SphHarmFrequencySeries_to_dict(hlms, Lmax):
    """
    Convert a SphHarmFrequencySeries SWIG-wrapped linked list into a dictionary.

    The keys are tuples of integers of the form (l,m)
    and a key-value pair will be created for each (l,m) with
    2 <= l <= Lmax, |m| <= l for which
    lalsimulation.SphHarmFrequencySeriesGetMode(hlms, l, m)
    returns a non-null pointer.
    """
    if isinstance(hlms, dict):
        return hlms
    hlm_dict = {}
    for l in range(2, Lmax+1):
        for m in range(-l, l+1):
            hxx = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
            if hxx is not None:
                hlm_dict[(l,m)] = hxx

    return hlm_dict

def complex_hoft(P, sgn=-1,**kwargs):
    """
    Generate a complex TD waveform from ChooseWaveformParams P
    Returns h(t) = h+(t) + 1j sgn hx(t)
    where sgn = -1 (default) or 1

    Returns a COMPLEX16TimeSeries object
    """
    assert sgn == 1 or sgn == -1
    # hp, hc = lalsim.SimInspiralTD(P.phiref, P.deltaT, P.m1, P.m2, 
    #         P.s1x, P.s1y, P.s1z, P.spin2x, P.spin2y, P.spin2z, P.fmin, P.fref, P.dist, 
    #         P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
    #         P.ampO, P.phaseO, P.approx)
    extra_waveform_args = {}
    if 'extra_waveform_args' in kwargs:
        extra_waveform_args.update(kwargs['extra_waveform_args'])
    extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)

    hp, hc = lalsim.SimInspiralChooseTDWaveform( P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z,
            P.dist, P.incl, P.phiref,  \
            0 , P.eccentricity, P.meanPerAno, \
            P.deltaT, P.fmin, P.fref, \
            extra_params, P.approx)
    if P.taper != lsu_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(hp.data, P.taper)
        lalsim.SimInspiralREAL8WaveTaper(hc.data, P.taper)
    if P.deltaF is not None:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        if log_loud:
            print(TDlen,hp.data.length)
        assert TDlen >= hp.data.length
        hp = lal.ResizeREAL8TimeSeries(hp, 0, TDlen)
        hc = lal.ResizeREAL8TimeSeries(hc, 0, TDlen)

    ht = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, 
            hp.deltaT, lsu_DimensionlessUnit, hp.data.length)
    ht.epoch = ht.epoch + P.tref
    ht.data.data = hp.data.data + 1j * sgn * hc.data.data
    # impose polarization directly, using precisely the conventions we demand
    ht.data.data*= np.exp(2j*sgn*P.psi)
    return ht

def complex_hoft_IMRPv2(P_copy,sgn=-1):
    """
    Generate a complex TD waveform from ChooseWaveformParams P, using SimInspiralTD
    Returns h(t) = h+(t) + 1j sgn hx(t)
    where sgn = -1 (default) or 1

    Returns a COMPLEX16TimeSeries object

    Designed specifically to help interface with IMRPhenomPv2
    """
    extra_params = P_copy.to_lal_dict()
    hp, hc = lalsim.SimInspiralTD( \
            P_copy.m1, P_copy.m2, \
            P_copy.s1x, P_copy.s1y, P_copy.s1z, \
            P_copy.s2x, P_copy.s2y, P_copy.s2z, \
            P_copy.dist, P_copy.incl, P_copy.phiref,  \
            P_copy.psi, P_copy.eccentricity, P_copy.meanPerAno, \
            P_copy.deltaT, P_copy.fmin, P_copy.fref, \
            extra_params, P_copy.approx)
    hT = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, 
            hp.deltaT, lsu_DimensionlessUnit, hp.data.length)
    hT.epoch = hT.epoch + P_copy.tref
    hT.data.data = np.real(hp.data.data) + 1j * sgn * np.real(hc.data.data) # make absolutely sure no surprises
    return hT


def complex_hoff(P, sgn=-1, fwdplan=None,**kwargs):
    """
    CURRENTLY ONLY WORKS WITH TD APPROXIMANTS

    Generate a (non-Hermitian) FD waveform from ChooseWaveformParams P
    by creating a complex TD waveform of the form

    h(t) = h+(t) + 1j sgn hx(t)    where sgn = -1 (default) or 1

    If P.deltaF==None, just pad up to next power of 2
    If P.deltaF = 1/X, will generate a TD waveform, zero-pad to length X seconds
        and then FFT. Will throw an error if waveform is longer than X seconds

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, it is best to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    Returns a COMPLEX16FrequencySeries object
    """
    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
        # Raise exception if unused arguments were specified
        if fwdplan is not None:
            raise ValueError('FFT plan fwdplan given with FD approximant.\nFD approximants cannot use this.')
        if P.deltaT*P.deltaF >0:
            TDlen = int(1./(P.deltaT*P.deltaF))
        elif TDlen!=0: # Set values of P.deltaF from TDlen, P.deltaT
            P.deltaF = 1./P.deltaT/TDlen
        extra_waveform_args = {}
        if 'extra_waveform_args' in kwargs:
            extra_waveform_args.update(kwargs['extra_waveform_args'])
        extra_params = P.to_lal_dict_extended(extra_args_dict=extra_waveform_args)
        hptilde, hctilde = lalsim.SimInspiralChooseFDWaveform(#P.phiref, P.deltaF,
            P.m1, P.m2, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z,
            P.dist, P.incl, P.phiref,  \
            P.psi, P.eccentricity, P.meanPerAno, \
            P.deltaF, P.fmin, TDlen*P.deltaF/2, P.fref, \
            extra_params, P.approx)

        if TDlen > 0:
            if P.approx != lalsim.IMRPhenomP:
                assert TDlen/2+1 >= hptilde.data.length  # validates nyqist for real-valued series
            hptilde = lal.ResizeCOMPLEX16FrequencySeries(hptilde, 0, int(TDlen/2)+1)
            hctilde = lal.ResizeCOMPLEX16FrequencySeries(hctilde, 0, int(TDlen/2)+1)


        # Pack so f=0 occurs at one side
        hoff = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            hptilde.epoch, hptilde.f0, 1./P.deltaT/TDlen, lsu_HertzUnit, 
            TDlen)

        # create the 2-sided hoff
        tmp  = hptilde.data.data + sgn*1j*hctilde.data.data
        hoff.data.data[-int(TDlen/2)-2:-1] = tmp
        tmp= np.conj(hptilde.data.data) + sgn*1j*np.conj(hctilde.data.data)
        hoff.data.data[0:int(TDlen/2)+1] = tmp[::-1]

        # Translate the wavefront to the detector, if we are not at the origin of spacetime
        # Implement by just changing 'epoch', not by any fancy frequency-domain modulation? 
        #   - NO, we want all timesamples to have regular timestamps in the geocenter. So we WILL modulate
        if P.radec==False:
            return hoff
        else:
            # return h(f) at the detector
            detector = lalsim.DetectorPrefixToLALDetector(P.detector)
            dt = lal.TimeDelayFromEarthCenter(detector.location, P.phi, P.theta, P.tref)
#            print " Translating ", P.detector, " by ", dt
            hoff.data.data *= np.exp(-2*np.pi*1j*evaluate_fvals(hoff)*dt)
            return hoff

    ht = complex_hoft(P, sgn,**kwargs)

    if P.deltaF == None: # h(t) was not zero-padded, so do it now
        TDlen = nextPow2(ht.data.length)
        ht = lal.ResizeCOMPLEX16TimeSeries(ht, 0, TDlen)
    else: # Check zero-padding was done to expected length
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen == ht.data.length

    if fwdplan==None:
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    FDlen = int(TDlen/2+1)
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lsu_HertzUnit, 
            TDlen)
    lal.COMPLEX16TimeFreqFFT(hf, ht, fwdplan)
    return hf

def complex_norm_hoff(P, IP, sgn=-1, fwdplan=None):
    """
    CURRENTLY ONLY WORKS WITH TD APPROXIMANTS

    Generate a (non-Hermitian) FD waveform from ChooseWaveformParams P
    by creating a complex TD waveform of the form

    h(t) = h+(t) + 1j sgn hx(t)    where sgn = -1 (default) or 1

    If P.deltaF==None, just pad up to next power of 2
    If P.deltaF = 1/X, will generate a TD waveform, zero-pad to length X seconds
        and then FFT. Will throw an error if waveform is longer than X seconds

    If you do not provide a forward FFT plan, one will be created.
    If you are calling this function many times, it is best to create it
    once beforehand and pass it in, e.g.:
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)

    Returns a COMPLEX16FrequencySeries object
    """
    htilde = complex_hoff(P, sgn, fwdplan)
    norm = IP.norm(htilde)
    htilde.data.data /= norm
    return htilde

#
# Functions to read an ASCII file in NINJA data format (see arXiv:0709.0093)
# and return REAL8TimeSeries or COMPLEX16FrequencySeries objects containing
# the waveform, possibly after rescaling the time and/or TD strain
#

def NINJA_data_to_hoft(fname, TDlen=-1, scaleT=1., scaleH=1., Fp=1., Fc=0.):
    """
    Function to read in data in the NINJA format, i.e.
    t_i   h+(t_i)   hx(t_i)
    and convert it to a REAL8TimeSeries holding the observed
    h(t) = Fp*h+(t) + Fc*hx(t)

    If TDlen == -1 (default), do not zero-pad the returned waveforms 
    If TDlen == 0, zero-pad returned waveforms to the next power of 2
    If TDlen == N, zero-pad returned waveforms to length N

    scaleT and scaleH can be used to rescale the time steps and strain resp.
    e.g. to get a waveform appropriate for a total mass M you would scale by
    scaleT = G M / c^3
    scaleH = G M / (D c^2)
    """
    t, hpdat, hcdat = np.loadtxt(fname, unpack=True)
    tmplen = len(t)
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    tStart = t[0]
    deltaT = (t[1] - t[0]) * scaleT

    ht = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0.,
            deltaT, lsu_DimensionlessUnit, TDlen)

    for i in range(tmplen):
        ht.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        ht.data.data[i] = 0.

    return ht

def NINJA_data_to_hp_hc(fname, TDlen=-1, scaleT=1., scaleH=1., deltaT=0):
    """
    Function to read in data in the NINJA format in file 'fname', i.e.
    t_i   h+(t_i)   hx(t_i)
    and convert it to two REAL8TimeSeries holding polarizations hp(t) and hc(t)

    If TDlen == -1 (default), do not zero-pad the returned waveforms
    If TDlen == 0, zero-pad returned waveforms to the next power of 2
    If TDlen == N, zero-pad returned waveforms to length N

    scaleT and scaleH can be used to rescale the time steps and strain resp.
    e.g. if the input file provides time steps in t/M
    and the strain has mass and distance scaled out, to get a waveform
    appropriate for a total mass M and distance D you would scale by
    scaleT = G M / c^3
    scaleH = G M / (D c^2)

    Once time is properly scaled into seconds, you can interpolate the waveform
    to a different sample rate deltaT.
    e.g. if the file has time steps of t/M = 1 and you use scaleT to rescale
    for a 100 Msun binary, the time steps will be ~= 0.00049 s.
    If you provide the argument deltaT = 1./4096. ~= 0.00024 s
    the waveform will be interpolated and resampled at 4096 Hz.

    If deltaT==0, then no interpolation will be done

    NOTE: For improved accuracy, we convert
    h+ + i hx --> A e^(i Phi),
    interpolate and resample A and Phi, then convert back to h+, hx
    """
    t, hpdat, hcdat = np.loadtxt(fname, unpack=True)
    tmplen = len(t)
    tStart = t[0] * scaleT
    deltaT = (t[1] - t[0]) * scaleT
    hpdat *= scaleH
    hcdat *= scaleH

    if deltaT==0: # No need to interpolate or resample
        if TDlen == -1:
            TDlen = tmplen
        elif TDlen==0:
            TDlen = nextPow2(tmplen)
        else:
            assert TDlen >= tmplen

        hp = lal.CreateREAL8TimeSeries("hplus(t)", lal.LIGOTimeGPS(tStart),
                0., deltaT, lsu_DimensionlessUnit, TDlen)
        hc = lal.CreateREAL8TimeSeries("hcross(t)", lal.LIGOTimeGPS(tStart),
                0., deltaT, lsu_DimensionlessUnit, TDlen)

        for i in range(tmplen):
            hp.data.data[i] = hpdat[i]
            hc.data.data[i] = hcdat[i]
        for i in range(tmplen,TDlen):
            hp.data.data[i] = 0.
            hc.data.data[i] = 0.
        return hp, hc

    else: # do interpolation and resample at rate deltaT
        assert deltaT > 0
        times = tStart + np.arange(tmplen) * deltaT
        newlen = np.floor( (tmplen-1) * deltaT / deltaT)
        newtimes = tStart + np.arange(newlen) * deltaT 
        newlen = len(newtimes)
        if TDlen == -1:
            TDlen = newlen
        elif TDlen==0:
            TDlen = 1
            while TDlen < newlen:
                TDlen *= 2
        else:
            assert TDlen >= newlen
        hp = lal.CreateREAL8TimeSeries("hplus(t)", lal.LIGOTimeGPS(tStart),
                0., deltaT, lsu_DimensionlessUnit, TDlen)
        hc = lal.CreateREAL8TimeSeries("hcross(t)", lal.LIGOTimeGPS(tStart),
                0., deltaT, lsu_DimensionlessUnit, TDlen)

        # build complex waveform, cubic spline interpolate amp and phase
        hcmplx = hpdat + 1j * hcdat
        amp = np.abs(hcmplx)
        phase = unwind_phase( np.angle(hcmplx) )
        ampintp = interpolate.InterpolatedUnivariateSpline(times, amp, k=3)
        phaseintp = interpolate.InterpolatedUnivariateSpline(times, phase, k=3)
        # Resample interpolated waveform, convert back to hp, hc
        hcmplxnew = ampintp(newtimes) * np.exp(1j * phaseintp(newtimes) )
        hpnew = np.real(hcmplxnew)
        hcnew = np.imag(hcmplxnew)
        for i in range(newlen):
            hp.data.data[i] = hpnew[i]
            hc.data.data[i] = hcnew[i]
        for i in range(newlen,TDlen):
            hp.data.data[i] = 0.
            hc.data.data[i] = 0.
        return hp, hc


def NINJA_data_to_hoff(fname, TDlen=0, scaleT=1., scaleH=1., Fp=1., Fc=0.):
    """
    Function to read in data in the NINJA format, i.e.
    t_i   h+(t_i)   hx(t_i)
    and convert it to a COMPLEX16FrequencySeries holding the observed
    h(f) = FFT[ Fp*h+(t) + Fc*hx(t) ]

    If TDlen == -1, do not zero-pad the TD waveform before FFTing
    If TDlen == 0 (default), zero-pad the TD waveform to the next power of 2
    If TDlen == N, zero-pad the TD waveform to length N before FFTing

    scaleT and scaleH can be used to rescale the time steps and strain resp.
    e.g. to get a waveform appropriate for a total mass M you would scale by
    scaleT = G M / c^3
    scaleH = G M / (D c^2)
    """
    t, hpdat, hcdat = np.loadtxt(fname, unpack=True)
    tmplen = len(t)
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    tStart = t[0]
    deltaT = (t[1] - t[0]) * scaleT

    ht = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0.,
            deltaT, lsu_DimensionlessUnit, TDlen)

    for i in range(tmplen):
        ht.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        ht.data.data[i] = 0.

    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("h(f)", 
            ht.epoch, ht.f0, 1./deltaT/TDlen, lsu_HertzUnit, 
            TDlen/2+1)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    return hf

def NINJA_data_to_norm_hoff(fname, IP, TDlen=0, scaleT=1., scaleH=1.,
        Fp=1., Fc=0.):
    """
    Function to read in data in the NINJA format, i.e.
    t_i   h+(t_i)   hx(t_i)
    and convert it to a COMPLEX16FrequencySeries holding
    h(f) = FFT[ Fp*h+(t) + Fc*hx(t) ]
    normalized so (h|h)=1 for inner product IP

    If TDlen == -1, do not zero-pad the TD waveform before FFTing
    If TDlen == 0 (default), zero-pad the TD waveform to the next power of 2
    If TDlen == N, zero-pad the TD waveform to length N before FFTing

    scaleT and scaleH can be used to rescale the time steps and strain resp.
    e.g. to get a waveform appropriate for a total mass M you would scale by
    scaleT = G M / c^3
    scaleH = G M / (D c^2)
    """
    t, hpdat, hcdat = np.loadtxt(fname, unpack=True)
    tmplen = len(t)
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    tStart = t[0]
    deltaT = (t[1] - t[0]) * scaleT

    ht = lal.CreateREAL8TimeSeries("h(t)", lal.LIGOTimeGPS(tStart), 0.,
            deltaT, lsu_DimensionlessUnit, TDlen)

    for i in range(tmplen):
        ht.data.data[i] = (Fp*hpdat[i] + Fc*hcdat[i]) * scaleH
    for i in range(tmplen,TDlen):
        ht.data.data[i] = 0.

    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("h(f)", 
            ht.epoch, ht.f0, 1./deltaT/TDlen, lsu_HertzUnit, 
            TDlen/2+1)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    norm = IP.norm(hf)
    hf.data.data /= norm
    return hf

import lalframe
def hoft_to_frame_data(fname, channel, hoft):
    """
    Function to write h(t) to a frame file in a specific channel.
    NOT YET DEBUGGED
    """
    epoch = hoft.epoch
    duration = hoft.deltaT*hoft.data.length
    hoft.name = channel  # used by low-level routines

    # Dump to frame, as FAKE-STRAIN
    # detectorFlags =   lal.LHO_4K_DETECTOR_BIT | lal.LLO_4K_DETECTOR_BIT
    frame = lalframe.FrameNew(epoch, duration, "LIGO",0 , 0, 0)
    lalframe.FrameAddREAL8TimeSeriesProcData(frame, hoft) 
    lalframe.FrameWrite(frame, fname)
    return True


def frame_data_to_hoft_old(fname, channel, start=None, stop=None, window_shape=0.,
        verbose=True):
    """
    Function to read in data in the frame format and convert it to 
    a REAL8TimeSeries. fname is the path to a LIGO cache file.

    Applies a Tukey window to the data with shape parameter 'window_shape'.
    N.B. if window_shape=0, the window is the identity function
         if window_shape=1, the window becomes a Hann window
         if 0<window_shape<1, the data will transition from zero to full
            strength over that fraction of each end of the data segment.
    """
    if verbose:
        print( " ++ Loading from cache ", fname, channel)
    with open(fname) as cfile:
        cachef = Cache.fromfile(cfile)
    for i in range(len(cachef))[::-1]:
        # FIXME: HACKHACKHACK
        if cachef[i].observatory != channel[0]:
            del cachef[i]
    if verbose:
        print( cachef.to_segmentlistdict())

    import os
    tmpdir = None
    if 'TMP' in os.environ:
        tmpdir  = os.environ['TMP']
    elif 'TMPDIR' in os.environ:
        tmpdir = os.environ['TMPDIR']
    else:
        tmpdir = '/tmp'
    fcache = frutils.FrameCache(cachef,scratchdir=tmpdir)

#    fcache = frutils.FrameCache(cachef)
    # FIXME: Horrible, horrible hack -- will only work if all requested channels
    # span the cache *exactly*
    if start is None:
        start = cachef.to_segmentlistdict()[channel[0]][0][0]
    if stop is None:
        stop = cachef.to_segmentlistdict()[channel[0]][-1][-1]
    
    ht = fcache.fetch(channel, start, stop)
        
    tmp = lal.CreateREAL8TimeSeries("h(t)", 
            lal.LIGOTimeGPS(float(ht.metadata.segments[0][0])),
            0., ht.metadata.dt, lsu_DimensionlessUnit, len(ht))
    print(   "  ++ Frame data sampling rate ", 1./tmp.deltaT, " and epoch ", stringGPSNice(tmp.epoch))
    tmp.data.data[:] = ht
    # Window the data - N.B. default is identity (no windowing)
    hoft_window = lal.CreateTukeyREAL8Window(tmp.data.length, window_shape)
    tmp.data.data *= hoft_window.data.data

    return tmp

def frame_data_to_hoft(fname, channel, start=None, stop=None, window_shape=0.,
        verbose=True,deltaT=None):
    """
    Function to read in data in the frame format and convert it to 
    a REAL8TimeSeries. fname is the path to a LIGO cache file.

    Applies a Tukey window to the data with shape parameter 'window_shape'.
    N.B. if window_shape=0, the window is the identity function
         if window_shape=1, the window becomes a Hann window
         if 0<window_shape<1, the data will transition from zero to full
            strength over that fraction of each end of the data segment.

    Modified to rely on the lalframe read_timeseries function
      https://github.com/lscsoft/lalsuite/blob/master/lalframe/python/lalframe/frread.py
    """
    if verbose:
        print( " ++ Loading from cache ", fname, channel)
    with open(fname) as cfile:
        cachef = Cache.fromfile(cfile)
    cachef=cachef.sieve(ifos=channel[:1])
    # for i in range(len(cachef))[::-1]:
    #     # FIXME: HACKHACKHACK
    #     if cachef[i].observatory != channel[0]:
    #         del cachef[i]
    if verbose:
        print( cachef.to_segmentlistdict())
        
    duration = stop - start if None not in (start, stop) else None
    try:
        tmp = frread.read_timeseries(cachef, channel, start=start,duration=duration,verbose=verbose,datatype='REAL8')
    except Exception as fail:
        if str(fail) == "RuntimeError: Failure in an XLAL routine":
            print(f"Encountered {fail}")
            sys.exit(91)
        else:
            print(fail)
            sys.exit(1)
    # Window the data - N.B. default is identity (no windowing)
    hoft_window = lal.CreateTukeyREAL8Window(tmp.data.length, window_shape)
    tmp.data.data *= hoft_window.data.data

    # Resample the timeries as requested
    if (not (deltaT is None)) and deltaT > tmp.deltaT:
        lal.ResampleREAL8TimeSeries(tmp,deltaT)

    return tmp


def frame_data_to_hoff(fname, channel, start=None, stop=None, TDlen=0,
        window_shape=0., verbose=True):
    """
    Function to read in data in the frame format
    and convert it to a COMPLEX16FrequencySeries holding
    h(f) = FFT[ h(t) ]

    Before the FFT, applies a Tukey window with shape parameter 'window_shape'.
    N.B. if window_shape=0, the window is the identity function
         if window_shape=1, the window becomes a Hann window
         if 0<window_shape<1, the data will transition from zero to full
            strength over that fraction of each end of the data segment.

    If TDlen == -1, do not zero-pad the TD waveform before FFTing
    If TDlen == 0 (default), zero-pad the TD waveform to the next power of 2
    If TDlen == N, zero-pad the TD waveform to length N before FFTing
    """
    ht = frame_data_to_hoft(fname, channel, start, stop, window_shape, verbose)

    tmplen = ht.data.length
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    for i in range(tmplen,TDlen):
        ht.data.data[i] = 0.

    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("h(f)", 
            ht.epoch, ht.f0, 1./deltaT/TDlen, lsu_HertzUnit, 
            TDlen/2+1)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    return hf


def frame_data_to_non_herm_hoff(fname, channel, start=None, stop=None, TDlen=0,
        window_shape=0., verbose=True,deltaT=None):
    """
    Function to read in data in the frame format
    and convert it to a COMPLEX16FrequencySeries 
    h(f) = FFT[ h(t) ]
    Create complex FD data that does not assume Hermitianity - i.e.
    contains positive and negative freq. content

    Before the FFT, applies a Tukey window with shape parameter 'window_shape'.
    N.B. if window_shape=0, the window is the identity function
         if window_shape=1, the window becomes a Hann window
         if 0<window_shape<1, the data will transition from zero to full
            strength over that fraction of each end of the data segment.

    If TDlen == -1, do not zero-pad the TD waveform before FFTing
    If TDlen == 0 (default), zero-pad the TD waveform to the next power of 2
    If TDlen == N, zero-pad the TD waveform to length N before FFTing
    """
    ht = frame_data_to_hoft(fname, channel, start, stop, window_shape, verbose,deltaT=deltaT)

    tmplen = ht.data.length
    if TDlen == -1:
        TDlen = tmplen
    elif TDlen==0:
        TDlen = nextPow2(tmplen)
    else:
        assert TDlen >= tmplen

    ht = lal.ResizeREAL8TimeSeries(ht, 0, TDlen)
    hoftC = lal.CreateCOMPLEX16TimeSeries("h(t)", ht.epoch, ht.f0,
            ht.deltaT, ht.sampleUnits, TDlen)
    # copy h(t) into a COMPLEX16 array which happens to be purely real
    hoftC.data.data = ht.data.data + 0j
    FDlen = TDlen
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lsu_HertzUnit,
            FDlen)
    lal.COMPLEX16TimeFreqFFT(hf, hoftC, fwdplan)
    if verbose:
        print( " ++ Loaded data h(f) of length n= ", hf.data.length, " (= ", len(hf.data.data)*ht.deltaT, "s) at sampling rate ", 1./ht.deltaT    )
    return hf


def stringGPSNice(tgps):
    """
    Return a string with nice formatting displaying the value of a LIGOTimeGPS
    """
    if isinstance(tgps, float):
        return str(tgps)
    elif isinstance(tgps, lal.LIGOTimeGPS):
        return "%d.%d" % (tgps.gpsSeconds, tgps.gpsNanoSeconds)
    else:
        return "FAILED"

def pylal_psd_to_swig_psd(raw_pylal_psd):
    """
    pylal_psd_to_swig_psd
    Why do I do a conversion? I am having trouble returning modified PSDs
    """
    data = raw_pylal_psd.data
    df = raw_pylal_psd.deltaF
    psdNew = lal.CreateREAL8FrequencySeries("PSD", lal.LIGOTimeGPS(0.), 0., df ,lsu_HertzUnit, len(data))
    for i in range(len(data)):
        psdNew.data.data[i] = data[i]   # don't mix memory management between pylal and swig
    return psdNew

def regularize_swig_psd_series_near_nyquist(raw_psd,DeltaFToZero):
    """
    regularize_psd_series_near_nyquist
    Near nyquist, some PSD bins are insane.  As a *temporary measure*, I am going to use this routine to
    explicitly zero out those frequency bins, using a 30 Hz window near nyquist
    """
    global rosDebugMessagesContainer
    df = raw_psd.deltaF
    nToZero = int(1.0* DeltaFToZero/df)
    new_psd = raw_psd # copy.deepcopy(raw_psd)  # I actually don't need a copy
    n = len(new_psd.data.data)
    if rosDebugMessagesContainer[0]:
        print( " zeroing ", nToZero ," last elements of the psd, out of ",n)
    for i in range(nToZero):
        new_psd.data.data[n - i-1] = 0.
# Vectorized assignment would be better
    return new_psd

def enforce_swig_psd_fmin(raw_psd, fmin):
    global rosDebugMessagesContainer
    df = raw_psd.deltaF
    nToZero = int(1.0*fmin/df)
    new_psd = raw_psd   # Not a copy - I can alter the original
    n = len(new_psd.data.data)
    if rosDebugMessagesContainer[0]:
        print( " zeroing ", nToZero ," first elements of the psd, out of ",n)
    for i in range(nToZero):
        new_psd.data.data[i] = 0.
    return new_psd

def extend_swig_psd_series_to_sampling_requirements(raw_psd, dfRequired, fNyqRequired):
    """
    extend_psd_series_to_sampling_requirements: 
    Takes a conventional 1-sided PSD and extends into a longer 1-sided PSD array by filling intervening samples.
    Currently only works by 'interpolating' between samples.
    *Assumes* fNyq does not change.
    Assumes the input and output are swig REAL8Timeseries objects
    """
    # Allocate new series
    df = raw_psd.deltaF
    n = len(raw_psd.data.data)                                     # odd number for one-sided PSD
    nRequired = int(fNyqRequired/dfRequired)+1     # odd number for one-sided PSD
    facStretch = int((nRequired-1)/(n-1))  # n-1 should be power of 2
    if rosDebugMessagesContainer[0]:
        print( " extending psd of length ", n, " to ", nRequired, "(i.e., fNyq = ", fNyqRequired, ") elements requires a factor of ", facStretch)
    psdNew = lal.CreateREAL8FrequencySeries("PSD", lal.LIGOTimeGPS(0.), 0., dfRequired ,lsu_HertzUnit, n*facStretch)
    # Populate the series.  Slow because it is a python loop
    for i in np.arange(n):
        for j in np.arange(facStretch):
            psdNew.data.data[facStretch*i+j] = raw_psd.data.data[i]  # 
#    psdNew.data.data = (np.array([raw_psd.data.data for j in np.arange(facStretch)])).transpose().flatten()  # a bit too large, but that's fine for our purposes
    return psdNew


my_content=lal.series.PSDContentHandler

try:
    my_content = lal.series.PSDContentHandler

    def get_psd_series_from_xmldoc(fname, inst):
        return read_psd_xmldoc(utils.load_filename(fname ,contenthandler = my_content),root_name=None)[inst]  # return value is pylal wrapping of the data type; index data by a.data[k]
except:
    def get_psd_series_from_xmldoc(fname, inst):
        return read_psd_xmldoc(utils.load_filename(fname))[inst]  # return value is pylal wrapping of the data type; index data by a.data[k]




def get_intp_psd_series_from_xmldoc(fname, inst):
    psd = get_psd_series_from_xmldoc(fname, inst)
    return intp_psd_series(psd)

def resample_psd_series(psd, df=None, fmin=None, fmax=None):
    # handle pylal REAL8FrequencySeries
    #if isinstance(psd, pylal.xlal.datatypes.real8frequencyseries.REAL8FrequencySeries):
    #    psd_fmin, psd_fmax, psd_df, data = psd.f0, psd.f0 + psd.deltaF*len(psd.data), psd.deltaF, psd.data
    #    fvals_orig = psd.f0 + np.arange(len(psd.data))*psd.deltaF
    # handle SWIG REAL8FrequencySeries
    if isinstance(psd, lal.REAL8FrequencySeries):
        psd_fmin, psd_fmax, psd_df, data = psd.f0, psd.f0 + psd.deltaF*len(psd.data.data), psd.deltaF, psd.data.data
        fvals_orig = psd.f0 + np.arange(psd.data.length)*psd_df
    # die horribly
    else:
        raise ValueError("resample_psd_series: Don't know how to handle %s." % type(psd))
    fmin = fmin or psd_fmin
    fmax = fmax or psd_fmax
    df = df or psd_df

    if sci_ver[0]==0 and sci_ver[1]<9:
        # Original interpolation; see also Pankow tweaks in master: https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/rapid_pe/lalsimutils.py
        f = np.arange(psd_fmin, psd_fmax, psd_df)
        ifunc = interpolate.interp1d(f, data)
        def intp_psd(freq):
            return float("inf") if freq >= psd_fmax-psd_df or ifunc(freq) == 0.0 else ifunc(freq)
        intp_psd = np.vectorize(intp_psd)
        psd_intp = intp_psd(np.arange(fmin, fmax, df))
    else: 
        # Proposed new faster interpolation -- the other code to call 1d interpolation uses several slow map functions
        f = np.arange(fmin,fmax,df)
        psd_intp = interpolate.griddata( fvals_orig,data,f,fill_value=float("inf"))

    tmpepoch = lal.LIGOTimeGPS(float(psd.epoch))
    # FIXME: Reenable when we figure out generic error
    """
    tmpunit = lal.Unit()
    lal.ParseUnitString(tmpunit, str(psd.sampleUnits))
    """
    tmpunit = lsu_SecondUnit
    new_psd = lal.CreateREAL8FrequencySeries(epoch = tmpepoch, deltaF=df,
            f0 = fmin, sampleUnits = tmpunit, name = psd.name,
            length=len(psd_intp))
    new_psd.data.data = psd_intp
    return new_psd

def load_resample_and_clean_psd(psd_fname, det, deltaF,verbose=False):
    psd_here = get_psd_series_from_xmldoc(psd_fname, det)  # pylal type!
    tmp = psd_here.data
    if verbose:
        print ("Sanity check reporting : pre-extension, min is ", np.min(tmp), " and maximum is ", np.max(tmp))
    fmin = psd_here.f0
    fmax = fmin + psd_here.deltaF*len(psd_here.data.data)-deltaF
    if verbose:
        print( "PSD deltaF before interpolation %f" % psd_here.deltaF)
    psd_here = resample_psd_series(psd_here, deltaF)
    if verbose:
        print( "PSD deltaF after interpolation %f" % psd_here.deltaF)
        print( "Post-extension the new PSD has 1/df = ", 1./psd_here.deltaF, " (data 1/df = ", 1./deltaF, ") and length ", len(psd_here.data.data))
    tmp = psd_here.data.data
    nBad = np.argmin(tmp[np.nonzero(tmp)])
    fBad = nBad*deltaF
    if verbose:
        print( "Post-extension sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), "(at n=", np.argmin(tmp[np.nonzero(tmp)])," or f=", fBad, ")  and maximum is ", np.max(psd_here.data.data))
    return psd_here


def psd_windowing_factor(window_shape, TDlen):
    """
    [Implementing <w^2>, due to https://dcc.ligo.org/LIGO-T1900249]
    If a PSD is calculated using repeated instances of windowed data, the PSD will be biased downward (i.e., we have less noise than we expect)
    That's fine and self-consistent  if we will be analyzing data which has the same windowing applied.   
    But if we're *not* applying the same window shape to the data we analyze,  we need to correct for the difference
    in noise *actually present* to the noise *used to create the PSD*.

    Ths routine creates <w^2> for w a window shape.
    This could be computed much more efficiently (it is not memory-efficient), but this code is cleaner
    """
    hoft_window = lal.CreateTukeyREAL8Window(TDlen, window_shape)
    # Note this term does not depend much on TDlen, so we can very accurately estimate it with a FIXED TDlen
    return np.sum(hoft_window.data.data**2)/TDlen


def evaluate_tvals(lal_tseries):
    return float(lal_tseries.epoch) +lal_tseries.deltaT*np.arange(lal_tseries.data.length)

def evaluate_fvals(lal_2sided_fseries):
    r"""
    evaluate_fvals(lal_2sided_fseries)
    Associates frequencies with a 2sided lal complex array.  Compare with 'self.longweights' code
    Done by HAND in PrecessingOrbitModesOfFrequency
    Manually *reverses* convention re sign of \omega used in lal!

    Notes:
       a) XXXFrequencySeries  have an f0 and a deltaF, and *logically* they should run from f0....f0+N*df
       b)  I will always use COMPLEX16FrequencySeries, which should run from -fNyq...fNyq-df 
            with fNyq=fSample/2.=1/(2*deltaT)
       c) In practice, I have a REVERSED frequency series associated...WHY? (finish doc review)
          Probably an overall sign difference in \omega t vs -\omega t in FFT definition? LAL documentation
          isn't clear with the overall sign.
                lal:       \int dt exp( -i \omega t) F(t)  = \tilde{F}(\omega)  [lal]
                me:      \int dt exp(  i \omega t) F(t) = \tilde{F}(omega)
                        following Poisson and Will, my old 'DataFourier' code, etc
       d) The low-level code just rotates the original vector right by half the length -- so the usual FFTW
           code data is mapped right.  
    Checks:
       P=lalsimutils.ChooseWaveformParams()
       hf = lalsimutils.complex_hoff(P)
       lalsimutils.evaluate_fvals(hf)
       hf.f0
    Notes:
     ''Physics convention : <math>  \int dt exp(-i \omega t) F(t) </math>
      - my notes use -2 pi i f t convention
      - my NR papers *text* uses \int dt exp(-i \omega t)  convention

    ''Other convention (PN-convenient)   <math>  \int dt exp(i \omega t) F(t) </math>
     - Prakash notes use opposite convention
     - low-level mathematica code (DataFourier) uses \int dt exp(i \omega t) convention
     - Poisson and Will uses opposite convention http://arxiv.org/pdf/gr-qc/9502040v1.pdf, as do all subsequent PN work
      - Cutler and Flanagan use the opposite convention

    """
    npts = lal_2sided_fseries.data.length
    df = lal_2sided_fseries.deltaF
    fvals = np.zeros(npts)
    # https://www.lsc-group.phys.uwm.edu/daswg/projects/lal/nightly/docs/html/group___time_freq_f_f_t__h.html
    # https://www.lsc-group.phys.uwm.edu/daswg/projects/lal/nightly/docs/html/_time_freq_f_f_t_8h.html
    # https://www.lsc-group.phys.uwm.edu/daswg/projects/lal/nightly/docs/html/_time_freq_f_f_t_8c_source.html
    fvals = df* np.array([ npts/2 -k if  k<=npts/2 else -k+npts/2 for k in np.arange(npts)])  # How lal packs its fft
    return fvals


def vecCross(v1,v2):
    return [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]]

def vecDot(v1,v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

def vecUnit(v):
    return v/np.sqrt(vecDot(v,v))


def VectorToFrame(vecRef):
    """
    Convert vector to a frame, by picking relatively arbitrary vectors perp to it.
    Used to convert J to a frame. 
    spin_convention == radiation:
         **Uses the radiation frame conventions to do so.**, so the 'z' direction is orthogonal to the 2nd vector and 'n' (the line of sight) defines things
    spin_convention == L:
         Trickier, because we will want to define angles of L relative to J !  Pick a random alternate vector
    """

    if spin_convention =="radiation":
        vecStart = [0,0,1]
    else:
        vecStart = [0,1,0]  # vector close to the above, but not exactly equal, so I can find the direction of L relative to J

    vec1 = np.array(vecCross(vecStart,vecRef))
    if np.dot(vecRef, np.array(vecStart)) > 1-1e-5 or vecDot(vec1,vec1) < 1e-6:  # if we are aligned, return the cartesian frame
        return np.array([ [1,0,0], [0,1,0], [0,0,1]])
    vec1 = vec1/np.sqrt(vecDot(vec1,vec1))
    vec2 = np.array(vecCross(vecRef,vec1))
    vec2 = vec2/np.sqrt(vecDot(vec2,vec2)) 
#    frame = np.array([-vec2,vec1,vecRef])  # oops,backwards order
    frame = np.array([vec1,vec2,vecRef])  # oops,backwards order
    return frame


def nhat(th,ph):
    return np.array([np.cos(ph)*np.sin(th),np.sin(ph)*np.sin(th), np.cos(th)])

def unit_frame():
    return np.array([[1,0,0], [0,1,0], [0,0,1]])

def polar_angles_in_frame(frm,vec):
    """
    Take a vector in the default frame.
    Evaluate the polar angles of that unit vector in a new (orthonormal) frame 'frm'.
    Not the fastest solution.  Not naturally vectorizable.
    """
    xhat = frm[0]
    yhat = frm[1]
    zhat = frm[2]
    # probably not needed but just in case
    nmz = np.sqrt(np.dot(zhat,zhat))
    zhat = zhat/nmz
    if len(vec.shape)==1:
        th = np.arccos( np.dot(zhat,vec))/np.sqrt(np.dot(vec,vec))
        vPerp = vec - zhat *np.dot(zhat,vec)
        ph = np.angle( np.dot(vPerp,xhat+ 1j*yhat))
    return th,ph


def polar_angles_in_frame_alt(frm, theta,phi,xpy=np): 
    """
    Take polar angles in the default frame.
    Evaluate the polar angles of that unit vector in a new (orthonormal) frame 'frmInverse'.
    Probably easier to vectorize
    """
    frmInverse = frm.T 
    if isinstance(theta,float):
        vec = xpy.cos(phi)*xpy.sin(theta)*frmInverse[0] \
            + xpy.sin(phi)*xpy.sin(theta)*frmInverse[1] \
            + xpy.cos(theta)*frmInverse[2] 
        return xpy.arccos(vec[2]), xpy.angle(vec[0]+1j*vec[1])
    else:
        vec = xpy.outer(xpy.cos(phi)*xpy.sin(theta),frmInverse[0]) \
            + xpy.outer(xpy.sin(phi)*xpy.sin(theta),frmInverse[1]) \
            + xpy.outer(xpy.cos(theta),frmInverse[2])
        return xpy.arccos(vec[:,2]), xpy.angle(vec[:,0]+1j*vec[:,1])
        


# Borrowed: http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
import math
def rotation_matrix(axis,theta):
    axis = axis/math.sqrt(np.dot(axis,axis))
    a = math.cos(theta/2)
    b,c,d = -axis*math.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


# TEST CODE
# import lalsimutils
# import numpy as np
# lalsimutils.polar_angles_in_frame_alt(np.array([[1,0,0], [0,1,0], [0,0,1]]), 0.1, 0.2)
# lalsimutils.polar_angles_in_frame(np.array([[1,0,0], [0,1,0], [0,0,1]]), lalsimutils.nhat(0.1, 0.2))
# lalsimutils.polar_angles_in_frame(lalsimutils.rotation_matrix(np.array([0,0,1]), 0.1), lalsimutils.nhat(0.1, 0.2))
# lalsimutils.polar_angles_in_frame(lalsimutils.rotation_matrix(np.array([0,1,0]), 0.01), lalsimutils.nhat(0.1, 0.0))
# lalsimutils.polar_angles_in_frame_alt(lalsimutils.rotation_matrix(np.array([0,0,1]), 0.1), 0.1, 0.2)
# lalsimutils.polar_angles_in_frame_alt(lalsimutils.rotation_matrix(np.array([0,1,0]), 0.01), 0.1, 0.0)

def DataFourierREAL8(ht):   # Complex fft wrapper (REAL8Time ->COMPLEX16Freq. No error checking or padding!
    TDlen = ht.data.length
    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    FDlen = int(TDlen/2+1)
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lsu_HertzUnit, 
            FDlen)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    # assume memory freed by swig python
    return hf
def DataInverseFourierREAL8(hf):   # Complex fft wrapper (COMPLEX16Freq ->REAL8TimeSeries. No error checking or padding!
    FDlen = hf.data.length
    dt = 1./hf.deltaF/FDlen
    TDlen = 2*(FDlen-1)
    revplan=lal.CreateReverseREAL8FFTPlan(TDlen,0)
    ht = lal.CreateREAL8TimeSeries("Template h(t)", 
            hf.epoch, 0, dt, lsu_DimensionlessUnit, 
            TDlen)
    lal.REAL8FreqTimeFFT( ht, hf, revplan)  
    # assume memory freed by swig python
    return ht

def DataInverseFourier(hf):   # Complex fft wrapper (COMPLEX16Freq ->COMPLEX16Time. No error checking or padding!
    FDlen = hf.data.length
    dt = 1./hf.deltaF/FDlen
    revplan=lal.CreateReverseCOMPLEX16FFTPlan(FDlen,0)
    ht = lal.CreateCOMPLEX16TimeSeries("Template h(t)", 
            hf.epoch, hf.f0, dt, lsu_DimensionlessUnit, 
            FDlen)
    lal.COMPLEX16FreqTimeFFT( ht, hf, revplan)  
    # assume memory freed by swig python
    return ht
def DataFourier(ht):   # Complex fft wrapper (COMPLEX16Time ->COMPLEX16Freq. No error checking or padding!
    TDlen = ht.data.length
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lsu_HertzUnit, 
            TDlen)
    lal.COMPLEX16TimeFreqFFT(hf, ht, fwdplan)
    # assume memory freed by swig python
    return hf
def DataFourierREAL8(ht):   # Complex fft wrapper (REAL8Time ->COMPLEX16Freq. No error checking or padding!
    TDlen = ht.data.length
    fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
    FDlen = int(TDlen/2+1)
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lsu_HertzUnit, 
            FDlen)
    lal.REAL8TimeFreqFFT(hf, ht, fwdplan)
    # assume memory freed by swig python
    return hf

def DataRollBins(ht,nL):  # ONLY FOR TIME DOMAIN.  ACTS IN PLACE.
    assert (isinstance(ht, lal.COMPLEX16TimeSeries) or isinstance(ht,lal.REAL8TimeSeries)) and isinstance(nL,int)
    t0 = ht.epoch 
    ht.epoch += nL*ht.deltaT 
#    print " Rolling by ", nL*ht.deltaT, " or nbins = ", nL
    ht.data.data = np.roll(ht.data.data, -nL)
    return ht

def DataRollTime(ht,DeltaT):  # ONLY FOR TIME DOMAIN. ACTS IN PLACE
    nL = int(DeltaT/ht.deltaT)
    return DataRollBins(ht, nL)            


def convert_waveform_coordinates(x_in,coord_names=['mc', 'eta'],low_level_coord_names=['m1','m2'],enforce_kerr=False,source_redshift=0):
    """
    A wrapper for ChooseWaveformParams() 's coordinate tools (extract_param, assign_param) providing array-formatted coordinate changes.  BE VERY CAREFUL, because coordinates may be defined inconsistently (e.g., holding different variables constant: M and eta, or mc and q).  Note that if ChooseWaveformParam structuers are built ,the loops can be quite slow

    Special cases:
      - coordinates in x_out already in x_in are copied over directly
      - xi==chi_eff, chiMinus, mu1,mu2 : transformed directly from mc, delta_mc, s1z,s2z coordinates, using fast vectorized transformations.
      - source_redshift: if nonzero, convert m1 -> m1 (1+z)=m_z, as fit is done in the detector frame.  We are **assuming source-frame sampling**
    """
    x_out = np.zeros( (len(x_in), len(coord_names) ) )
    # Check for trivial identity transformations and do those by direct copy, then remove those from the list of output coord names
    coord_names_reduced = coord_names.copy() 
    for p in low_level_coord_names:
        if p in coord_names:
            indx_p_out = coord_names.index(p)
            indx_p_in = low_level_coord_names.index(p)
            coord_names_reduced.remove(p)
            x_out[:,indx_p_out] = x_in[:,indx_p_in]

    if 'ecc_cos_meanPerAno' in coord_names_reduced and 'eccentricity' in low_level_coord_names and 'meanPerAno' in low_level_coord_names:
        indx_p_out = coord_names.index('ecc_cos_meanPerAno')
        indx_p_ecc = low_level_coord_names.index('eccentricity')
        indx_p_ell = low_level_coord_names.index('meanPerAno')
        x_out[:,indx_p_out] = x_in[:,indx_p_ecc]*np.cos(x_in[:,indx_p_ell])
        coord_names_reduced.remove('ecc_cos_meanPerAno')
        if 'ecc_sin_meanPerAno' in coord_names_reduced:
            indx_p_out = coord_names.index('ecc_sin_meanPerAno')
            x_out[:,indx_p_out] = x_in[:,indx_p_ecc]*np.sin(x_in[:,indx_p_ell])
            coord_names_reduced.remove('ecc_sin_meanPerAno')
            
    if 'mc' in low_level_coord_names and ('eta' in low_level_coord_names or 'delta_mc' in low_level_coord_names):
        indx_mc = low_level_coord_names.index('mc')
        eta_vals = np.zeros(len(x_in))
        if ('delta_mc' in low_level_coord_names):
                indx_delta = low_level_coord_names.index('delta_mc')
                eta_vals = 0.25*(1- x_in[:,indx_delta]**2)
        elif ('eta' in low_level_coord_names):
                indx_eta = low_level_coord_names.index('eta')
                eta_vals = x_in[:,indx_eta]
        if 'eta' in coord_names_reduced:
            indx_p_out = coord_names.index('eta')
            x_out[:,indx_p_out] = eta_vals
            coord_names_reduced.remove('eta')
        if 'm1' in coord_names_reduced:
            m1_vals =np.zeros(len(x_in))  
            m2_vals =np.zeros(len(x_in))  
            m1_vals,m2_vals = m1m2(x_in[:,indx_mc],eta_vals)
            indx_p_out = coord_names.index('m1')
            x_out[:,indx_p_out] = m1_vals
            coord_names_reduced.remove('m1')
            if 'm2' in coord_names_reduced:
                indx_p_out = coord_names.index('m2')
                x_out[:,indx_p_out] = m2_vals
                coord_names_reduced.remove('m2')

    if 'delta_mc' in coord_names_reduced and 'eta' in low_level_coord_names:
        indx_p_out = coord_names.index('delta_mc')
        indx_eta = low_level_coord_names.index('eta')
        x_out[:,indx_p_out] = np.sqrt(1-4*x_in[:,indx_eta])
        coord_names_reduced.remove('delta_mc')

    # Check for common coordinates we need to transform: xi, chiMinus as the most common, from cartesian
    if ('xi' in coord_names_reduced) and ('s1z' in low_level_coord_names) and ('s2z' in low_level_coord_names) and ('mc' in low_level_coord_names):
        indx_p_out = coord_names.index('xi')
        p = 'xi'
        indx_mc = low_level_coord_names.index('mc')
        indx_s1z = low_level_coord_names.index('s1z')
        indx_s2z = low_level_coord_names.index('s2z')
        eta_vals = np.zeros(len(x_in))  
        m1_vals =np.zeros(len(x_in))  
        m2_vals =np.zeros(len(x_in))  
        if ('delta_mc' in low_level_coord_names):
            indx_delta = low_level_coord_names.index('delta_mc')
            eta_vals = 0.25*(1- x_in[:,indx_delta]**2)
        elif ('eta' in low_level_coord_names):
            indx_eta = low_level_coord_names.index('eta')
            eta_vals = x_in[:,indx_eta]
        m1_vals,m2_vals = m1m2(x_in[:,indx_mc],eta_vals)
        x_out[:,indx_p_out] = (m1_vals*x_in[:,indx_s1z] + m2_vals*x_in[:,indx_s2z])/(m1_vals+m2_vals)
        coord_names_reduced.remove('xi')
    if ('chiMinus' in coord_names_reduced) and ('s1z' in low_level_coord_names) and ('s2z' in low_level_coord_names) and ('mc' in low_level_coord_names):
        indx_p_out = coord_names.index('chiMinus')
        p = 'chiMinus'
        indx_mc = low_level_coord_names.index('mc')
        indx_s1z = low_level_coord_names.index('s1z')
        indx_s2z = low_level_coord_names.index('s2z')
        eta_vals = np.zeros(len(x_in))  
        m1_vals =np.zeros(len(x_in))  
        m2_vals =np.zeros(len(x_in))  
        if ('delta_mc' in low_level_coord_names):
            indx_delta = low_level_coord_names.index('delta_mc')
            eta_vals = 0.25*(1- x_in[:,indx_delta]**2)
        elif ('eta' in low_level_coord_names):
            indx_eta = low_level_coord_names.index('eta')
            eta_vals = x_in[:,indx_eta]
        m1_vals,m2_vals = m1m2(x_in[:,indx_mc],eta_vals)
        x_out[:,indx_p_out] = (m1_vals*x_in[:,indx_s1z] - m2_vals*x_in[:,indx_s2z])/(m1_vals+m2_vals)
        coord_names_reduced.remove(p)

    # mu1,mu2 if cartesian
    if ('mu1' in coord_names_reduced) and ('mu2' in coord_names_reduced) and ('mc' in low_level_coord_names) and ('delta_mc' in low_level_coord_names) and ('s1z' in low_level_coord_names) and ('s2z' in low_level_coord_names):
        indx_pout_mu1 = coord_names.index('mu1')
        indx_pout_mu2 = coord_names.index('mu2')
        indx_mc = low_level_coord_names.index('mc')
        if 'delta_mc' in low_level_coord_names:
            indx_delta = low_level_coord_names.index('delta_mc')
            qvals = (1- x_in[:,indx_delta])/(1+x_in[:,indx_delta])
        elif 'eta' in low_level_coord_names:
            indx_eta = low_level_coord_names.index('eta')
            eta_vals = x_in[:,indx_eta]
            qvals = -1 + (1-np.sqrt(1-4*eta_vals))/(2*eta_vals)
        indx_s1z = low_level_coord_names.index('s1z')
        indx_s2z = low_level_coord_names.index('s2z')
        # delta == (m1-m2)/(m1+m2) == (1-q)/(1+q), so q ==(1-delta)/(1+delta)
        fac =1
        if x_in[:,indx_mc][0]>1e10:
            fac = lal.MSUN_SI
        mu1,mu2,mu3 = tools.Mcqchi1chi2Tomu1mu2mu3(x_in[:,indx_mc]/fac, qvals, x_in[:,indx_s1z], x_in[:,indx_s2z])
        x_out[:,indx_pout_mu1] = mu1
        x_out[:,indx_pout_mu2] = mu2
        coord_names_reduced.remove('mu1')
        coord_names_reduced.remove('mu2')

        # overwrite q, if present
        if 'q' in coord_names_reduced:
            indx_pout_q = coord_names.index('q')
            print(qvals)
            x_out[:,indx_pout_q] = qvals
            coord_names_reduced.remove('q')

        

    # mc,eta,xi -> m1,m2,s1z,s2z
    #   Note this requires ASSUMPTIONS about s2z.  Default will be s2z=s1z HERE, since degenerate
    if ('mc' in low_level_coord_names) and  ('eta' in low_level_coord_names) and ('xi' in low_level_coord_names or 'chieff_aligned' in low_level_coord_names) and ('m1' in coord_names_reduced) and ('s1z' in coord_names_reduced):
        indx_pout_m1 = coord_names.index('m1')
        indx_pout_m2 = coord_names.index('m2')
        indx_pout_s1z = coord_names.index('s1z')
        indx_pout_s2z = coord_names.index('s2z')
        indx_mc = low_level_coord_names.index('mc')
        indx_eta = low_level_coord_names.index('eta')
        if 'xi' in low_level_coord_names:
           indx_xi = low_level_coord_names.index('xi')
        else:
           indx_xi = low_level_coord_names.index('chieff_aligned')
        m1_vals, m2_vals = m1m2(x_in[:,indx_mc], x_in[:,indx_eta])
        # ASSUME s1z=s2z, so s1z==s2z==xi
        x_out[:,indx_pout_m1] = m1_vals
        x_out[:,indx_pout_m2] = m2_vals
        x_out[:,indx_pout_s1z] = x_in[:,indx_xi]
        x_out[:,indx_pout_s2z] = x_in[:,indx_xi]
        coord_names_reduced.remove('m1')
        coord_names_reduced.remove('m2')
        coord_names_reduced.remove('s1z')
        coord_names_reduced.remove('s2z')

    # Spin spherical coordinate names
    if ('chi1' in low_level_coord_names) and ('cos_theta1' in low_level_coord_names) and ('phi1' in low_level_coord_names) and ('chi2' in low_level_coord_names) and ('cos_theta2' in low_level_coord_names) and ('phi2' in low_level_coord_names) and ('mc' in low_level_coord_names) and ('delta_mc' in low_level_coord_names):
        indx_mc = low_level_coord_names.index('mc')
        indx_delta = low_level_coord_names.index('delta_mc')
        indx_chi1 = low_level_coord_names.index('chi1')
        indx_chi2 = low_level_coord_names.index('chi2')
        indx_ct1 = low_level_coord_names.index('cos_theta1')
        indx_ct2 = low_level_coord_names.index('cos_theta2')

        s1z= x_in[:,indx_chi1]*x_in[:,indx_ct1]
        s2z= x_in[:,indx_chi2]*x_in[:,indx_ct2]

        m1_vals =np.zeros(len(x_in))  
        m2_vals =np.zeros(len(x_in))  
        eta_vals = np.zeros(len(x_in))  
        eta_vals = 0.25*(1- x_in[:,indx_delta]**2)
        m1_vals,m2_vals = m1m2(x_in[:,indx_mc],eta_vals)
        if ('xi' in coord_names_reduced):
            indx_pout_xi = coord_names.index('xi')
            x_out[:,indx_pout_xi] = (m1_vals*s1z + m2_vals*s2z)/(m1_vals+m2_vals)
            coord_names_reduced.remove('xi')

        # also build mu1, mu2, ... if present!
        if ('mu1' in coord_names_reduced) and ('mu2' in coord_names_reduced) and ('mc' in low_level_coord_names) and ('delta_mc' in low_level_coord_names):
            indx_pout_mu1 = coord_names.index('mu1')
            indx_pout_mu2 = coord_names.index('mu2')

            qvals = m2_vals/m1_vals
            mu1,mu2,mu3 = tools.Mcqchi1chi2Tomu1mu2mu3(x_in[:,indx_mc], qvals, s1z, s2z)
            x_out[:,indx_pout_mu1] = mu1
            x_out[:,indx_pout_mu2] = mu2
            coord_names_reduced.remove('mu1')
            coord_names_reduced.remove('mu2')



        # also build chiMinus, s1x,s1y, ... , if present : usual use case of doing all of these in spherical coordinates
        if 'chiMinus' in coord_names_reduced:
            indx_pout_chiminus = coord_names.index('chiMinus')
            x_out[:,indx_pout_chiminus] = (m1_vals*s1z - m2_vals*s2z)/(m1_vals+m2_vals)
            coord_names_reduced.remove('chiMinus')
        if ('s1x' in coord_names_reduced) and ('s1y' in coord_names_reduced):
            indx_pout_s1x = coord_names.index('s1x')
            indx_pout_s1y = coord_names.index('s1y')

            indx_phi1=low_level_coord_names.index('phi1')
            cosphi1 = np.cos(x_in[:,indx_phi1])
            sinphi1 = np.sin(x_in[:,indx_phi1])
            sintheta1 = np.sqrt(1-x_in[:,indx_ct1]**2)

            x_out[:,indx_pout_s1x] = x_in[:,indx_chi1]*sintheta1*cosphi1
            x_out[:,indx_pout_s1y] = x_in[:,indx_chi1]*sintheta1*sinphi1
            coord_names_reduced.remove('s1x')
            coord_names_reduced.remove('s1y')
        if ('s2x' in coord_names_reduced) and ('s2y' in coord_names_reduced):
            indx_pout_s2x = coord_names.index('s2x')
            indx_pout_s2y = coord_names.index('s2y')

            indx_phi2=low_level_coord_names.index('phi2')
            cosphi2 = np.cos(x_in[:,indx_phi2])
            sinphi2 = np.sin(x_in[:,indx_phi2])
            sintheta2 = np.sqrt(1-x_in[:,indx_ct2]**2)

            x_out[:,indx_pout_s2x] = x_in[:,indx_chi2]*sintheta2*cosphi2
            x_out[:,indx_pout_s2y] = x_in[:,indx_chi2]*sintheta2*sinphi2
            coord_names_reduced.remove('s2x')
            coord_names_reduced.remove('s2y')
            
    # Spin pseudo-cylindrical coordinate names, standard framing
    if  ('s1z_bar' in low_level_coord_names) and ('phi1' in low_level_coord_names)  and ('s2z_bar' in low_level_coord_names) and ('phi2' in low_level_coord_names) and ('mc' in low_level_coord_names) and ('eta' in low_level_coord_names or 'delta_mc' in low_level_coord_names):
        indx_mc = low_level_coord_names.index('mc')
        indx_s1z = low_level_coord_names.index('s1z_bar')
        indx_s2z = low_level_coord_names.index('s2z_bar')

        eta_vals = np.zeros(len(x_in))  
        s1z= x_in[:,indx_s1z]
        s2z= x_in[:,indx_s2z]
        
        m1_vals =np.zeros(len(x_in))  
        m2_vals =np.zeros(len(x_in))  
        if ('delta_mc' in low_level_coord_names):
            indx_delta = low_level_coord_names.index('delta_mc')
            eta_vals = 0.25*(1- x_in[:,indx_delta]**2)
        elif 'eta' in low_level_coord_names:
            indx_eta = low_level_coord_names.index('eta')
            eta_vals = x_in[:,indx_eta]
        m1_vals,m2_vals = m1m2(x_in[:,indx_mc],eta_vals)

        if 'xi' in coord_names_reduced:
            indx_pout_xi = coord_names.index('xi')
            x_out[:,indx_pout_xi] = (m1_vals*s1z + m2_vals*s2z)/(m1_vals+m2_vals)
            coord_names_reduced.remove('xi')

        # also build mu1, mu2, ... if present!
        if ('mu1' in coord_names_reduced) and ('mu2' in coord_names_reduced) and ('mc' in low_level_coord_names) and ('delta_mc' in low_level_coord_names):
            indx_pout_mu1 = coord_names.index('mu1')
            indx_pout_mu2 = coord_names.index('mu2')

            qvals = m2_vals/m1_vals
            mu1,mu2,mu3 = tools.Mcqchi1chi2Tomu1mu2mu3(x_in[:,indx_mc], qvals, s1z, s2z)
            x_out[:,indx_pout_mu1] = mu1
            x_out[:,indx_pout_mu2] = mu2
            coord_names_reduced.remove('mu1')
            coord_names_reduced.remove('mu2')


        # also build chiMinus, s1x,s1y, ... , if present : usual use case of doing all of these for likelihood fit
        if 'chiMinus' in coord_names_reduced:
            indx_pout_chiminus = coord_names.index('chiMinus')
            x_out[:,indx_pout_chiminus] = (m1_vals*s1z - m2_vals*s2z)/(m1_vals+m2_vals)
            coord_names_reduced.remove('chiMinus')

        if 'chi1' in coord_names_reduced:
            indx_pout_chi1 = coord_names.index('chi1')
        
            if ('chi1_perp_bar' in low_level_coord_names):
                indx_chi1_perp_bar = low_level_coord_names.index('chi1_perp_bar')
                chi1_perp_bar = x_in[:,indx_chi1_perp_bar]
            elif 'chi1_perp_u' in low_level_coord_names:
                indx_chi1_perp_u = low_level_coord_names.index('chi1_perp_u')
                chi1_perp_bar = np.power(x_in[:,indx_chi1_perp_u], 4.)

            chi1_vals = np.sqrt(chi1_perp_bar**2 * (1-s1z**2) + s1z**2)
            x_out[:,indx_pout_chi1]=chi1_vals
            coord_names_reduced.remove('chi1')

        if 'chi2' in coord_names_reduced:
            indx_pout_chi2 = coord_names.index('chi2')
        
            if ('chi2_perp_bar' in low_level_coord_names):
                indx_chi2_perp_bar = low_level_coord_names.index('chi2_perp_bar')
                chi2_perp_bar = x_in[:,indx_chi2_perp_bar]
            elif 'chi2_perp_u' in low_level_coord_names:
                indx_chi2_perp_u = low_level_coord_names.index('chi2_perp_u')
                chi2_perp_bar = np.power(x_in[:,indx_chi2_perp_u], 4.)

            chi2_vals = np.sqrt(chi2_perp_bar**2 * (1-s2z**2) + s2z**2)
            x_out[:,indx_pout_chi2]=chi2_vals
            coord_names_reduced.remove('chi2')

        if ('s1x' in coord_names_reduced) and ('s1y' in coord_names_reduced):
            indx_pout_s1x = coord_names.index('s1x')
            indx_pout_s1y = coord_names.index('s1y')

            indx_phi1=low_level_coord_names.index('phi1')
            cosphi1 = np.cos(x_in[:,indx_phi1])
            sinphi1 = np.sin(x_in[:,indx_phi1])

            if ('chi1_perp_bar' in low_level_coord_names):
                indx_chi1_perp_bar = low_level_coord_names.index('chi1_perp_bar')
                chi1_perp_bar = x_in[:,indx_chi1_perp_bar]
            elif 'chi1_perp_u' in low_level_coord_names:
                indx_chi1_perp_u = low_level_coord_names.index('chi1_perp_u')
                chi1_perp_bar = np.power(x_in[:,indx_chi1_perp_u], 4.)
            chi1_perp = chi1_perp_bar * np.sqrt(1.-s1z**2)  # R = Rbar * sqrt(1-z^2)
            x_out[:,indx_pout_s1x] = chi1_perp*cosphi1
            x_out[:,indx_pout_s1y] = chi1_perp*sinphi1
            coord_names_reduced.remove('s1x')
            coord_names_reduced.remove('s1y')

        if ('s2x' in coord_names_reduced) and ('s2y' in coord_names_reduced):
            indx_pout_s2x = coord_names.index('s2x')
            indx_pout_s2y = coord_names.index('s2y')

            indx_phi2=low_level_coord_names.index('phi2')
            cosphi2 = np.cos(x_in[:,indx_phi2])
            sinphi2 = np.sin(x_in[:,indx_phi2])

            if ('chi2_perp_bar' in low_level_coord_names):
                indx_chi2_perp_bar = low_level_coord_names.index('chi2_perp_bar')
                chi2_perp_bar = x_in[:,indx_chi2_perp_bar]
            elif 'chi2_perp_u' in low_level_coord_names:
                indx_chi2_perp_u = low_level_coord_names.index('chi2_perp_u')
                chi2_perp_bar = np.power(x_in[:,indx_chi2_perp_u], 4.)
            chi2_perp = chi2_perp_bar * np.sqrt(1.-s2z**2)
#            sintheta2 = chi2_perp/np.sqrt(s2z**2+chi2_perp**2)

            x_out[:,indx_pout_s2x] = chi2_perp*cosphi2
            x_out[:,indx_pout_s2y] = chi2_perp*sinphi2
            coord_names_reduced.remove('s2x')
            coord_names_reduced.remove('s2y')
        if 'chi_p' in coord_names_reduced:
            indx_pout_chip = coord_names.index('chi_p')
            q_vals =   m2_vals/m1_vals
            A1 = (2+3.*q_vals/2)
            A2 = (2+3./(2*q_vals))

            if ('chi1_perp_bar' in low_level_coord_names):
                indx_chi1_perp_bar = low_level_coord_names.index('chi1_perp_bar')
                chi1_perp_bar = x_in[:,indx_chi1_perp_bar]
            elif 'chi1_perp_u' in low_level_coord_names:
                indx_chi1_perp_u = low_level_coord_names.index('chi1_perp_u')
                chi1_perp_bar = np.power(x_in[:,indx_chi1_perp_u], 4.)

            if ('chi2_perp_bar' in low_level_coord_names):
                indx_chi2_perp_bar = low_level_coord_names.index('chi2_perp_bar')
                chi2_perp_bar = x_in[:,indx_chi2_perp_bar]
            elif 'chi2_perp_u' in low_level_coord_names:
                indx_chi2_perp_u = low_level_coord_names.index('chi2_perp_u')
                chi2_perp_bar = np.power(x_in[:,indx_chi2_perp_u], 4.)


            A1S1p_scalar = A1*(m1_vals**2) *chi1_perp_bar*np.sqrt(1-s1z**2)
            A2S2p_scalar = A2*(m2_vals**2)*chi2_perp_bar*np.sqrt(1-s2z**2)
            Sp = np.maximum(A1S1p_scalar, A2S2p_scalar)
            x_out[:,indx_pout_chip] = Sp/(A1*m1_vals**2)
            coord_names_reduced.remove('chi_p')



    if ('chi_p' in coord_names_reduced) and ('s1x' in low_level_coord_names  and 's1y' in low_level_coord_names) and ('mc' in low_level_coord_names):
        indx_pout_chip = coord_names.index('chi_p')
        indx_mc = low_level_coord_names.index('mc')
        indx_s1x = low_level_coord_names.index('s1x')
        indx_s2x = low_level_coord_names.index('s2x')
        indx_s1y = low_level_coord_names.index('s1y')
        indx_s2y = low_level_coord_names.index('s2y')

        s1x= x_in[:,indx_s1x]
        s2x= x_in[:,indx_s2x]
        s1y= x_in[:,indx_s1y]
        s2y= x_in[:,indx_s2y]

        if ('delta_mc' in low_level_coord_names):
            indx_delta = low_level_coord_names.index('delta_mc')
            eta_vals = 0.25*(1- x_in[:,indx_delta]**2)
        elif ('eta' in low_level_coord_names):
            indx_eta = low_level_coord_names.index('eta')
            eta_vals = x_in[:,indx_eta]
            if 'delta_mc' in coord_names_reduced:
                indx_pout_delta = coord_names.index('delta_mc')
                delta_vals = np.sqrt(1.-4*eta_vals)
                x_out[:,indx_pout_delta] = delta_vals
                coord_names_reduced.remove('delta_mc')
        m1_vals =np.zeros(len(x_in))  
        m2_vals =np.zeros(len(x_in))  
        m1_vals,m2_vals = m1m2(x_in[:,indx_mc],eta_vals)

        q_vals =   m2_vals/m1_vals
        A1 = (2+3.*q_vals/2)
        A2 = (2+3./(2*q_vals))
        A1S1p =  np.c_[A1*m1_vals**2 *s1x,  A1*m1_vals**2 *s1y]
        A2S2p =  np.c_[A2*m2_vals**2 *s2x,  A2*m2_vals**2 *s2y]
        Sp =np.maximum(np.linalg.norm( A1S1p,axis=-1), np.linalg.norm(A2S2p,axis=-1))
        x_out[:,indx_pout_chip] = Sp/(A1*m1_vals**2)
        coord_names_reduced.remove('chi_p')
        

    if ('LambdaTilde' in coord_names_reduced) and ('lambda1' in low_level_coord_names and 'lambda2' in low_level_coord_names):
        # deal with scenario where only one of lambda1, lambda2 specified IN FUTURE
        indx_p_out = coord_names.index('LambdaTilde')
        indx_la1 = low_level_coord_names.index('lambda1')
        indx_la2 = low_level_coord_names.index('lambda2')
        la1_vals = x_in[:,indx_la1]
        la2_vals = x_in[:,indx_la2]
        if 'mc' in low_level_coord_names and 'delta_mc' in low_level_coord_names:
            indx_mc = low_level_coord_names.index('mc')
            eta_vals = np.zeros(len(x_in))  
            m1_vals =np.zeros(len(x_in))  
            m2_vals =np.zeros(len(x_in))  
            if ('delta_mc' in low_level_coord_names):
                indx_delta = low_level_coord_names.index('delta_mc')
                eta_vals = 0.25*(1- x_in[:,indx_delta]**2)
            elif ('eta' in low_level_coord_names):
                indx_eta = low_level_coord_names.index('eta')
                eta_vals = x_in[:,indx_eta]
            m1_vals,m2_vals = m1m2(x_in[:,indx_mc],eta_vals)
            Lt, dLt   = tidal_lambda_tilde(m1_vals, m2_vals, la1_vals, la2_vals)
            x_out[:,indx_p_out] = Lt
            coord_names_reduced.remove('LambdaTilde')
            if 'DeltaLambdaTilde' in coord_names_reduced:
                indx_q_out = coord_names.index('DeltaLambdaTilde')
                x_out[:,indx_q_out] = dLt
                coord_names_reduced.remove('DeltaLambdaTilde')


    # perform any mass conversions needed, so output in detector frame given input in source frame
    if source_redshift:
        for name in ['mc', 'm1', 'm2']:
            if name in coord_names:
                indx_name = coord_names.index(name)
                x_out[name] *= (1+source_redshift)

    # return if we don't need to do any more conversions (e.g., if we only have --parameter specification)
    if len(coord_names_reduced)<1:
        return x_out

    print(" Fallthrough to non-vector-coords for ", coord_names_reduced,low_level_coord_names)
    
    P = ChooseWaveformParams()
    # note NO MASS CONVERSION here, because the fit is in solar mass units!
    for indx_out  in np.arange(len(x_in)):
        for indx in np.arange(len(low_level_coord_names)):
            if low_level_coord_names[indx] != 'chi_pavg':
                P.assign_param( low_level_coord_names[indx], x_in[indx_out,indx])            
        # Apply redshift: assume input is source-frame mass, convert m1 -> m1(1+z) = m1_z, as fit used detector frame
        P.m1 = P.m1*(1+source_redshift)
        P.m2 = P.m2*(1+source_redshift)
        for indx in np.arange(len(coord_names_reduced)):
            p = coord_names_reduced[indx]
            indx_p_out= coord_names.index(p)
            x_out[indx_out,indx_p_out] = P.extract_param(p)
        if enforce_kerr and (P.extract_param('chi1') > 1 or P.extract_param('chi2') >1):  # insure Kerr bound satisfied
            x_out[indx_out] = -np.inf*np.ones( len(coord_names) ) # return negative infinity for all coordinates, if Kerr bound violated
    return x_out

def convert_waveform_coordinates_with_eos(x_in,coord_names=['mc', 'eta'],low_level_coord_names=['m1','m2'],enforce_kerr=False,eos_class=None,no_matter1=False,no_matter2=False,source_redshift=0):
    """
    A wrapper for ChooseWaveformParams() 's coordinate tools (extract_param, assign_param) providing array-formatted coordinate changes.  BE VERY CAREFUL, because coordinates may be defined inconsistently (e.g., holding different variables constant: M and eta, or mc and q)
    """
    try:
        import RIFT.physics.EOSManager  as EOSManager # be careful to avoid recursive dependence!
    except:
        print( " - Failed to load EOSManager - ")  # this will occur at the start
    assert not (eos_class==None)
    x_out = np.zeros( (len(x_in), len(coord_names) ) )
    P = ChooseWaveformParams()
    for indx_out  in np.arange(len(x_in)):
        # WARNING UNUSUAL CONVENTION
        #   note, P.m1, P.m2 in Msun units here
        for indx in np.arange(len(low_level_coord_names)):
            P.assign_param( low_level_coord_names[indx], x_in[indx_out,indx])
        # Impose EOS, unless no_matter1
        if no_matter1:
            P.lambda1=0
        else:
          try:
            if P.m1 < eos_class.mMaxMsun:
                P.lambda1 = eos_class.lambda_from_m(P.m1*lal.MSUN_SI)
            else:
                if rosDebugMessagesContainer[0]:
                    print( " Failed (safely) for ", P.m1)
                P.lambda1= -np.inf
          except:
#              print " Failed for ", P.m1
              P.lambda1 = - np.inf
        if no_matter2:
            P.lambda2=0
        else:
          try:
            if P.m2 < eos_class.mMaxMsun:
                P.lambda2 = eos_class.lambda_from_m(P.m2*lal.MSUN_SI)
            else:
                if rosDebugMessagesContainer[0]:
                    print( " Failed (safely) for ", P.m2)
                P.lambda2= -np.inf
          except:
#              print " Failed for ", P.m2
              P.lambda2=-np.inf
        # Apply redshift: assume input is source-frame mass, convert m1 -> m1(1+z) = m1_z, as fit used detector frame
        P.m1 = P.m1*(1+source_redshift)
        P.m2 = P.m2*(1+source_redshift)
        # extract
        for indx in np.arange(len(coord_names)):
            x_out[indx_out,indx] = P.extract_param(coord_names[indx])
        if enforce_kerr and (P.extract_param('chi1') > 1 or P.extract_param('chi2') >1):  # insure Kerr bound satisfied
            x_out[indx_out] = -np.inf*np.ones( len(coord_names) ) # return negative infinity for all coordinates, if Kerr bound violated
    return x_out

def test_coord_output(x_out):
    """
    Checks if any of the x_out are -np.inf.  Returns a boolean array [ True, False, False, ...] with True if the corresponding coordinate row is ok, false otherwise
    """
    if len(x_out.shape) > 1:
        ret = np.apply_along_axis(all, 1,map( np.isfinite, x_out))
    else:
        ret = map(np.isfinite, x_out)
    return ret
def RangeProtect(fn,val):
    """
    RangeProtect(fn) wraps fn, returning  - np.inf in cases where an argument is not finite, and calling the function otherwise.
    Intended to be used with test_coord_output and coordinate conversion routines, to identify range errors, etc.

    This function assumes fn acts on EACH ELEMENT INDIVIDUALLY, and does not reduce the dimension.
    Not to use.
    """
    def my_protected(x):
        x_test = test_coord_output(x)
        return np.piecewise(x, [x_test, np.logical_not(x_test)], [fn, (lambda x: val)])
    return my_protected 
def RangeProtectReduce(fn,val):
    """
    RangeProtect(fn) wraps fn, returning  - np.inf in cases where an argument is not finite, and calling the function otherwise.
    Intended to be used with test_coord_output and coordinate conversion routines, to identify range errors, etc.

    This function assumes fn acts to REDUCE the data to one dimension (e.g., a list of points)
    """
    def my_protected(x):
        x_test = test_coord_output(x)
        ret = val*np.ones(len(x_test))
        ret[x_test] = fn( x[x_test])  # only apply the function to the rows that pass the test, otherwise return val
        return ret
    return my_protected 

def symmetry_sign_exchange(coord_names):
    P=ChooseWaveformParams()
    P.randomize()

    sig_list = []
    for indx in np.arange(len(coord_names)):
        val1 = P.extract_param(coord_names[indx])

        # LAL spin convention!  Assumes spins do not rotate with phiref
        phiref = P.phiref
        m1,s1x,s1y,s1z = [P.m1,P.s1x,P.s1y,P.s1z]
        m2,s2x,s2y,s2z = [P.m2,P.s2x,P.s2y,P.s2z]
        P.m1,P.s1x,P.s1y,P.s1z = [m2,s2x,s2y,s2z]
        P.m2,P.s2x,P.s2y,P.s2z = [m1,s1x,s1y,s1z]
        
        lambda1,lambda2 = [P.lambda1,P.lambda2]
        P.lambda1,P.lambda2= [lambda2,lambda1]

        val2 = P.extract_param(coord_names[indx])
        if np.abs(val1-val2) < 1e-5 *np.abs(val1*2):
            sig_list.append(1)
        elif np.abs(val1+val2) < 1e-5 * np.abs(val1*2):
            sig_list.append(-1)
        else:
            sig_list.append(0)

    return sig_list


## prior range argument
def guess_mc_range(event_dict,force_mc_range=None):
    Mchirp_event = mchirp( event_dict["m1"],event_dict["m2"])
    # from helper code: choose some mc range that's plausible, not a delta function at trigger mass
    fmin_fiducial = 20
    v_PN_param = (np.pi* Mchirp_event*fmin_fiducial*MsunInSec)**(1./3.)  # 'v' parameter
    snr_fac = 1 # not using that information
    v_PN_param = v_PN_param
    v_PN_param_max = 0.2
    fac_search_correct = 1.5   # if this is too large we can get duration effects / seglen limit problems when mimicking LI
    ln_mc_error_pseudo_fisher = 1.5*np.array(fac_search_correct)*0.3*(v_PN_param/v_PN_param_max)**(7.)/snr_fac 
    if ln_mc_error_pseudo_fisher  >1:
        ln_mc_error_pseudo_fisher =0.8   # stabilize
    mc_max = np.exp( ln_mc_error_pseudo_fisher) * Mchirp_event
    mc_min = np.exp( -ln_mc_error_pseudo_fisher) * Mchirp_event

    if force_mc_range:
        mc_min,mc_max = list(map(float, force_mc_range.replace('[','').replace(']','').split(',')))

    return mc_min,mc_max

## prior range argument for injections
def guess_mc_range_mdc(P,force_mc_range=None):
    Mchirp_event = mchirp(P.m1/lal.MSUN_SI,P.m2/lal.MSUN_SI)
    # from helper code: choose some mc range that's plausible, not a delta function at trigger mass
    fmin_fiducial = 20
    v_PN_param = (np.pi* Mchirp_event*fmin_fiducial*MsunInSec)**(1./3.)  # 'v' parameter
    snr_fac = 1 # not using that information
    v_PN_param = v_PN_param
    v_PN_param_max = 0.2
    fac_search_correct = 1.5   # if this is too large we can get duration effects / seglen limit problems when mimicking LI
    ln_mc_error_pseudo_fisher = 1.5*np.array(fac_search_correct)*0.3*(v_PN_param/v_PN_param_max)**(7.)/snr_fac
    if ln_mc_error_pseudo_fisher  >1:
        ln_mc_error_pseudo_fisher =0.8   # stabilize
    mc_max = np.exp( ln_mc_error_pseudo_fisher) * Mchirp_event
    mc_min = np.exp( -ln_mc_error_pseudo_fisher) * Mchirp_event

    if force_mc_range:
        mc_min,mc_max = list(map(float, force_mc_range.replace('[','').replace(']','').split(',')))

    return mc_min,mc_max


def rotate_hlm_static(hlm,alphaA,betaA,gammaA,extra_polarization=None):
    """
    Rotate output hlm by these Euler angles
    Note this is a full system rotation, and therefore should be associated with rotation of the physical quantities (e.g., spin vectors, J,L)

    extra_polarization is applied at the end, as a multiplicative factor, if nonzero
    """
    hCatOut = {}
    modes = list(hlm)
    Lmax = np.max(np.array(modes)[:,0]) # Lmax
    h0 = hlm[modes[0]] # some reference mode, for types
    lal_type=False

    for L in range(2, Lmax+1):
        for M in range(-L,L+1):
            hCatOut[(L,M)] = None
            # Define output data
            if isinstance(hlm[(L,M)], lal.COMPLEX16TimeSeries):
                hCatOut[(L,M)] = lal.CreateCOMPLEX16TimeSeries("Template h(t)", 
                                                 h0.epoch, h0.f0, h0.deltaT, lsu_DimensionlessUnit, 
                                                 h0.data.length)
                hCatOut[(L,M)].data.data *=0 # initialize
                lal_type=True
            elif isinstance(hlm[(L,M)] , lal.Complex16FrequencySeries):
                hCatOut[(L,M)] =  lal.CreateCOMPLEX16FrequencySeries("Template h(t)", 
                                                                     h0.epoch, h0.f0, h0.deltaF, lsu_HertzUnit ,
                                                                     h0.data.length)
                hCatOut[(L,M)].data.data *=0 # initialize
                lal_type=True
            elif isinstance(hlm[(L,M)], np.ndarray):
                hCatOut[(L,M)] = np.zeros(h0.shape)
            else:
                raise ValueError(" hlm unknown data type for rotation")
            # Loop and populate, based on input modes. Does not need to be full in input, but will be full output
            for mode in modes:
                if mode[0]==L:
                    if lal_type:
                        hCatOut[(L,M)].data.data += lal.WignerDMatrix(L,M,mode[1], alphaA, betaA,gammaA)*hlm[mode].data.data
                    else:
                        hCatOut[(L,M)] += lal.WignerDMatrix(L,M,mode[1], alphaA, betaA,gammaA)*hlm[mode]
            # multiply by extra polarization, if requexted
            if extra_polarization:
                if lal_type:
                    hCatOut[(L,M)].data.data *= np.exp(-2j*extra_polarization)
                else:
                    hCatOut[(L,M)] *= np.exp(-2j*extra_polarization)

    return hCatOut


def extract_JL_angles(P,return_inverse=True,phase_factor=1):
    """
    extract_JL_angles:  finds Euler angles for transformation  see eg C13, C14 in 2004.06503

    Using directly-coded J,L expressions rather than calling the lalsuite routine, to make sure phases agree with what I intend.
    """
    # Directly provide code needed to pull out theta_JL, phi_JL : don't use P.extract_param() for clarity
    Lref = P.OrbitalAngularMomentumAtReferenceOverM2()
    Lhat = Lref/np.sqrt(np.dot(Lref,Lref))
    Jref = P.TotalAngularMomentumAtReferenceOverM2()
    Jhat = Jref/np.sqrt(np.dot(Jref, Jref))

    # polar angles of J (as constructed in L frame) relative to  L frame
    theta_JL, phi_JL = polar_angles_in_frame(VectorToFrame(Lhat), Jhat)
    # To make review-stable, be VERY explicit : try to use lalsuite function calls, not above
    my_cos = np.dot(Lhat, Jhat)
    theta_JL = P.incl   # this is pretty close. Good enough if we are nearly aligned
    if my_cos < 1-1e-5:  # but if we are even slightly misaligned, use the acos of above
        theta_JL = np.arccos( my_cos ) #P.extract_param('beta')  # this is just Jhat.Jhat

    # Remaining angle:
    nhat_obs = np.array([ np.sin(P.incl)*np.cos(np.pi/2*phase_factor-P.phiref), np.sin(P.incl)*np.sin(np.pi/2*phase_factor-P.phiref), np.cos(P.incl)])  # base vector
    vecZ = np.array([0,0,1])
    vecY = np.array([0,1,0])

    # thetaJL == beta.  However, we're noit going to use this
#    theta_jn, _, theta_1, theta_2, phi_12, a_1, a_2 =lalsim.SimInspiralTransformPrecessingWvf2PE(
#            P.incl, P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.m1, P.m2, P.fref, P.phiref)

    # rotation from J to point along L for example
    rot = np.matmul(rotation_matrix( vecY, -theta_JL), rotation_matrix( vecZ, -phi_JL))

    # Rotation around L needed to rotate viewing direction
    nhat_rot = np.matmul(rot, nhat_obs)
    last_angle = np.angle( nhat_rot[0]+1j*nhat_rot[1])

    # Phase conventions as in XPHM paper https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.104056
    if return_inverse:
        # the two np.pi factors end up producing a mode sign  (-1)^m (-1)^m'  to deal with  thetaJL < 0  , so we don't have a negative angle there.
        return np.pi - last_angle, theta_JL, np.pi-phi_JL
    else:
        # 
        return phi_JL, theta_JL, last_angle
