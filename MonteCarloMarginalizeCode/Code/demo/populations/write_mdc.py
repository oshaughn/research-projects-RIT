#! /usr/bin/env python

# KJ Wagner 2023

import numpy as np
import argparse
#import os
#import sys
import pandas as pd
import configparser
import lal
import lalsimulation as lalsim
#from RIFT.misc.dag_utils import mkdir
import RIFT.lalsimutils as lalsimutils
#from RIFT.misc.dag_utils import which
#lalapps_path2cache = which('lal_path2cache')
from ligo.lw import lsctables, table, utils
#from scipy.stats import norm

parser = argparse.ArgumentParser()
parser.add_argument("--param-file",default=None,required=True,help='Population file with params to pass to PE')
parser.add_argument("--ini-file",default=None,required=True,help="read standard ini file so inj and PE match")
opts=  parser.parse_args()

param_file = opts.param_file

config = configparser.ConfigParser(allow_no_value=True)
config.read(opts.ini_file)

# Fixed time/sky location
fix_sky_location=config.getboolean('priors','fix_sky_location')
fix_event_time=config.getboolean('priors','fix_event_time')
fiducial_ra=float(config.get('priors','fiducial_ra'))
fiducial_dec=float(config.get('priors','fiducial_dec'))
fiducial_event_time=float(config.get('priors','fiducial_event_time'))

# Spin and eccentricity requests
no_spin=config.getboolean('priors','no_spin')
precessing_spin=config.getboolean('priors','precessing_spin')
aligned_spin=config.getboolean('priors','aligned_spin')
use_eccentric=config.getboolean('priors','use_eccentric')

# Noise/data requirements
ifos = ['H1','L1', 'V1']
channels = {'H1': 'FAKE-STRAIN', 'L1':'FAKE-STRAIN','V1':'FAKE-STRAIN'}
flow = {'H1':20, 'L1':20, 'V1':20}

fmin_template = float(config.get('waveform','fmin_template'))
fref = float(config.get('waveform','fmin_template')) # the same for now
fmax = float(config.get('data','fmax'))
srate_data = float(config.get('data', 'srate_data'))
seglen_data = float(config.get('data','seglen_data'))
seglen_analysis = float(config.get('data','seglen_analysis'))

# Approximant requests - TaylorF2Ecc requires extra (check rift_O4c for f_ecc???)
approx_str = config.get('waveform','approx')
if 'TaylorF2Ecc' in approx_str:
  set_fecc=20.0
  amporder=-1
lmax = 4

## Read in and store values from Zee
read_vals = pd.read_csv(param_file, header=0, index_col=False, delim_whitespace=True)
n_events = len(read_vals)

## Create list of all waveforms to send to xml, like ish pp_RIFT
# mass_1_source mass_2_source a_1 a_2 cos_tilt_1 cos_tilt_2 phi_1 phi_2 phi_12 eccentricity mean_anomaly redshift ra dec detection_time cos_iota psi phi_orb mass_1_detector mass_2_detector luminosity_distance
P_list =[]; indx=0
while len(P_list) < n_events:
  print("event ++{}".format(indx))
  P = lalsimutils.ChooseWaveformParams()
  # Randomize (sky location, etc)
  #P.randomize(dMax=d_max,dMin=d_min,aligned_spin_Q=aligned_spin,volumetric_spin_prior_Q=volumetric_spin,sMax=chi_max)
  P.tref = fiducial_event_time
  P.fmin = fmin_template
  P.fref = fref
  P.deltaF = 1./seglen_data
  P.deltaT = 1./srate_data

  # sky location
  if fix_sky_location:
      P.theta = fiducial_dec
      P.phi = fiducial_ra
  else:
      P.theta = read_vals['ra'][indx]
      P.phi = read_vals['dec'][indx]

  # spins, none and aligned for now
  if no_spin:
      P.s1x=P.s1y=P.s1z=0.0
      P.s2x=P.s2y=P.s2z=0.0
  elif aligned_spin:
      P.s1x = P.s1y=0
      P.s2x = P.s2y=0
      P.s1z = read_vals['spin1_z'][indx]
      P.s2z = read_vals['spin2_z'][indx]
  P.approx = lalsim.GetApproximantFromString(approx_str)
  
  # Read in values from Zee
  P.m1 = read_vals['mass_1_source'][indx] * lal.MSUN_SI
  P.m2 = read_vals['mass_2_source'][indx] * lal.MSUN_SI

  if use_eccentric:
    P.eccentricity = read_vals['eccentricity'][indx]
    P.meanPerAno = read_vals['mean_anomaly'][indx]
  if 'TaylorF2Ecc' in approx_str:
    P.ampO = amporder

  P.dist = read_vals['luminosity_distance'][indx] * 1e6 * lal.PC_SI
  P.phiref = read_vals['phi_orb'][indx]
  P.psi = read_vals['psi'][indx]
  P.incl = np.arccos(read_vals['cos_iota'][indx]) # check!!
  ## event time set in param file, but for inj set it??
  #P.tref = read_vals['detection_time'][indx] 
  
  P_list.append(P)
  indx+=1

lalsimutils.ChooseWaveformParams_array_to_xml(P_list,"mdc",fref=fmin_template, deltaF=1/16.)
