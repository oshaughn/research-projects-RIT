#! /usr/bin/env python
#

import argparse
import sys
import numpy as np
import scipy
import lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools
import itertools

from scipy.optimize import brentq



import NRWaveformCatalogManager as nrwf
import scipy.optimize as optimize

parser = argparse.ArgumentParser()
# Parameters
parser.add_argument('--nr-group', default=None,help="NR group to search (otherwise, will loop over all)")
parser.add_argument('--fname-fisher', default=None,help="Fisher name")
parser.add_argument('--fname', default=None,help="Name for XML file")
opts=  parser.parse_args()



###
### Key routine
###

def nr_closest_to(P,distance_function_P1P2,nr_group,mass_ref=None):


    mass_start = 150*lal.MSUN_SI
    if mass_ref:
        mass_start = mass_ref
    result_list = []
    for param  in nrwf.internal_WaveformsAvailable[nr_group]:
        acat = nrwf.WaveformModeCatalog(nr_group,nr_param,metadata_only=True)
        P1 = acat.P1
        def distance_at_mass(m):
            P1.assign_param('mtot', m)
            return distance_function_P1P2(P,P1)
        
        res = optimize.minimize(distance_at_mass, mass_start)
        result_list.append(param,res.x)
        

    return result_list


def make_distance_for_fisher(mtx, param_names):
    def my_distance(P1,P2):
        # extract parameters for P1
        vec1 = np.zeros(len(param_names))
        for indx in np.arange(len(param_names)):
            vec1[indx] = P1.extract_param(param_names[indx])

        # extract parameters for P2
        vec2 = np.zeros(len(param_names))
        for indx in np.arange(len(param_names)):
            vec2[indx] = P2.extract_param(param_names[indx])


        deltaV = vec1-vec2
        return np.dot(deltaV, np.dot(mtx,deltaV))

    return my_distance


###
### Load in Fisher. Create default distance function (WARNING: needs parameter labels to be appended!)
###
datFisher = np.genfromtxt(opts.fname_fisher,records=True)

# parameter names
param_names = []

# parameter matrix
mtx = np.zeros( (len(param_names),len(param_names)) )
for indx1 in np.arange(len(param_names)):
  for indx2 in np.arange(len(param_names)):
      mtx = datFisher[indx1][indx2]

# make distance function


###
### Load in xml
###

P_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname)


###
### Loop over XML
###
if opts.group:
    glist = [opts.group]
else:
    glist = nrwf.internal_ParametersAvailable.keys()
best_fits =[]
for P in P_list:
    for group in glist:
        res = nr_closest_to(P, group)
        print " Best fit ", res, " for ", group
        best_fits.append(group, res[0],res[1])  # add best fit per group
