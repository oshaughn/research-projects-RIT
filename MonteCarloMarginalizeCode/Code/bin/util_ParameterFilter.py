#! /usr/bin/env python
#
# GOAL
#   - use downselect-parameter, downselect-parameter-range to filter *.xml.gz files for target samples
#
#


import argparse
import sys
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lalframe
import lal
import functools
import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--fname",default=None)
parser.add_argument("--inj",default=None,help="Reference configuration")
parser.add_argument("--event",default=0,type=int,help="Reference event in the inj file")
parser.add_argument("--fname-out",default="filtered-output.xml.gz")
# Parameters
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--verbose",action='store_true')
opts=  parser.parse_args()

downselect_dict = {}
if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = map(eval,opts.downselect_parameter_range)
else:
    dlist = []
    dlist_ranges = []
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]

if otps.verbose:
    print(downselect_dict)

# downselection procedure: fix units so I can downselect on mass parameters
for p in ['mc', 'm1', 'm2', 'mtot']:
    if p in downselect_dict.keys():
        downselect_dict[p] = np.array(downselect_dict[p],dtype=np.float64)
        downselect_dict[p] *= lal.MSUN_SI  # add back units


P_list_in = lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname)
P_inj = lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname)[opts.event]
param_ref_vals = {}
for param in downselect_dict.keys():
    param_ref_vals[param] = P_inj.extract_param(param)

P_list = []
for P in P_list_in:
    include_item = True
    for param in downselect_dict.keys():
        param_here = P.extract_param(param)
        if not (param_here > downselect_dict[param][0] and param_here < downselect_dict[param][1]):
            include_item=False
            continue
    if include_item:
        P_list.append(P)


lalsimutils.ChooseWaveformParams_array_to_xml(P_list, opts.fname_out.replace(".xml.gz", "")) # don't write the postfix twice
