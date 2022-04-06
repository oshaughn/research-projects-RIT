#! /usr/bin/env python
#
# util_RandomizeOverlapOrder.py
#
# 
# GOAL
#   - take an overlap.xml.gz (SimInspiralTable), and randomize its order.  
#   - Important when merging files from many workers, to avoid accidentally using only the output from one of them.  Run after ligolw_add
#   - takes equal number from each file by default ... WARNING THIS MEANS WE CAN HAVE PATHOLOGICAL SIZE LIMITS


import sys
from optparse import OptionParser
import numpy as np
from ligo.lw import utils, table, lsctables, ligolw
try:
    import h5py
except:
    print(" - no h5py - ")

import RIFT.lalsimutils as lalsimutils

# Contenthandlers : argh
#   - http://software.ligo.org/docs/glue/
lsctables.use_in(ligolw.LIGOLWContentHandler)

optp = OptionParser()
optp.add_option("--fref",default=20,type=float,help="Reference frequency. Depending on approximant and age of implementation, may be ignored")
optp.add_option("--n-min",default=20,type=int,help="Minimum size of file to include")
optp.add_option("--output-file",default='merged_output',type=str,help="Merged output file")
opts, args = optp.parse_args()


P_list = []
P_list_list =[]
n_list = []
for fname in args:
    P_list_this_file = lalsimutils.xml_to_ChooseWaveformParams_array(fname)
    if len(P_list_this_file)<opts.n_min:
        continue
    P_list_list.append(P_list_this_file)
    n_list.append(len(P_list_this_file))
n_min = np.min(n_list)
for indx in np.arange(len(P_list_list)):
    indx_to_take = np.random.choice(np.arange(len(P_list_list[indx])),size=n_min)
    P_to_add = [P_list_list[indx][a] for a in indx_to_take]
    P_list += P_to_add

lalsimutils.ChooseWaveformParams_array_to_xml(P_list,opts.output_file)
