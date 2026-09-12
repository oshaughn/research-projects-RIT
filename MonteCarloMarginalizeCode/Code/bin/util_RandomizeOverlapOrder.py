#! /usr/bin/env python
#
# util_RandomizeOverlapOrder.py
#
# 
# GOAL
#   - take an overlap.xml.gz (SimInspiralTable), and randomize its order.  
#   - Important when merging files from many workers, to avoid accidentally using only the output from one of them.  Run after ligolw_add
#   - takes equal number from each file by default ... WARNING THIS MEANS WE CAN HAVE PATHOLOGICAL SIZE LIMITS


#import sys
from optparse import OptionParser
import numpy as np
from igwn_ligolw import utils, lsctables, ligolw
# try:
#     import h5py
# except:
#     print(" - no h5py - ")

import RIFT.lalsimutils as lalsimutils

# Contenthandlers : argh
#   - http://software.ligo.org/docs/glue/
lsctables.use_in(ligolw.LIGOLWContentHandler)

optp = OptionParser()
optp.add_option("--fref",default=20,type=float,help="Reference frequency. Depending on approximant and age of implementation, may be ignored")
optp.add_option("--n-min",default=20,type=int,help="Minimum size of file to include. NOT USED")
optp.add_option("--output-file",default='merged_output',type=str,help="Merged output file")
optp.add_option("--preserve-block-order",action='store_true',help="Do NOT interleave across input files; append each file's block in file order. This is the pre-2026-08 behaviour and is only for reproducing runs made before the interleave fix.")
optp.add_option("--verbose",action='store_true',help="Print messages")
opts, args = optp.parse_args()


P_list = []
P_list_list =[]
n_list = []
for fname in args:
    P_list_this_file = lalsimutils.xml_to_ChooseWaveformParams_array(fname)
    if False: # len(P_list_this_file)<opts.n_min:
        if opts.verbose: print("  Skipping ", fname)
        continue
    P_list_list.append(P_list_this_file)
    n_list.append(len(P_list_this_file))


# Find mean and standard deviation
n_list = np.array(n_list)
n_mean = np.mean(n_list)
n_std = np.std(n_list)+0.5 # in case all equal !
indx_good = n_list > np.max([n_mean  - 3*n_std, 1])  # reject points with too few points. So at least 2 per worker hard minimum.
indx_valid = np.arange(len(n_list),dtype=int)[indx_good]
P_list_list = [P_list_list[k] for k in indx_valid] # dumb
n_list = n_list[indx_good]
    
if len(n_list) <1:
    raise Exception(" util_RandomizeOverlapOrder : Failure, not enough points in each file, after stripping outliers ")
n_min = np.min(n_list)
for indx in np.arange(len(P_list_list)):
    indx_to_take = np.random.choice(np.arange(len(P_list_list[indx])),size=n_min,replace=False) # do not take duplicate entries from any file. Files may be small!
    P_to_add = [P_list_list[indx][a] for a in indx_to_take]
    P_list += P_to_add

# Interleave ACROSS files.  The draw above randomises only WITHIN each file, and the blocks are
# appended in file order, so the merged output is [file0][file1]...[fileN].  That is invisible to a
# consumer that reads the whole file, but the nested ILE reads only a PREFIX (measured: the first
# 3000 rows), so it saw just the first few workers -- with 25 workers x 800, four of them, and with
# 6 x 3334, only the first.  That is precisely the "accidentally using only the output from one of
# them" failure this tool exists to prevent.  The hyperpipeline branch of write_joingrids_sub
# already does this ("shuffle so spokes are interleaved"); this brings the XML path into line.
if not opts.preserve_block_order:
    P_list = [P_list[k] for k in np.random.permutation(len(P_list))]

lalsimutils.ChooseWaveformParams_array_to_xml(P_list,opts.output_file)
