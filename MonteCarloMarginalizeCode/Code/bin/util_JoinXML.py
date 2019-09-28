#! /usr/bin/env python
#
# USAGE
#  python util_JoinXML.py ~/overlap-grid.xml.gz  ~/overlap-grid.xml.gz  --output-file merged_output  --verbose X --force-approximant SEOBNRv2
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import numpy as np
import lal

rosDebug = True

lalsimutils.rosDebugMessagesContainer[0] = False

import optparse
parser = optparse.OptionParser()
parser.add_option("--output-file", default="merged_output", help="Filename, note xml.gz is appened automatically")
parser.add_option("--compare-parameter", action='append')
parser.add_option("--force-approximant",default=None)  # because there is a weird bug that sometimes hits me with type conversion
parser.add_option("--verbose")
opts, args = parser.parse_args()

# Loop over arguments, joining
P_list =[]
for fname in args:
    print(" File name : ", fname)
    P_list_here = lalsimutils.xml_to_ChooseWaveformParams_array(fname)
    print(" Read n= ", len(P_list_here))
    for P in P_list_here:
        P.tref = 0.0  # insure it is ok  -- there have been problems with malformed times. DANGEROUS if used for another purpose than mine!
    print(" Revised length before cutting: N = ", len(P_list_here))
    if opts.force_approximant:
        approx = lalsim.GetApproximantFromString(opts.force_approximant)
        print(" Forcing approx = ", approx,  " AKA ", lalsim.GetStringFromApproximant(approx), " from ", opts.force_approximant)
        for P in P_list_here:
            P.approx = approx
    P_list = P_list +  P_list_here

if opts.verbose:
    for P in P_list:
        P.print_params()
#        print P.approx


# Loop, removing duplicates sequentially
P_list_out =[]
n = len(P_list)
indxList = np.arange(n-1)
print(indxList)
param_names = opts.compare_parameter
for indx in indxList:
    print(" Testing ", indx, "out of ",  n)
    do_we_add_it = True
    for indx2 in np.arange(indx+1,n):
        are_all_parameters_the_same = True
        for param in param_names:
            param_val = P_list[indx].extract_param(param)
            param_val2 = P_list[indx2].extract_param(param)
            if rosDebug:
                print(param, param_val, param_val2)
            if not(param_val  == param_val2):
                are_all_parameters_the_same = False
        # conclude test for this index
        if are_all_parameters_the_same:
            do_we_add_it = False
            if rosDebug:
                print(" Match between ", indx, " and ",  indx2)
    if do_we_add_it:
        P_list_out = P_list_out + [P_list[indx]]

# Write output
print(" Saving ", len(P_list_out))
lalsimutils.ChooseWaveformParams_array_to_xml(P_list_out, fname=opts.output_file) # note xml.gz is appended
