#! /usr/bin/env python
#
# USAGE
#  python util_TestXMLParameterStorage.py ; ligolw_print -t sim_inspiral -c alpha5 output.xml.gz
import RIFT.lalsimutils as lalsimutils
import numpy as np
import lal

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output-file", default="output")
opts = parser.parse_args()

P_list_in =[]
for i in np.arange(5):
    P_list_in.append( lalsimutils.ChooseWaveformParams(m1=lal.MSUN_SI*(i+1)))
    P_list_in[-1].lambda1 = 3*i+1
    P_list_in[-1].lambda2 = 3*i+2
    P_list_in[-1].approx = 3*i+2
    P_list_in[-1].print_params()

lalsimutils.ChooseWaveformParams_array_to_xml(P_list_in, fname=opts.output_file)

print(" ------ ")
P_list_out = lalsimutils.xml_to_ChooseWaveformParams_array(opts.output_file+".xml.gz")

for P in P_list_out:
    P.print_params()
