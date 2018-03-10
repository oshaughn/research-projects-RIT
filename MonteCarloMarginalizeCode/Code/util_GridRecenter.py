#! /usr/bin/env python
import numpy as np
import lalsimutils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--fname-reference",help="xml also")
opts=  parser.parse_args()

P_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname)
P_ref_list= lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname_reference)

# Find the mean value of the reference point
parameters_to_center = ['mc', 'eta']
dat = np.zeros((len(P_list), len(parameters_to_center)))
for indx in np.arange(len(P_list)):
   for pIndex in np.arange(len(parameters_to_center)):
	param = parameters_to_center[pIndex]
	val = P_list[indx].extract_param(param)
	dat[indx,pIndex]=val
dat_ref = np.zeros((len(P_ref_list), len(parameters_to_center)))
for indx in np.arange(len(P_ref_list)):
   for pIndex in np.arange(len(parameters_to_center)):
        param = parameters_to_center[pIndex]
        val = P_ref_list[indx].extract_param(param)
        dat_ref[indx,pIndex]=val

P_out = []
dx = np.mean(dat,axis=0) - np.mean(dat_ref,axis=0)
for indx in np.arange(len(P_list)):
   bInclude = True
   for pIndex in np.arange(len(parameters_to_center)):
	param =  parameters_to_center[pIndex]
	valNew = dat[indx,pIndex]+dx[pIndex]
        if param == 'eta':
	    if valNew > 0.25 or valNew < 0.001:
		bInclude=False
		continue
	P_list[indx].assign_param(param, valNew)
   P_list[indx].tref = float(P_list[indx].tref)
   P_out.append(P_list[indx])

lalsimutils.ChooseWaveformParams_array_to_xml(P_out, "shifted.xml.gz")
