#! /usr/bin/env python
import numpy as np
import RIFT.lalsimutils as lalsimutils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--fname-reference",help="xml also")
parser.add_argument("--parameter",action='append',help="list of parameter names")
opts=  parser.parse_args()

P_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname)
P_ref_list= lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname_reference)

# Find the mean value of the reference point
parameters_to_center = ['mc', 'eta']   # could also add 'chieff_aligned', in cases with significant spin
if opts.parameter:
   parameters_to_center =opts.parameter
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
dat_out = []
for indx in np.arange(len(P_list)):
   bInclude = True
   vec = np.zeros(len(parameters_to_center))
   for pIndex in np.arange(len(parameters_to_center)):
      param =  parameters_to_center[pIndex]
      vec[pIndex] = valNew = dat[indx,pIndex]-dx[pIndex]
      if param == 'eta':
         if valNew > 0.25 or valNew < 0.001:
            bInclude=False
            continue
      if param == 's1z' or param == 's2z':
         if valNew > 1 or valNew < -1:
            bInclude=False
            continue
      P_list[indx].assign_param(param, valNew)
   P_list[indx].tref = float(P_list[indx].tref)
   P_out.append(P_list[indx])
   dat_out.append(vec)

print(" Recentering report: last two should be equal : ", np.mean(dat,axis=0),np.mean(dat_ref,axis=0),np.mean(dat_out,axis=0))

lalsimutils.ChooseWaveformParams_array_to_xml(P_out, "shifted.xml.gz")
