#! /usr/bin/env python
# util_WriteInjectionFile.py
# 
#  Lets user create injection xml files from the command line.  More efficient than monoblock python codes
#  User has option to create parameters identical to NR injections (i.e., correct spin values)
#
# EXAMPLES
#   /util_WriteInjectionFile.py  --group Sequence-SXS-All --param 1 --parameter mtot --parameter-value 70 --parameter dist --parameter-value 1000



import argparse
import sys
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import functools

parser = argparse.ArgumentParser()
# Parameters
parser.add_argument("--parameter", action='append', help='Explicit list of parameters to use')
parser.add_argument("--parameter-value", action="append",help="Explicit list of parameter values to use")
parser.add_argument("--fname",default="mdc",help="Injection xml name (.xml.gz will be added)")
parser.add_argument("--approximant",default="TaylorT4",help="Set approximant. Useful for creating LAL comparison injections.")
# Add option to use NR waveforms!
parser.add_argument("--group",default=None)
parser.add_argument("--param",default=None)
opts=  parser.parse_args()

hasNR = False
if opts.group:
    import NRWaveformCatalogManager3 as nrwf
    hasNR=True


P=None
if not hasNR:
    P=lalsimutils.ChooseWaveformParams()
else:
    param = opts.param
    if nrwf.internal_ParametersAreExpressions[opts.group]:
        param = eval(param)
    wfP = nrwf.WaveformModeCatalog(opts.group, param, metadata_only=True)
    P = wfP.P

# default sky location, etc [will be overridden]
P.radec = True
P.theta = np.pi/2
P.phi = 0
P.phiref = 0
P.psi = 0
P.approx = lalsimutils.lalsim.GetApproximantFromString(opts.approximant)  # allow user to override the approx setting. Important for NR followup, where no approx set in sim_xml!

param_names = opts.parameter
for param in param_names:
    # Check if in the valid list
    if not(param in lalsimutils.valid_params):
        print(' Invalid param ', param, ' not in ', lalsimutils.valid_params)
        sys.exit(0)

param_values = []
if len(param_names) == len(opts.parameter_value):
    param_values = list(map(eval, opts.parameter_value))
for p in ['mc', 'm1', 'm2', 'mtot']:
    if p in param_names:
        indx = param_names.index(p)
        param_values[indx]= np.array(param_values[indx])* lal.MSUN_SI

for p in ['dist']:
    if p in param_names:
        indx = param_names.index(p)
        param_values[indx]= np.array(param_values[indx])* lal.PC_SI*1e6


for indx in np.arange(len(param_values)):
    P.assign_param(param_names[indx], param_values[indx])

print(" ------- WRITING INJECTION FILE ----- ")
P.print_params()

P_list = [P]
lalsimutils.ChooseWaveformParams_array_to_xml(P_list, fname=opts.fname, fref=P.fref)
