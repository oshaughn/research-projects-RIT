#! /usr/bin/env python
#  convert_angular_to_cartesian.py
#
#  does a batch call to assign spin DOF and then convert them
#  if not set, parameter values are zero


import lal
import RIFT.lalsimutils as lalsimutils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--parameter",action='append')
parser.add_argument("--parameter-value",type=float,action='append')
parser.add_argument("--output",type=str,default="converted_from_polar")
opts= parser.parse_args()

param_names = opts.parameter
param_values = []
if len(param_names) == len(opts.parameter_value):
    param_values = opts.parameter_value #list(map(eval, opts.parameter_value))

print(param_names)
print(param_values)

my_vals = {}
for name in ['thetaJN', 'phiJL','theta1', 'theta2', 'phi12', 'chi1', 'chi2','psiJ','fref', 'm1', 'm2','phiref','mc','q']:
    if name in param_names:
        indx = param_names.index(name)
        my_vals[name] = param_values[indx]
    else:
        my_vals[name] =0
if not 'fref' in param_names:
    my_vals['fref']=20

my_vals['m1'] *= lal.MSUN_SI
my_vals['m2'] *= lal.MSUN_SI
my_vals['mc'] *= lal.MSUN_SI
    
P=lalsimutils.ChooseWaveformParams()

if 'm1' in param_names:
    P.m1 = float(my_vals['m1'])
    P.m2 = float(my_vals['m2'])
else:
    eta = lalsimutils.symRatio(1,my_vals['q'])
    P.assign_param('mc', float(my_vals['mc']))
    P.assign_param('eta', eta)
P.fref = my_vals['fref']
P.phiref = my_vals['phiref']


m1 =P.m1/lal.MSUN_SI
m2 =P.m2/lal.MSUN_SI

del(my_vals['fref'])  # not allowed
del(my_vals['m1'])
del(my_vals['m2'])
del(my_vals['mc'])
del(my_vals['q'])
del(my_vals['phiref'])
P.init_via_system_frame(**my_vals)


# Write command line, which MUST BE FIRST (write script modifies it?)
print(" --parameter m1 --parameter-value {} --parameter m2 --parameter-value {} --parameter s1x --parameter-value {} --parameter s1y --parameter-value {} --parameter s1z --parameter-value {} --parameter s2x --parameter-value {} --parameter s2y --parameter-value {} --parameter s2z --parameter-value {} --parameter fref --parameter-value {} --parameter phiref --parameter-value {}".format(m1,m2, P.s1x,P.s1y,P.s1z,P.s2x,P.s2y,P.s2z, P.fref, P.phiref))

# Write general xml file in the desired coordinate system and with fref
lalsimutils.ChooseWaveformParams_array_to_xml([P], opts.output)

