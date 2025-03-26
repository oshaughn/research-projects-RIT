#! /usr/bin/env python
#
# GOAL
#   test convert_waveform_coordinates against


import numpy as np
import RIFT.lalsimutils as lalsimutils

import optparse
parser = optparse.OptionParser()
parser.add_option("--npts",type=int,default=3)
parser.add_option("--as-test",action='store_true')
parser.add_option("--verbose",action='store_true')
opts, args = parser.parse_args()


npts=opts.npts

# Generate symthetic injections
P_list =[]
for indx in np.arange(npts):
    P = lalsimutils.ChooseWaveformParams()
    P.randomize()
    P_list.append(P)
npts = len(P_list)


# Cartesian test 0
coord_names=['mc','delta_mc', 'xi', 'chiMinus']
low_level_coord_names=['mc','eta','s1z','s2z'] # assume this is the underlying.  This setup is very common

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Cartesian aligned test 0", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")


# Cartesian test 1
coord_names=['mc','delta_mc','xi','chiMinus']
low_level_coord_names=['mc','delta_mc','s1z','s2z'] # assume this is the underlying.  This setup is very common

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Cartesian aligned test 1", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")

# Cartesian aligned test 2: NOT USED CONVERSION, falls back to default converter
# coord_names=['mc','delta_mc','xi','chiMinus']
# low_level_coord_names=['mc','eta','s1z','s2z'] # assume this is the underlying.  This setup is very common

# x1 = np.zeros((npts,len(coord_names)))
# x2 = np.zeros((npts,len(coord_names)))
# y2 = np.zeros((npts,len(low_level_coord_names)))

# for indx in np.arange(npts):
#     P = P_list[indx]
#     for indx_name  in np.arange(len(coord_names)):
#         x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
#     for indx_name2 in np.arange(len(low_level_coord_names)):
#         y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

# x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
# print(x2)

# print("Cartesian aligned test 2", np.max(np.abs(x1 - x2)))
# err = np.max(np.abs(x1 - x2))
# if opts.as_test and err > 1e-9:
#     raise ValueError(" Large deviation seen ")

# tidal test (puffball).  Note this is NOT ONE TO ONE
coord_names = ['mc', 'eta',   'chi1', 's1z', 's2z']
low_level_coord_names = ['mc', 'eta',  'chieff_aligned']

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)

print("Cartesian aligned test 3 (puffball): NOT ONE TO ONE AT LOW LEVEL, OK ", np.max(np.abs(x1 - x2)))
# err = np.max(np.abs(x1 - x2))
# if opts.as_test and err > 1e-9:
#     raise ValueError(" Large deviation seen ")


# Cartesian precessing test A: chi_p etc
coord_names=['mc','delta_mc','xi','chiMinus','chi_p']
low_level_coord_names=['mc','delta_mc','s1z','s2z','s1x','s1y','s2x', 's2y'] # assume this is the underlying

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

err = np.max(np.abs(x1 - x2))
print("Cartesian precessing test A", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")

# Cartesian test 2b: chi_p
coord_names=['mc','delta_mc','xi','chiMinus','chi_p']
low_level_coord_names=['mc','delta_mc','s1z','s2z','s1x','s1y','s2x', 's2y'] # assume this is the underlying

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

err = np.max(np.abs(x1 - x2))
print("Cartesian precessing test B", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")


# Cartesian test 3
coord_names=['mu1','mu2','delta_mc','chiMinus']
low_level_coord_names=['mc','delta_mc','s1z','s2z'] # assume this is the underlying

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Cartesian aligned test 3", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")


# Polar test
coord_names=['mc','delta_mc','xi','chiMinus','s1x','s1y', 's2x', 's2y']
low_level_coord_names=['mc','delta_mc','chi1','cos_theta1', 'phi1','chi2', 'cos_theta2', 'phi2'] # assume this is the underlying


x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Polar test 1", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")


# Cylinder pseudo-polar
coord_names=['mc','delta_mc','xi','chiMinus','s1x','s1y', 's2x', 's2y', 'chi_p']
low_level_coord_names=['mc','delta_mc','chi1_perp_bar','s1z_bar', 'phi1','chi2_perp_bar', 's2z_bar', 'phi2'] # assume this is the underlying

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Cylinder test 1", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")


coord_names=['mc','delta_mc','xi','chiMinus','chi_p','s1x','s1y', 's2x', 's2y']
low_level_coord_names=['mc','delta_mc','chi1_perp_u','s1z_bar', 'phi1','chi2_perp_u', 's2z_bar', 'phi2'] # assume this is the underlying

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Cylinder test 2", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")


coord_names=['chi1', 'chi2', 'mc', 'eta', 'xi']
low_level_coord_names=['mc','eta','chi1_perp_u','s1z_bar', 'phi1','chi2_perp_u', 's2z_bar', 'phi2'] # assume this is the underlying

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Cylinder test 3", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")


# Eccentric test
#   - assign variables
for P in P_list:
    P.eccentricity = np.random.uniform(0,0.3)
    P.meanPerAno = np.random.uniform(0, 2*np.pi)


coord_names=['eccentricity', 'meanPerAno', 'ecc_cos_meanPerAno', 'ecc_sin_meanPerAno']
low_level_coord_names=['eccentricity', 'meanPerAno'] # assume this is the underlying

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)

print("Eccentric test ", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")


# tidal test (CIP : LambdaTilde, DeltaLambdaTilde from lambda1, lambda2)
coord_names =['mu1','mu2', 'delta_mc', 'LambdaTilde', 'DeltaLambdaTilde', 'xi', 'chiMinus']
low_level_coord_names = ['mc', 'delta_mc', 's1z', 's2z', 'lambda1', 'lambda2']

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)

print("Tidal test 1 ", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")



# tidal test (puffball)
coord_names = ['mc', 'eta',  'lambda1', 'lambda2']
low_level_coord_names = ['mc', 'eta',  'DeltaLambdaTilde', 'LambdaTilde']

x1 = np.zeros((npts,len(coord_names)))
x2 = np.zeros((npts,len(coord_names)))
y2 = np.zeros((npts,len(low_level_coord_names)))

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)

print("Tidal test 2 ", np.max(np.abs(x1 - x2)))
err = np.max(np.abs(x1 - x2))
if opts.as_test and err > 1e-9:
    raise ValueError(" Large deviation seen ")
