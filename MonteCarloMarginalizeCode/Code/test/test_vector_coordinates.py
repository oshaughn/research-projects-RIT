#! /usr/bin/env python
#
# GOAL
#   test convert_waveform_coordinates against


import numpy as np
import RIFT.lalsimutils as lalsimutils

npts=3

# Generate symthetic injections
P_list =[]
for indx in np.arange(npts):
    P = lalsimutils.ChooseWaveformParams()
    P.randomize()
    P_list.append(P)
npts = len(P_list)


# Cartesian test 1
coord_names=['mc','delta_mc','xi','chiMinus']
low_level_coord_names=['m1','m2','s1z','s2z'] # assume this is the underlying

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

print("Cartesian aligned test 2", np.max(np.abs(x1 - x2)))

# Cartesian test 2: delta_mc
coord_names=['mc','delta_mc','xi','chiMinus']
low_level_coord_names=['m1','m2','s1z','s2z'] # assume this is the underlying

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

print("Cartesian aligned test 2", np.max(np.abs(x1 - x2)))


# Cartesian test 3
coord_names=['mu1','mu2','q','chiMinus']
low_level_coord_names=['m1','m2','s1z','s2z'] # assume this is the underlying

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


# Polar test
coord_names=['mc','delta_mc','xi','chiMinus','s1x','s1y', 's2x', 's2y']
low_level_coord_names=['m1','m2','chi1','cos_theta1', 'phi1','chi2', 'cos_theta2', 'phi2'] # assume this is the underlying


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


# Cylinder pseudo-polar
coord_names=['mc','delta_mc','xi','chiMinus','s1x','s1y', 's2x', 's2y']
low_level_coord_names=['mc','delta_mc','chi1_perp_bar','s1z_bar', 'phi1','chi2_perp_bar', 's2z_bar', 'phi2'] # assume this is the underlying

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Cylinder test 1", np.max(np.abs(x1 - x2)))


coord_names=['mc','delta_mc','xi','chiMinus','s1x','s1y', 's2x', 's2y']
low_level_coord_names=['mc','delta_mc','chi1_perp_u','s1z_bar', 'phi1','chi2_perp_u', 's2z_bar', 'phi2'] # assume this is the underlying

for indx in np.arange(npts):
    P = P_list[indx]
    for indx_name  in np.arange(len(coord_names)):
        x1[indx,indx_name]  = P.extract_param( coord_names[indx_name])
    for indx_name2 in np.arange(len(low_level_coord_names)):
        y2[indx,indx_name2]  = P.extract_param( low_level_coord_names[indx_name2])

x2 = lalsimutils.convert_waveform_coordinates(y2, coord_names=coord_names, low_level_coord_names=low_level_coord_names)
print(x2)

print("Cylinder test 2", np.max(np.abs(x1 - x2)))

