#! /usr/bin/env python


import argparse

import lal
import numpy as np
import RIFT.lalsimutils as lalsimutils

import NRWaveformCatalogManager3 as nrwf

###
### Get hdot
###

parser = argparse.ArgumentParser()
# Parameters
parser.add_argument("--group", default="Sequence-GT-Aligned-UnequalMass",help="Specific NR simulation group to use")
parser.add_argument("--param", default=(0.0,2.),help="Parameter value")
parser.add_argument("--lmax",default=2,type=int)
opts=  parser.parse_args()



wfP = nrwf.WaveformModeCatalog(opts.group, opts.param, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, lmax=opts.lmax,align_at_peak_l2_m2_emission=True,build_strain_and_conserve_memory=True)
wfP.P.dist = 1e6*lal.PC_SI  # avoid underflow
hlmT = wfP.hlmoft()
hlmT_dot = wfP.hlmoft()  # allocate fresh copy to hold derivatives
hlmT_memory_integrand = wfP.hlmoft()  # allocate another copy to hold results
hlmT_memory = wfP.hlmoft()  # allocate another copy to hold results
for pair in hlmT:
    dt = hlmT[pair].deltaT
    dat = hlmT[pair].data.data
    hlmT_dot[pair].data.data = (dat - np.roll(dat,-1))/dt
    hlmT_memory_integrand[pair].data.data *=0  # zero out 
    hlmT_memory[pair].data.data *=0  # zero out 

# Expression 0: Aligned spin (being lazy: Eq. 2.8 in 1108.3121
# Should find python code for Clebsh-Gordon coefficients
R_in_sec  = wfP.P.dist / lal.C_SI
hlmT_memory_integrand[(2,0)].data.data = (R_in_sec)* np.sqrt(15./(2*np.pi))/42 * ( 2* np.power(np.abs(hlmT_dot[(2,2)].data.data),2)  - 2*np.power(np.abs(hlmT_dot[(2,0)].data.data),2))
hlmT_memory_integrand[(2,2)].data.data = (R_in_sec)* np.sqrt(15./(2*np.pi))/21 * hlmT_dot[(2,2)].data.data  * hlmT_dot[(2,0)].data.data
hlmT_memory_integrand[(2,-2)].data.data = (R_in_sec)* np.sqrt(15./(2*np.pi))/21 * hlmT_dot[(2,-2)].data.data  * hlmT_dot[(2,0)].data.data
if (4,4) in hlmT.keys():
    hlmT_memory_integrand[(4,0)].data.data = (R_in_sec)* np.sqrt(5./(2*np.pi))/1260 * (  np.power(np.abs(hlmT_dot[(2,2)].data.data),2)  +3*np.power(np.abs(hlmT_dot[(2,0)].data.data),2))
    hlmT_memory_integrand[(4,2)].data.data = (R_in_sec)* np.sqrt(3./(2*np.pi))/252 * hlmT_dot[(2,2)].data.data  * hlmT_dot[(2,0)].data.data
    hlmT_memory_integrand[(4,-2)].data.data = (R_in_sec)* np.sqrt(3./(2*np.pi))/252 * hlmT_dot[(2,-2)].data.data  * hlmT_dot[(2,0)].data.data
    hlmT_memory_integrand[(4,4)].data.data = (R_in_sec)* np.sqrt(14./(2*np.pi))/504 * np.power(hlmT_dot[(2,2)].data.data ,2)
    hlmT_memory_integrand[(4,-4)].data.data = (R_in_sec)* np.sqrt(14./(2*np.pi))/504 * np.power(hlmT_dot[(2,-2)].data.data ,2)

# Integrate term by term
for pair in hlmT_memory:
    dat =     hlmT_memory_integrand[pair].data.data
    dt = hlmT_memory_integrand[pair].deltaT
    hlmT_memory[pair].data.data = np.cumsum(dat)*dt

    print(pair, np.max(hlmT_memory[pair].data.data), np.max(np.abs(hlmT[pair].data.data)), np.max(np.abs(hlmT_dot[pair].data.data)))


# Save
for pair in hlmT_memory:
    tvals = float(hlmT_memory[pair].epoch) + hlmT_memory[pair].deltaT*np.arange(hlmT_memory[pair].data.length)#lalsimutils.evalate_tvals(hlmT_memory[pair])
    np.savetxt("memory_"+str(pair).replace(' ','')+".dat", np.array([tvals, np.real(hlmT_memory[pair].data.data)]).T)
    np.savetxt("h_"+str(pair).replace(' ','')+".dat", np.array([tvals, np.real(hlmT[pair].data.data),np.imag(hlmT[pair].data.data)]).T)
