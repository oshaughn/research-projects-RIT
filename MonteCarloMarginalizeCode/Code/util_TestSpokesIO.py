#!/bin/env python
#
#  GOAL
#     - demonstrate spoke reconstruction from xml and from ILE *.dat output
#
#
# EXAMPLE
#    - util_TestSpokesIO.py --fname overlap-grid.xml.gz  --fname-dat ~/g184098-FlipFlop.dat --test-refinement
#    - util_TestSpokesIO.py   --fname-dat ~/g184098-RIT-Generic-v0dfix.dat  --verbose --test-refinement
import spokes

import argparse
import sys
import numpy as np
import lalsimutils

from matplotlib import pyplot as plt

###
### Load options
###

parser = argparse.ArgumentParser()
parser.add_argument("--fname", default=None, help="Base output file for XML loading")
parser.add_argument("--fname-dat",default=None)
parser.add_argument("--test-refinement",action='store_true')
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
opts=  parser.parse_args()



###
### Load xml grid
###

if opts.fname:
 sdHere = spokes.LoadSpokeXML(opts.fname)
 for key in sdHere:
    print " Spoke : ", key
    for P in sdHere[key]:
        mtot = spokes.ChooseWaveformParams_to_spoke_mass(P)  # used so rounding is consistent
        print mtot


###
### Load ILE .dat output.  REFINE test
###

if opts.fname_dat:
 sdHere = spokes.LoadSpokeDAT(opts.fname_dat)
 fig_index =0
 for key in sdHere:
    print " Spoke : ", key
    for spoke_entry in sdHere[key]:
        print spoke_entry
    print " --- cleaning --- "
    sdHereCleaned = spokes.CleanSpokeEntries(sdHere[key])
    for spoke_entry in sdHereCleaned:
        print spoke_entry
    if opts.test_refinement:
        print sdHereCleaned[:,0], sdHereCleaned[:,1]
        code, xvals_new = spokes.Refine(sdHereCleaned[:,0], sdHereCleaned[:,1])
        print code, xvals_new

    if opts.verbose:
        fig_index+=1
        plt.figure(fig_index)
#        plt.plot(sdHereCleaned[:,0], sdHereCleaned[:,1],'o')
        plt.errorbar(sdHereCleaned[:,0], sdHereCleaned[:,1],yerr=sdHereCleaned[:,2],linestyle='none')
        if opts.test_refinement:
            plt.plot(xvals_new, 200*np.ones(len(xvals_new)),'o')

if opts.verbose:
    plt.show()

###
### Refinement code: Call on each.  SHOW POINTS (vertical bars)
###

