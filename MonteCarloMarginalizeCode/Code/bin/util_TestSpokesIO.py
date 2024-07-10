#!/usr/bin/env python
#
#  GOAL
#     - demonstrate spoke reconstruction from xml and from ILE *.dat output
#
#
# EXAMPLE
#    - util_TestSpokesIO.py --fname overlap-grid.xml.gz  --fname-dat ~/g184098-FlipFlop.dat --test-refinement
#    - util_TestSpokesIO.py   --fname-dat ~/g184098-RIT-Generic-v0dfix.dat  --verbose --test-refinement
# EXAMPLE FULL REFINEMENT
#   python util_TestSpokesIO.py   --fname-dat ~/g184098-FlipFlop.dat  --fname overlap-grid.xml.gz   --test-refinement
import RIFT.misc.spokes as spokes

import argparse
import sys
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lal

try:
        from matplotlib import pyplot as plt
except:
        print(" no plots")
###
### Load options
###

parser = argparse.ArgumentParser()
parser.add_argument("--fname", default=None, help="Base output file for XML loading")
parser.add_argument("--fname-dat",default=None)
parser.add_argument("--is-eccentric",action='store_true',default=False)
parser.add_argument("--test-refinement",action='store_true')
parser.add_argument("--save-refinement-fname", default="refined-grid",help="Output to save refined grid xml. REQUIRES valid pairing")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--mega-verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
opts=  parser.parse_args()



###
### Load xml grid
###

if opts.fname and opts.mega_verbose:
 sdHere = spokes.LoadSpokeXML(opts.fname,is_eccentric=opts.is_eccentric)
 for key in sdHere:
    print(" Spoke : ", key)
    for P in sdHere[key]:
        mtot = spokes.ChooseWaveformParams_to_spoke_mass(P)  # used so rounding is consistent
        print(mtot)


###
### Load ILE .dat output.  REFINE test
###

if opts.fname_dat:
 sdHere = spokes.LoadSpokeDAT(opts.fname_dat,is_eccentric=opts.is_eccentric)
 print(" Spoke count : ", len(sdHere.keys()))
 fig_index =0
 for key in sdHere:
    sdHereCleaned = spokes.CleanSpokeEntries(sdHere[key])
    print(" Spoke ", key, len(sdHereCleaned))
    if opts.mega_verbose:
     for spoke_entry in sdHere[key]:
        print(spoke_entry)
     print(" --- cleaning --- ")
     for spoke_entry in sdHereCleaned:
        print(spoke_entry)
    if opts.test_refinement:
        #print sdHereCleaned[:,0], sdHereCleaned[:,1]
        code, xvals_new = spokes.Refine(sdHereCleaned[:,0], sdHereCleaned[:,1],xmin=1)  
        if xvals_new is None:
                continue
        xvals_new = np.array(xvals_new)
        xvals_new = xvals_new[ xvals_new > 0]  # eliminate negtive masses!

        if opts.mega_verbose:
           print(code, xvals_new)

    if opts.verbose:
        fig_index+=1
        plt.figure(fig_index)
#        plt.plot(sdHereCleaned[:,0], sdHereCleaned[:,1],'o')
        plt.text(np.mean(sdHereCleaned[:,0])-10,np.max(sdHereCleaned[:,1]), str(key))
        plt.errorbar(sdHereCleaned[:,0], sdHereCleaned[:,1],yerr=sdHereCleaned[:,2],linestyle='none')
        if opts.test_refinement and code!='fail':
            plt.plot(xvals_new, 200*np.ones(len(xvals_new)),'o')

        xvals_min_here = np.min(sdHereCleaned[:,0])
        xvals_max_here = np.max(sdHereCleaned[:,0])
        fn = spokes.FitSpokeNearPeak(sdHereCleaned[:,0], sdHereCleaned[:,1])
        xvals_plot = np.linspace(xvals_min_here,xvals_max_here,100)
        plt.plot(xvals_plot,fn(xvals_plot),'k')

if opts.verbose:
    plt.show()


###
### Use DAT + XML lookup to generate refinement, per spoke
###


if opts.fname and opts.fname_dat:
   sd_dat = spokes.LoadSpokeDAT(opts.fname_dat,is_eccentric=opts.is_eccentric)
   sd_P =  sdHere = spokes.LoadSpokeXML(opts.fname,is_eccentric=opts.is_eccentric)

   print(" --- Refinement: Writing XML --- ")
   print(" data file ", len(sd_dat))
   print(" xml file ", len(sd_P))

   P_list = []
   nCount = 0;
   nFailures = 0
   for spoke_id in sd_dat.keys():
      nCount +=1
      # Cross-look-up
      print(spoke_id)
      try:
         P_sample = sd_P[spoke_id][0] # if this fails
         P_sample.waveFlags =None
         P_sample.nonGRparams = None
         P_sample.print_params()
      except:
         nFailures +=1
         print(" Failed cross lookup for ", spoke_id, nCount, " failure count = ", nFailures)
         continue
      # Clean
      sd_here =spokes.CleanSpokeEntries(sd_dat[spoke_id])
      # Refine: find mass values
      code, mvals_new = spokes.Refine(sd_here[:,0], sd_here[:,1])
      if mvals_new is None:
              continue  # Failed
      mvals_new = np.array(mvals_new)
      mvals_new = mvals_new[ mvals_new > 0]  # eliminate negtive masses!
      print(key, len(sd_here), code, mvals_new)
      if code == 'refined' or code =='extended':
         for m in mvals_new:
            print(m)
            P = P_sample.manual_copy()
            P.tref = P_sample.tref   # BE VERY CAREFUL: The structure swig-bound to store time is NOT a float, and requires careful memory management
            P.assign_param('mtot',m*lal.MSUN_SI)
            P_list.append(P)
            
   if P_list:
      lalsimutils.ChooseWaveformParams_array_to_xml(P_list, opts.save_refinement_fname)
