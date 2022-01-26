#! /usr/bin/env python
"""
util_PrintInjection --inj [inj]  # prints standard dump of properties
util_PrintInjection --inj [inj]  --psiJ --psiL --iota --m1  # print rows of these properties
"""

import argparse
import numpy as np
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
import lal
import sys
from ligo.lw import utils, lsctables, table, ligolw

cthdler = lalsimutils.cthdler  #ligolw.LIGOLWContentHandler #defines a content handler to load xml grids
lsctables.use_in(cthdler)


parser = argparse.ArgumentParser()
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing injection information.")
parser.add_argument("--event",type=int, dest="event_id", default=None,help="event ID of injection XML to use. If not specified loops over all")
parser.add_argument("--verbose", action="store_true",default=False)
parser.add_argument("--distance",action="store_true",default=False)
parser.add_argument("--right-ascension",action="store_true",default=False)
parser.add_argument("--declination",action="store_true",default=False)
parser.add_argument("--psiJ",action="store_true",default=False)
parser.add_argument("--psiL",action="store_true",default=False)
parser.add_argument("--iota",action="store_true",default=False)
parser.add_argument("--thetaJL",action="store_true",default=False)
parser.add_argument("--thetaJN",action="store_true",default=False)
parser.add_argument("--alpha",action="store_true",default=False)
parser.add_argument("--system-frame",action="store_true",default=True,help="If verbose, show system frame")
parser.add_argument("--force-morphology",type=int,default=3,help="If -1,0,1, select only that morphology, otherwise all")
parser.add_argument("--force-J-limit",type=float,default=-1,help="If a number, selects injections with J/M^2 less than it.")
parser.add_argument("--fref",type=float, default=0.0,help="For injections being interpreted at a different reference frequency than actually used in the file (historical)")
parser.add_argument("--show-fend",default=False,action='store_true')
opts=  parser.parse_args()

filename = opts.inj
event = opts.event_id
xmldoc = utils.load_filename(filename,contenthandler = cthdler, verbose = True)
Jlim = opts.force_J_limit


sim_inspiral_table = lsctables.SimInspiralTable.get_table(xmldoc)
if event is not None:
    events = [event]
else:
    events = np.arange( len(sim_inspiral_table))

P=lalsimutils.ChooseWaveformParams()
for event in events:
   P.copy_sim_inspiral(sim_inspiral_table[int(event)])
   if opts.fref:
       P.fref = opts.fref
   Jvec = P.TotalAngularMomentumAtReferenceOverM2()

   toprint = opts.verbose
   toprintHead = opts.verbose
   if not (opts.force_morphology == -1 or opts.force_morphology == 0  or opts.force_morphology == 1):
       toprint = opts.verbose
   else:
       if (opts.force_morphology == P.EvaluateMorphology()):
           toprintHead=True
       else:
           toprint = False
           
   if not( Jlim == -1) and np.sqrt(Jvec*Jvec)>Jlim:
       toprint=False
   else:
       toprintHead=True
   

       
   if toprint:
       if toprintHead:
                      print(" ----   Event number ", event , " ------")
       P.print_params(show_system_frame=opts.system_frame) # show_morphology=True, ,show_fend=opts.show_fend

 
