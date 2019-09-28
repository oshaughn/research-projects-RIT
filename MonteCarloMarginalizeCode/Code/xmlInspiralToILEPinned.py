#! /usr/bin/env python
"""
  xmlInspiralToILEPinned.py
   - Script to read an injction xml and generate ILE-style pinned parameter arguments, as desired
    ./xmlInspiralToILEPinned.py mdc.xml.gz 0 polarization inclination  phi_orb distance right_ascension declination t_ref
   - Intended for use in makefiles and command lines
   - Can provide any subset of parameters as needed
"""

from __future__ import print_function

import sys
#from lalsimutils import *

# https://www.lsc-group.phys.uwm.edu/daswg/projects/glue/doc/glue.ligolw-module.html
from glue.ligolw import table
from glue.ligolw import lsctables
from glue.ligolw import utils

import numpy as np

def mc(m1, m2):
    return  np.power(m1 * m2, (3./5.))/np.power(m1+m2,1./5.)

filename = sys.argv[1]
event = sys.argv[2]
params = sys.argv[2::]
xmldoc = utils.load_filename(filename, verbose = True)

sim_inspiral_table = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
row = sim_inspiral_table[int(event)]

if "right_ascension" in params:
    print(" --right-ascension ", row.longitude, end=' ')

if "declination" in params:
    print(" --declination ", row.latitude, end=' ')

if "inclination" in params:
    print(" --inclination ", row.inclination, end=' ')

if "polarization" in params:
    print(" --psi ", row.polarization, end=' ')


if "phi_orb" in params:
    print(" --phi-orb ", row.coa_phase, end=' ')

import lal
if "distance" in params:
    print(" --distance ", row.distance) # *lal.LAL_PC_SI*1e6,   # currently, idiotically

if "t_ref" in params:
    print(" --time ", row.geocent_end_time + 1.0e-9*row.geocent_end_time_ns)

#print " --iota", row.inclination, " --psi", row.polarization , " --dec", row.latitude, " --ra", row.longitude, " --dist", row.distance, " --phi", row.coa_phase
