#! /usr/bin/env python
#
# util_AddExtrXML.py
# 
# GOAL
#   - to join XML files of ILE extrinsic output  
#   - uses ligolw_add, but combines the XML files in stages by finding a pattern from the filename string
# EXAMPLE
#   - util_AddExtrXML.py iteration_6_ile/EXTR_out-*.xml_*_.xml.gz --output OUTPUT.xml.gz


import sys
from optparse import OptionParser
import numpy as np
try:
    import h5py
except:
    print(" - no h5py - ")

import glob
import os
import subprocess 
import itertools


def chunk(arr_range, arr_size):
    arr_range = iter(arr_range)
    return iter(lambda: list(itertools.islice(arr_range, arr_size)), ())

optp = OptionParser()
optp.add_option("--output",default='merged_output',type=str,help="Merged output file")
opts, args = optp.parse_args()

P_list = []
for pat in args:
    fname_list = glob.glob(pat)
    P_list += fname_list
print("Strings identified")


batches = []
while P_list:
    if len(P_list) > 1000:
      new_batch, P_list = P_list[:1000],P_list[1000:]
      batches += [new_batch]
    else:
      batches += [P_list]
      P_list = []

P_list_list=[' ' for i in range(len(batches))]
for j in np.arange(0,len(batches)):
    for i in batches[j]:
        P_list_list[j] += ' '+ i
print("List of strings created")

for i in range(len(P_list_list)):
    print(i, ' EXTR-INTERIM_'+str(i)+'.xml.gz', batches[i][:2])
    Grid = subprocess.run(['ligolw_add '+ P_list_list[i]+ ' --output  EXTR-INTERIM_'+str(i)+'.xml.gz'], shell=True,capture_output=True)
print("Intermediate grids generated; Joining all now")

subprocess.run(['ligolw_add EXTR-INTERIM_*.xml.gz --output '+ opts.output],shell=True, capture_output=True, text=True)
for name in glob.iglob(os.getcwd()+'/EXTR-INTERIM_*.xml.gz', recursive=True):
    os.remove(name)

