#! /usr/bin/env python

from itertools import combinations
import numpy as np
import fileinput, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prefix",default="",help="prefix appended (eg command) before each output line")
opts  = parser.parse_args()


#lines =list(map(lambda x:x.rstrip(),fileinput.input('')))
lines = list(map(lambda x:x.rstrip(), sys.stdin.readlines()))
lines 
#print(" Input ", lines)


res = list(map(list,combinations(lines,2)))
#print(res)
#res = list(map(lambda x:x.sorted(),res))
#print(res)
for indx in np.arange(len(res)):
   print("{} --samples {} --samples {} ".format(opts.prefix,res[indx][0], res[indx][1]))
