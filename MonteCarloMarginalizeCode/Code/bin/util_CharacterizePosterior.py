#! /bin/bash
# Very very quick tool to yank out posterior quantiles from an ascii file
#  Example

#  util_CharacterizePosterior.py --parameter mc --quantiles '[0.9]'



import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--fname",type=str,help="filename of *.dat file [standard ILE output]")
parser.add_argument("--parameter",action='append',help="name of param")
parser.add_argument("--quantiles",type=str,help="quantiles")
opts=  parser.parse_args()

#print opts.parameter

quantile_list = np.array(eval(opts.quantiles))
#print opts.quantiles, quantile_list

dat = np.genfromtxt(opts.fname,names=True)
for param in opts.parameter:
    # lightweight conversion: not full capability
    if param in dat.dtype.names:
        dat_1d = dat[param]
    elif param == 'mc':
        import RIFT.lalsimutils
        dat_1d = RIFT.lalsimutils.mchirp(dat['m1'],dat['m2'])
    quant_here  = np.percentile(dat_1d,100*quantile_list)
    print param, ' '.join(map(str,quant_here))
