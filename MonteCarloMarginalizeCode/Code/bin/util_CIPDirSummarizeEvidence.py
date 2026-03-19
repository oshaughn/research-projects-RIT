#! /usr/bin/env python

import numpy as np
import argparse
import glob


parser=argparse.ArgumentParser()
parser.add_argument("--cip-dir",help="CIP directory")
parser.add_argument("--internal-fix-double-log",action='store_true', help="Specific old code and container combinations had a double log applied to Z for AV output.")
parser.add_argument("--output",default="evidence.out")
parser.add_argument("--stream-output",action='store_true')
opts = parser.parse_args()

# withpriorchange versions: have correct lnL
# normal: have correct error (sometimes double log)
fnames_cip = glob.glob(opts.cip_dir+"/overlap-grid-*-*[0-9]+annotation.dat") # avoid the non-worker result
if len(fnames_cip) < 1:
    raise Exception(" No files for evidence in ", opts.cip_dir)
net_dat = []
for fname in fnames_cip:
    fname_alt =fname.replace('+annotation.dat', '_withpriorchange+annotation.dat')
    dat_base = np.genfromtxt(fname,names=True)
    sigma_lnL = dat_base['sigmaL']
    dat_alt = np.genfromtxt(fname_alt,names=True)
    lnL = dat_alt['lnL']
    n_eff = dat_alt['neff']
    net_dat.append([lnL, sigma_lnL, n_eff])
net_dat =np.array(net_dat)

lnL = np.average(lnL, weights=1./sigma_lnL**2)
sigma_lnL = np.max([np.sqrt(np.mean(sigma_lnL**2)/len(net_dat)),np.std(lnL)]) # not quite right but ok
dat_out = [lnL, sigma_lnL]
if opts.stream_output:
 print(*dat_out)
else:
 np.savetxt(opts.output, np.array([dat_out]), header=" lnL sigma_lnL")
