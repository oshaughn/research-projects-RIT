#! /usr/bin/env python
import argparse
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--cache", default="zero_noise.cache")
parser.add_argument("--psd",type=str,default="lalsim.SimNoisePSDaLIGOZeroDetHighPower",help="psd name ('eval'). lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, ... ")
parser.add_argument("-p", "--psd-file", action="append", help="instrument=psd-file, e.g. H1=H1_PSD.xml.gz. Can be given multiple times for different instruments.")
parser.add_argument("--fmin-snr",  type=float, default=10)
parser.add_argument("--fmax-snr", type=float, default=1500)
parser.add_argument("--plot-sanity",action='store_true')
opts=  parser.parse_args()

fminSNR = opts.fmin_snr
fmaxSNR = opts.fmax_snr

psd_name = {}
psd_dict = {}
if opts.psd_file:
    for inst, psdf in map(lambda c: c.split("="), opts.psd_file):
        psd_name[inst] = psdf


ifo_list = ["H1", "L1", "V1"]
if opts.psd_file:
    ifo_list = psd_name.keys()

data_dict ={}
rho_dict ={}
rho2Net =0
analyticPSD_Q=True
for ifo in ifo_list:
  try:
    channel = ifo+":FAKE-STRAIN"
    data_dict[ifo] =  lalsimutils.frame_data_to_non_herm_hoff(opts.cache,channel)
    fSample = len(data_dict[ifo].data.data)*data_dict[ifo].deltaF
    df = data_dict[ifo].deltaF
    if not (ifo in psd_name):
        print(ifo, " analytic psd ", opts.psd)
        analyticPSD_Q=True
        psd_dict[ifo] = eval(opts.psd)
    else:
        analyticPSD_Q=False
        print("Reading PSD for instrument %s from %s" % (ifo, psd_name[ifo]))
        psd_dict[ifo] = lalsimutils.load_resample_and_clean_psd(psd_name[ifo], ifo, df)
    IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[ifo],fMax=fmaxSNR,analyticPSD_Q=analyticPSD_Q)
    rhoDet = rho_dict[ifo] = IP.norm(data_dict[ifo])
    print(ifo, rho_dict[ifo])
    rho2Net += rhoDet*rhoDet
    if opts.plot_sanity:
        fvals = lalsimutils.evaluate_fvals(data_dict[ifo])
        plt.plot(fvals, np.power(np.abs(data_dict[ifo].data.data),2) *IP.weights2side)
        plt.xlim(-5000,5000)
        plt.figure(1)
        plt.savefig("frameplot_power_"+ifo+".png"); plt.clf()
        plt.figure(2)
        plt.plot(np.log10(fvals), np.log10(1./IP.weights2side)/2)
        plt. savefig("frameplot_psd_"+ifo+".png"); plt.clf()
    data_dict[ifo] = None  # clear it
  except:
      print(" No IFO ", ifo)
rho_dict['Network']=    np.sqrt(rho2Net )

print(rho_dict)
import json
with open("snr-report.txt", 'w') as f:
    json.dump(rho_dict, f)
    f.flush()

