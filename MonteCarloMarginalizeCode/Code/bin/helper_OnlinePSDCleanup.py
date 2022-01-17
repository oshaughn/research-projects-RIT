#! /usr/bin/env python
#
# GOAL
#   Online PSDs do not start at f0=0
#   We need to generate new PSDs (ideally one for each IFO) which do not do this.
#
# REFERENCES
#    test_psd_xml_io.py


import numpy as np
import RIFT.lalsimutils as lalsimutils
import lal
from optparse import OptionParser

from ligo.lw import utils

from lal.series import make_psd_xmldoc


optp = OptionParser()
optp.add_option("-p", "--psd-file", default="psd.xml.gz",help="instrument=psd-file, e.g. H1=H1_PSD.xml.gz. Can be given multiple times for different instruments.")
optp.add_option("-n", "--nyquist", type=float, default=2048.0, help="Set the nyquist frequency of the PSD. Default is 2048 Hz.")
optp.add_option("-i", "--ifo", action='append',help="Add instruments: --ifo H1 --ifo L1 ...")
opts, args = optp.parse_args()

ifos = opts.ifo

# Load in PSD
psd_dict_raw = {}
df = None
f0 = None
epoch = None
nyquist = None
npts_orig = None
for ifo in opts.ifo:
    print(" Reading ", opts.psd_file, " for ", ifo)
    psd_dict_raw[ifo] = lalsimutils.get_psd_series_from_xmldoc(opts.psd_file,ifo)
    npts_orig   = len(psd_dict_raw[ifo].data.data)
    df = psd_dict_raw[ifo].deltaF
    f0 = psd_dict_raw[ifo].f0
    nyquist = int (len(psd_dict_raw[ifo].data.data)*df+f0 )
    epoch = psd_dict_raw[ifo].epoch
    print(ifo, len(psd_dict_raw[ifo].data.data) , 1./psd_dict_raw[ifo].deltaF, nyquist)



npts_desired = int(nyquist/df + 0.5)
indx_start = int(f0/df + 0.5)

# Loop
for ifo in ifos:
    print(" Writing  for ", ifo)
    dat_here = psd_dict_raw[ifo].data.data
    psddict  = {}
    psd_s = lal.CreateREAL8FrequencySeries(name=ifo, epoch=epoch, f0=0, deltaF=df, sampleUnits="s", length=npts_desired)
    psd_s.data.data[indx_start:indx_start+npts_orig] = dat_here[:npts_orig-1]
    psddict[ifo] = psd_s

    xmldoc = make_psd_xmldoc(psddict)
    utils.write_filename(xmldoc, ifo+"-psd.xml.gz", gz=True)
