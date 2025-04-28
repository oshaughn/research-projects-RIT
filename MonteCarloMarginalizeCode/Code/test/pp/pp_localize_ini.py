#! /usr/bin/env python
# Example
#   /scripts/localize_ini.py --ifo-classifier review_test_data/4s/all_networks.json --event 0 --ini proto_4s.ini --sim-xml scripts/mdc.xml.gz --mc-file-settings here_mc_ranges --guess-snr 20 --duration 4
import json
import sys
import argparse
import RIFT.lalsimutils as lalsimutils
import numpy as np
import configparser

parser = argparse.ArgumentParser()
#parser.add_argument("--ifos",help="Space-separated list of IFO names, eg 'H1 L1 V1', to use for this event ")
parser.add_argument("--event",type=int)
parser.add_argument("--ini")
parser.add_argument("--sim-xml")
parser.add_argument("--guess-snr",type=float)
parser.add_argument("--mc-range")
parser.add_argument("--cip-sampler",type=str)
parser.add_argument("--ile-sampler",type=str)
parser.add_argument("--assume-nonprecessing",action='store_true')
parser.add_argument("--assume-fixed-sky-location",action='store_true')
#parser.add_argument("--mc-range-limits-snr-safety",type=float,default=8)
#parser.add_argument("--duration",type=int)
opts = parser.parse_args()

#
#ifos =None
print("Localizing ini file for event {}".format(opts.event))
# Get IFOs for it
#with open(opts.ifos,'r') as f:
#    dat_ifos = json.load(f)
#    ifos  = dat_ifos[opts.event] 

# Get event time from saved event object
# sim-xml only has one event remember
P=lalsimutils.xml_to_ChooseWaveformParams_array(opts.sim_xml)[opts.event]

# open mc file
#line = None
#mc_range=eval(opts.mc_range.replace(' ', '')) # completely eliminate whitespace

snr_cut = 30
snr_cut= float(snr_cut)

#########################
# Open defaults ini files
#########################
config=configparser.ConfigParser()
config.optionxform = str
config.read('pseudo_ini.ini')

#########################
# Change default settings
# if needed for current run
#########################
def modify_ini(config, section, option, new_val):
    # Check and modify the option if needed
    if (not config.has_option(section, option)) or (config.get(section, option) != new_val):
        config[section][option] = new_val
        print(f"{option} in {section} changed.")
        return True
    else:
        #print(f"{option} in {section} unchanged.")
        return False

changes_made = False
with open(opts.ini,'r') as f:
    line = f.readline()
    print_explode = False
    ## make NOT a while loop, just scan file
    for line in f:
        changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'force-hint-snr', str(opts.guess_snr))
        changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'event-time', str(P.tref))
        if 'cip-explode-jobs' in line:
            print(line.rstrip(), print_explode,file=sys.stderr)
            if not(print_explode):
                changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'cip-explode-jobs-auto', 'True')
                print_explode=True
            continue
        elif opts.guess_snr > 37.5:
            changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'use-downscale-early', 'True')
            changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'internal-ile-use-lnL', 'True')
            changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'internal-ile-sky-network-coordinates', 'True')
            changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'cip-sigma-cut', '0.7')
            if opts.guess_snr > 100:
                changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'internal-cip-use-lnL', 'True')
                changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'internal-ile-rotate-phase', 'True')
                changes_made |= modify_ini(config, 'rift-pseudo-pipe', 'ile-sampler-method', "'GMM'")

#########################
# Write new ini file with
# changed settings
#########################
if changes_made:
        with open('localize.ini', 'w') as f:
            config.write(f)
        print("File saved.")
exit(0)


#########################
# old localize
#########################
with open(opts.ini,'r') as f:
    line = f.readline()

    print_explode=False
    while line:
     if line.startswith('ifos='):
        print('ifos={}'.format(ifos))
     elif 'cip-explode-jobs' in line: #.startswith('cip-explode-jobs'):
        print(line.rstrip(), print_explode,file=sys.stderr)
        if not(print_explode):
           print('cip-explode-jobs-auto=True')  # basically always do this, override proto settings which were not well chosen
           print_explode=True
     elif opts.guess_snr > 37.5 and line.startswith('internal-ile-auto-logarithm-offset'):
         True # Don't use the logarithm offset when lnL is active for ILE
     elif line.startswith('internal-ile-sky-network-coordinates'):
        True # skip this line, set with IFO list at end
     else:
        clean_line = line.strip()
        if (clean_line):
          print(clean_line)
     line  = f.readline()

print("force-hint-snr={}".format(opts.guess_snr))
if opts.guess_snr<snr_cut:
    print("force-mc-range={}".format(opts.mc_range))
else:
    print("limit-mc-range={}".format(mc_range))

sky_printed=False
# Try to set most of the high-SNR options here. Distance marginalization will be left to the top-level script
if opts.ile_sampler != 'AV':
  if opts.guess_snr>37.5:
    print("use-downscale-early=True")
    print("internal-ile-use-lnL=True")
    if not(opts.fix_sky_location):
        print("internal-ile-sky-network-coordinates=True")
    print("cip-sigma-cut=0.7")
  if opts.guess_snr>100:
    print("internal-cip-use-lnL=True")
    print("internal-ile-rotate-phase=True")
    print("ile-sampler-method=GMM")
elif opts.ile_sampler == 'AV':
    print("internal-ile-use-lnL=True")  # automatic/redundant
    if opts.guess_SNR > 26:
        print("manual-extra-ile-args=--force-adapt-all")  # adapt in all parameters
#if len(ifos) == 2 and not sky_printed:
#    print("internal-ile-sky-network-coordinates=True")
print("event-time={}".format(P.tref))
