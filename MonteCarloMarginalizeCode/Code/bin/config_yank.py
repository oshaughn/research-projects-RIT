#! /usr/bin/env python
#

# SHELL SCRIPT VERSION
# RUNDIR=$1

#   grep arguments ${RUNDIR}/ILE.sub |  tr '"' ' ' | sed 's/--/\n/g' > tmp_file_ile.args
#   grep arguments ${RUNDIR}/fitting-final.sub |  tr '"' ' ' | sed 's/--/\n/g' > tmp_file_cip.args

# for i in fmin-template channel-name fmin-ifo d-max approx event-time
# do
#   echo $i `grep $i tmp_file_ile.args | awk '{print $2}'`
# done

# for i in mc-range eta-range 
# do 
#   echo $i `grep $i tmp_file_cip.args | awk '{print $2}'` 
# done 
#

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("rundir")
parser.add_argument("--ile-name",default="ILE.sub")
parser.add_argument("--cip-name",default="CIP*.sub")
parser.add_argument("--puff-name",default="PUFF.sub")
opts = parser.parse_args()


# Generate file : use shell script patterns
#  (would be better to parse argument strings, but oh well)
cmd1 = "grep arguments {}/{} |  tr '\"' ' ' | sed 's/--/\\n/g' > tmp_file_ile.args".format(opts.rundir,opts.ile_name)
cmd2 = "grep arguments {}/{} |  tr '\"' ' ' | sed 's/--/\\n/g' > tmp_file_cip.args".format(opts.rundir,opts.cip_name)
cmd3 = "grep arguments {}/{} |  tr '\"' ' ' | sed 's/--/\\n/g' > tmp_file_puff.args".format(opts.rundir,opts.puff_name)
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)

# convert arguments into dictionary
args_ile = {}
with open("tmp_file_ile.args",'r') as f:
    for line in f:
        key_list = line.split()
        key=key_list[0]
        key_list.pop(0)
        if not(key in args_ile):
            args_ile[key]=' '.join(key_list)
        else:
            args_ile[key]+= '\t'+' '.join(key_list)

args_cip={}
with open("tmp_file_cip.args",'r') as f:
    for line in f:
        key_list = line.split()
        key=key_list[0]
        key_list.pop(0)
        args_cip[key]=' '.join(key_list)

args_puff={}
with open("tmp_file_puff.args",'r') as f:
    for line in f:
        key_list = line.split()
        key=key_list[0]
        key_list.pop(0)
        args_puff[key]=' '.join(key_list)


# dump out arguments desired

keys_ile = ['fmin-template', 'channel-name', 'fmin-ifo' ,'d-max', 'approx', 'l-max', 'event-time','distance-marginalization-lookup-table','internal-sky-network-coordinates','n-eff','sampler-method', 'srate','fmax','reference-freq','no-adapt-distance','no-adapt-after-first']
keys_cip = ['mc-range', 'eta-range','fit-method','sampler-method','lnL-offset']  # latter two are very helpful for some diagnostics
keys_puff = ['enforce-duration-bound', 'puff-factor', 'force-away']
for name in keys_ile:
    if name in args_ile:
        print("ILE ", name, args_ile[name])

for name in keys_cip:
    if name in args_cip:
        print("CIP ",name, args_cip[name])

for name in keys_puff:
    if name in args_puff:
        print('PUFF ',name, args_puff[name])
