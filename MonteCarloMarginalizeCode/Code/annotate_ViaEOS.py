#! /usr/bin/env python
#
# GOAL
#   - process file of eos_names.txt
#   - return array of same names, annotated with EOS
#       Notably, provides R_fiducial
#       Designed to work on *any* file (e.g., output of CIP *.integral)
#   - can also work on *any* file which has eos_name as a column.  
#
# COMPARE TO
#    O2_pe_rapidpe_results/Python


import argparse
import sys
import numpy as np
import lal
import lalsimulation as lalsim

import EOSManager

def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = np.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == np.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == np.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> np.all(sa['id'] == sb['id'])
    True
    >>> np.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b

parser = argparse.ArgumentParser()
parser.add_argument("--fname-to-annotate",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--fname-with-annotation",default='annotated_samples.dat',help="Use input format with tidal fields included.")
parser.add_argument("--eos-format",default='name',help="Use a specific EOS format")
parser.add_argument("--source-redshift",default=0,type=float,help="Used in cases where m1,m2 provided. Source redshift assumed ")
parser.add_argument("--annotate-lambda",action='store_true',help="Add lambda1(m1) lambda2(m2).  Requires m1,m2 defined. ")
parser.add_argument("--annotate-m-max",action='store_true',help="Add m_max_msun. ")
parser.add_argument("--fiducial-mass",default=1.4,help="Add R_fiducial for this radius ")
parser.add_argument("--no-use-lal-eos",action='store_true',help="Do not use LAL EOS interface. Used for spectral EOS. Do not use this.")
opts=  parser.parse_args()

def generate_eos_from_samples(samples,indx,format='name'):
    if format == 'name':
        eos_name = samples['eos_name'][indx].replace('lal_','') # convert
        my_eos = EOSManager.EOSLALSimulation(name=eos_name)
        return my_eos
    # No other call path implemented yet
    # see append_fields_via_eos.py
    return None


# Read first line, identify eos_name column
line_header=''
with open(opts.fname_to_annotate, 'r') as f:
    line_header = f.readline()
param_names_orig = line_header.replace('#','').split()
dtype_list =[]
for name in param_names_orig:
    if 'eos_name' == name:
        dtype_list.append( (name, 'S32'))
    else:
        dtype_list.append( (name, "<f8"))
# Load data, making sure to read the eos_name as a string
samples = np.genfromtxt(opts.fname_to_annotate,dtype=dtype_list) # 
param_names = samples.dtype.names
param_names_out = list(param_names) # duplicate
print(param_names, samples.dtype)
npts = len(samples[param_names[0]])

if not 'eos_name' in param_names:
    print(" no EOS name, why are you running this?")
    sys.exit(0)

# Add fields as appropriate
samples =add_field(samples, [('R_fiducial_km',float)])
if opts.annotate_m_max:
    samples =add_field(samples, [('m_max_msun',float)])
if opts.annotate_lambda and 'm1' in param_names:
    samples =add_field(samples, [('lambda1',float)])
    samples =add_field(samples, [('lambda2',float)])




for indx in np.arange(npts):
    # Do not populate BBH entries
    if samples['eos_name'][indx]=='BBH':
        continue
    # Generate EOS
    eos_now = generate_eos_from_samples(samples, indx,format=opts.eos_format)
    # Fill fiducial mass
    samples['R_fiducial_km'][indx] = lalsim.SimNeutronStarRadius(opts.fiducial_mass*lal.MSUN_SI,eos_now.eos_fam)/1e3
    # Fill maximum mass
    if opts.annotate_m_max:
        samples['m_max_msun'][indx] = eos_now.mMaxMsun
    # Fill lambda(m) values
    if opts.annotate_lambda and 'm1' in param_names:
        True
    

# write format specifier : 
fmt = ""
for param_name in samples.dtype.names:
    if param_name == 'eos_name':
        fmt += " %s "
    else:
        fmt += " %.18e "

# Vastly superior data i/o
import pandas
dframe = pandas.DataFrame(samples)
dframe.to_csv(opts.fname_with_annotation,sep=' ',header=' # '+' '.join(samples.dtype.names),index=False)
sys.exit(0)


# print fmt
# print samples
# samples.tofile(opts.fname_with_annotation+".test", sep= '\n',format=fmt)
# dat = np.empty( (npts,len(samples.dtype.names)), samples.dtype )
# for indx in np.arange( len(samples.dtype.names)):
#     dat_here = dat[:][indx]  # make a pointer or copy. This will help with typing
# #    print samples.dtype.names[indx], dat.dtype[indx], samples[samples.dtype.names[indx]]
#     dat_here = samples[ samples.dtype.names[indx]]

# print dat
# sys.exit(0)
# np.savetxt(opts.fname_with_annotation,dat,fmt=fmt)
