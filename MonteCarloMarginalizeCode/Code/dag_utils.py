# Copyright (C) 2013  Evan Ochsner
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
A collection of routines to manage Condor workflows (DAGs).
"""

import numpy as np
from time import time
from hashlib import md5

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>"

def generate_job_id():
    """
    Generate a unique md5 hash for use as a job ID.
    Borrowed and modified from the LAL code in glue/glue/pipeline.py
    """
    t = str( long( time() * 1000 ) )
    r = str( long( np.random.random() * 100000000000000000L ) )
    return md5(t + r).hexdigest()

def write_extrinsic_marginalization_dag(m1m2, extr_sub,
        fname='marginalize_extrinsic.dag'):
    """
    Write a dag to manage a set of parallel jobs to compute the likelihood
    marginalized over extrinsic parameters at a set of extrinsic points.

    Inputs:
        - 'm1m2' is an N x 2 array of values for mass1 and mass2 of a binary
          (in units of solar masses). Each of the N mass pairs will
          become a node in the DAG.
        - 'extr_sub' is a string giving the path to a condor submit file for
          launching a job to marginalize over extrinsic parameters.
        - 'fname' is the name of the DAG file to be output.

    N.B. This function does not return a value, but will write a DAG file.
    """
    dag = open(fname, 'w')
    Njobs = len(m1m2)
    for i in xrange(Njobs):
        job = generate_job_id()
        line = 'JOB ' + job + ' ' + extr_sub + '\n'
        dag.write(line)
        line = 'VARS ' + job + ' ' + 'm1=\"%.16g\" m2=\"%.16g\"\n' % (m1m2[i][0], m1m2[i][1])
        dag.write(line)
    dag.close()
    return fname

# FIXME: Keep in sync with arguments of integrate_likelihood_extrinsic
def write_integrate_likelihood_extrinsic_sub(tag='integrate',
        cache='local.cache',
        channels=None,
        psds=None,
        coinc=None,
        fref=None,
        seglen=None,
        tref=None,
        pad=None,
        time_marg=False
        ):
    """
    Write a submit file for launching jobs to marginalize the likelihood over
    extrinsic parameters.

    Inputs:
        - 'tag' is a string to specify the base name of output files. The output
          submit file will be named tag.sub, and the jobs will write their
          output to tag-ID.out, tag-ID.err, tag.log, where 'ID' is a unique
          identifier for each instance of a job run from the sub file.
        - 'cache' is the path to a cache file which gives the location of the
          data to be analyzed.
        - 'coinc' is the path to a coincident XML file, from which masses and
          times will be drawn FIXME: remove this once it's no longer needed.
        - 'channelH1/L1/V1' is the channel name to be read for each of the
          H1, L1 and V1 detectors.
        - 'psdH1/L1/V1' is the path to an XML file specifying the PSD of
          each of the H1, L1, V1 detectors.

    Outputs:
        - The name of the sub file that was generated.
    """
    # FIXME: Handle the case when we don't have data for some detector!
    assert len(psds) == len(channels)
    fname = tag + '.sub'
    sub = open(fname, 'w')
    sub.write('executable=integrate_likelihood_extrinsic\n')
    sub.write('universe=vanilla\n')
    line = 'arguments='
    if cache is not None:
        line += ' --cache-file %s' % cache
    if channels is not None:
        for chan in channels:
            line += ' --channel-name %s' % chan
    if psds is not None:
        for psd in psds:
            line += ' --psd-file %s' % psd
    if coinc is not None:
        line += ' --coinc-xml %s' % coinc
    if fref is not None:
        line += ' --reference-frequency %.16g' % fref
    if seglen is not None:
        line += ' --seglen %i' % seglen
    if tref is not None:
        line += ' --event-time %.16g' % tref
    if pad is not None:
        line += ' --padding %i' % pad
    if time_marg is True:
        line += ' --time-marginalization'
    line += ' --mass1 $(m1) --mass2 $(m2)\n'
    sub.write(line)
    sub.write('getenv=True\n')
    line = 'output=%s-$(cluster).out\n' % (tag)
    sub.write(line)
    line = 'error=%s-$(cluster).err\n' % (tag)
    sub.write(line)
    line = 'log=%s.log\n' % (tag)
    sub.write(line)
    sub.write('request_memory=2048\n')
    sub.write('notification=never\n')
    sub.write('queue\n')
    sub.close()
    return fname
