#!/usr/bin/env python3

# copied on 5 May 2021, from https://git.ligo.org/pe/O3/o3a_catalog_events/-/blob/master/scripts/make_uni_comov_skymap.py

import argparse
import tempfile
import resample_uniform_comoving
import subprocess
import sys
import os
import logging
import pesummary
from logging import info, error, warn

parser = argparse.ArgumentParser('Resample to uniform-in-comoving-volume and generate skymap')
parser.add_argument('infile',metavar='posterior_samples.h5',help='Input file')
parser.add_argument('--runid',type=str,default=None,help='Run ID to use, e.g. C01:IMRPhenomPv2')
parser.add_argument('--resampled-file',default=None,help='File to save resampled posterior to, if requested')
parser.add_argument('-v','--verbose',default=False, action='store_true',help='Print more info')
parser.add_argument('--Nsamps',default=None,help='Number of samples to pass to ligo-skymap-from-samples (default use all)')
parser.add_argument('-j','--jobs',default=1,type=int,help='Number of threads to use for making map')
parser.add_argument('--fitsoutname',default=None,help='Store fits output file',metavar='skymap.fits')
parser.add_argument('-p','--percentile',default=95, type=float, help='Percent probability to use for computing credible intervals (default: 95)')
parser.add_argument('-o','--output',default = 'skymap_stats.txt', help='Skymap statistics output')
parser.add_argument('--plot',default=None,metavar = 'skymap.{png,pdf}', help='Save the plot to given filename')

def main(infile, resampled_file=None, verbose=False, Nsamps=None, runid=None,
        jobs=1, fitsoutname=None, percentile = 90, output=None, plot=None):
    """
    Do the process of creating resampled file and computing the skymap and stats
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    tmp_samples_file = tempfile.mkstemp(suffix='.h5')[1]
    
    # Store the skymap in the new pesummary files
    cleanup_fits=False

    # Create resampled posterior
    resample_uniform_comoving.main(infile, tmp_samples_file, runid=runid, verbose=verbose)

    if resampled_file:
        commandline = f'h5repack -l CHUNK=4096 -f SHUF -f GZIP=9 {tmp_samples_file} {resampled_file}'
        info(f'Writing compressed output to {resampled_file}')
        retcode = subprocess.call(commandline, shell=True)
        if retcode != 0:
            error(f'Could not write output to {resampled_file}')


    if runid is None:
        import h5py
        with h5py.File(tmp_samples_file,'r') as f:
            runs = set(list(f)) - {'history', 'version', 'injection_data'}
        info(f'Regenerating map for {runs}')
    else:
        runs = [runid]

    for runid in runs:
        info(f'processing {runid}')
        cleanup_fits = False
        fitsoutname = resampled_file.replace('.h5',f'_{runid}.fits')
        
        regenerate_map_single_run(tmp_samples_file, runid, Nsamps=Nsamps, jobs=jobs, verbose=verbose,
                                  fitsoutname=fitsoutname, percentile=percentile, output=output, plot=plot)

        # Clean up fits file if we don't want to keep it
        if cleanup_fits:
            info(f'Removing temporary file {fitsoutname}')
            os.remove(fitsoutname)
    

    # Clean up tmp file
    os.remove(tmp_samples_file)

    

def regenerate_map_single_run(resampled_file, runid, Nsamps=None, jobs=1, fitsoutname=None, percentile = 90, output=None, plot=None, verbose=False):

    # Run ligo-skymap-from-samples
    commandline = 'ligo-skymap-from-samples ' \
            + (f' --maxpts {Nsamps}' if Nsamps else '') \
            + f' --seed 41' \
            + f' --jobs {jobs}' \
            + f' --fitsoutname {fitsoutname}' \
            + (f' --path {runid}/posterior_samples' if runid else '') \
            + (f' --loglevel INFO' if verbose else '') \
            + f' {resampled_file}'
    info(commandline)
    retcode = subprocess.call(commandline, shell=True)
    
    
    if retcode !=0 :
        error('Could not create FITS file')
        sys.exit(1)

    # Run ligo-skymap-stats
    commandline = 'ligo-skymap-stats ' \
            + ' --cosmology' \
            + f' -p {percentile}' \
            + (f' -o {output}' if output else '') \
            + (f' --loglevel INFO' if verbose else '') \
            + f' {fitsoutname}'
    info(commandline)
    retcode = subprocess.call(commandline, shell=True)
    if retcode != 0:
        error('Could not compute stats')
        sys.exit(1)

    # Make the skymap plot
    if plot:
        commandline = 'ligo-skymap-plot ' \
                + f' -o {plot}' \
                + f' --contour {percentile}' \
                + (f' --loglevel INFO' if verbose else '') \
                + f' {fitsoutname}'
        info(commandline)
        retcode = subprocess.call(commandline, shell=True)
        if retcode != 0:
            error('Failed to generate plot')
            sys.exit(1)

    info('Storing skymap {fitsoutname} into pesummary file {resampled_file}')
    commandline = 'summarymodify --delimiter , --force_replace --overwrite' \
                + f' --store_skymap {runid},{fitsoutname}' \
                + f' --samples {resampled_file}'
    info(commandline)
    retcode = subprocess.call(commandline, shell=True)
    if retcode != 0:
        error('Failed to store map in pesummary file')
        sys.exit(1)

if __name__=='__main__':
    args = parser.parse_args()
    main(**vars(args))

