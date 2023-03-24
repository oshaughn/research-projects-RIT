#!/usr/bin/env python3

# copied on 5 May 2021, from https://git.ligo.org/pe/O3/o3a_catalog_events/-/blob/master/scripts/make_uni_comov_skymap.py


#import pesummary
#from pesummary.gw.file.read import read
import argparse
import logging
from logging import info, error, warn
from lalinference.nest2pos import draw_posterior
import h5py
import numpy as np
import astropy
from astropy.cosmology import LambdaCDM
Planck15_lal = LambdaCDM(H0=67.90, Om0=0.3065, Ode0=0.6935)

parser = argparse.ArgumentParser('Program to resample lalinference posteriors from euclidean to uniform-in-comoving-volume distance prior')
parser.add_argument('--runid',help='RunID to use from file. If not given, will apply to all runs',default=None)
parser.add_argument('-v','--verbose',default=False, action='store_true',help='Produce verbose output')
parser.add_argument('--sfr-power',help='Provide a value X such that the reweighting includes a star formation rate evolution as (1+z)^X (default=0, no evolution)', default=0, metavar='X')
parser.add_argument('input',help='Input file to process', metavar='input.h5')
parser.add_argument('output',help='PESummary file to output after resampling',metavar='output.h5', default=None)


def resample(indata, verbose=False, lcdm=Planck15_lal, sfr_power=0):
    """
    Resample the input data with the given weight
    indata: PESummary samples object
    """
    # Resampling factor is (dV_c/dz * dz/ddl * ddl/dV_E ) * (dt_src / dt_det)
    dl = np.array(indata['luminosity_distance'])
    Mpc = astropy.units.megaparsec
    sr = astropy.units.steradian
    D_H = (astropy.constants.c / lcdm.H0).to_value(unit=Mpc)
    try:
        z=np.array(indata['redshift'])
    except KeyError:
        z=np.array([astropy.cosmology.z_at_value(lcdm.luminosity_distance, this_dl*Mpc) for this_dl in dl])
    E = lcdm.efunc(z)
    weight = 1.0/((1+z)**(2.0-sfr_power)*(E*(dl/D_H) + (1+z)**2))
    return draw_posterior(indata, np.log(weight), verbose=verbose)


def main(inputfile, outputfile, runid=None, verbose=False, sfr_power=0):
    infile = h5py.File(inputfile, 'r')
    
    if runid and runid not in infile:
        raise KeyError(f"Input file {inputfile} does not contain table {runid}")

    if runid is not None:
        tabnames = [runid]
    else:
        tabnames = set(infile.keys())
    info(f"Will process {tabnames} from {inputfile}")


    # Create new output file object
    outfile = h5py.File(outputfile, 'w')

    # For each run to process, perform reweighting
    for t in tabnames:
        #Copy structure for configs etc
        infile.copy(infile[t],outfile)
        if not 'posterior_samples' in list(infile[t]):
            continue
        oldpos = infile[t]['posterior_samples']
        newpos = resample(oldpos, verbose=verbose, sfr_power=sfr_power)
        info(f"Drew {len(newpos)}/{len(oldpos)} samples for {t}")
    
        # Add resampled run to output file
        del outfile[t]['posterior_samples']
        outfile[t]['posterior_samples']=newpos

        # Apply resampling to priors too, if they exist
        try:
            oldprior = infile[t]['priors']['samples']
            newprior = resample(oldprior, verbose=verbose, sfr_power=sfr_power)
            outfile[t]['priors']['samples']=newprior
        except (KeyError, ValueError, AttributeError):
            info(f"No prior samples found for {t}")
        except NotImplementedError:
            warn(f"Warning: cannot reweight prior samples as they have no redshift")

    # Save and exit
    infile.close()
    outfile.close()

if __name__=='__main__':
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    main(args.input, args.output, runid=args.runid, sfr_power=args.sfr_power)

