#! /usr/bin/env python
#import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import argparse
import os

'''
import RIFT.lalsimutils as lalsimutils
from RIFT.physics import EOSManager as EOSManager

from scipy.integrate import nquad
#import EOS_param as ep
import os
from EOSPlotUtilities import render_eos
import lal
'''

parser = argparse.ArgumentParser()
parser.add_argument("--fname",type=str,help="Dummy argument required by API")
parser.add_argument('--using-eos', type=str, help="Send eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
parser.add_argument('--using-eos-index',type=int, help="Line number for single calculation.  Single-line calculation only")
parser.add_argument('--eos_start_index',type=int, help="Line number from where to start the likelihood calculation.")
parser.add_argument('--eos_end_index',type=int, help="Line number which needs likelihood for which needs to be evaluated.")
parser.add_argument('--plot', action='store_true', help="Enable to plot resultant M-R and Gaussians.")
parser.add_argument('--outdir', type=str, help="Output eos file directory.")
parser.add_argument('--outdir-clean', type=str, help="Delete CleaOutile direc before starting the runtory.")
parser.add_argument('--fname-output-integral', type=str, help="Output eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
parser.add_argument('--fname-output-samples', type=str, help="NEVER USED, but is enabled. Output eos file with [lnL, sigma_lnL, lambda1, lambda2, lambda3, lambda4] as the parameters.")
parser.add_argument("--conforming-output-name",action='store_true')

opts = parser.parse_args()
if not(opts.using_eos_index is None):
    opts.eos_start_index = opts.using_eos_index
    opts.eos_end_index = opts.using_eos_index+1

dat_orig_names = None
fname_eos = opts.using_eos
fname_eos = fname_eos.replace("file:",'')
with open(fname_eos,'r') as f:
    header_str = f.readline()
    header_str = header_str.rstrip()
    dat_orig_names = header_str.replace('#','').split()[2:]
print("Original field names ", dat_orig_names)


if opts.outdir_clean:
    import shutil
    try: shutil.rmtree(opts.outdir)
    except: pass
    del shutil
elif opts.outdir is None:
    opts.outdir = "."

from pathlib import Path
Path(opts.outdir).mkdir(parents=True, exist_ok=True)
del Path

####################### Gaussian #######################

def likelihood_evaluation():
    from scipy.stats import multivariate_normal
    rv = multivariate_normal([3,0,0], [[2,0,0], [0,2,0], [0,0,2]])
    #rv2 = multivariate_normal([5,0,0], [[2,0,0], [0,2,0], [0,0,2]])
    
    if opts.using_eos[:5] =='file:': eoss = np.genfromtxt(opts.using_eos[5:],dtype='str')
    else: eoss = np.genfromtxt(opts.using_eos[:],dtype='str')
    
    likelihood_dict = {}
    
    for i in np.arange(opts.eos_start_index, opts.eos_end_index):
        #likelihood_dict[i] = rv.pdf([eoss[i,2:]])+rv2.pdf([eoss[i,2:]])
        likelihood_dict[i] = rv.pdf([eoss[i,2:]])
        
        eoss[i,0] = np.log(likelihood_dict[i])
        eoss[i,1] = 0.001  # nominal integration error
    
    postfix = ''
    if opts.conforming_output_name:
        postfix = '+annotation.dat'
    
    # opts.fname is not None only when using RIFT as is in RIT-matters/20230623
    if opts.fname is None: np.savetxt(opts.outdir+"/"+opts.fname_output_integral+postfix, eoss[opts.eos_start_index: opts.eos_end_index], fmt = '%10s', header="lnL     sigma_lnL   " + ' '.join(dat_orig_names))
    else: np.savetxt(opts.fname_output_integral+postfix, eoss[opts.eos_start_index: opts.eos_end_index], fmt = '%10s', header="lnL     sigma_lnL   " + ' '.join(dat_orig_names))
    
    
    # Plotting not implemented for Gaussian distribution.
    if opts.plot:
        import matplotlib
        import matplotlib.pyplot as plt
        
        
        matplotlib.rcParams.update({'font.size': 12.0,  'mathtext.fontset': 'stix'})
        matplotlib.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)
        matplotlib.rcParams['xtick.labelsize'] = 15.0
        matplotlib.rcParams['ytick.labelsize'] = 15.0
        matplotlib.rcParams['axes.labelsize'] = 25.0
        matplotlib.rcParams['lines.linewidth'] = 2.0
        plt.style.use('seaborn-v0_8-whitegrid')
        #fig,(ax1,ax2) = plt.subplots(2,1)
        
        
        fig = plt.figure(); ax = fig.add_subplot(111)
        from plotting import plot_data_and_gaussian
        if 'j0030' in observations : plot_data_and_gaussian(mn, cov, rv, data_j0030, ax)
        if 'j1731' in observations : plot_data_and_gaussian(mn2, cov2, rv2, data_j1731, ax)
        
        likelihood_array = np.empty((0))
        for i in likelihood_dict: likelihood_array = np.append(likelihood_array, likelihood_dict[i])
        
        lmax = max(likelihood_array)
        lmin = min(likelihood_array)
        
        
        for i in likelihood_dict:
            MRcolor = plt.cm.gist_rainbow((likelihood_dict[i]-lmin)/(lmax-lmin))
            ratio = (likelihood_dict[i]-lmin)/(lmax-lmin)
            ax.plot(R_dict[i],M_dict[i], alpha = ratio, color = 'b')
        
        ax.set_xlim(7,20)
        ax.set_xlabel('Radius [km]')
        ax.set_ylabel('Mass [M/M$_\odot$]')
        
        plt.savefig('MR_likelihood_sample_mMax.pdf',format = 'pdf')
        plt.savefig('MR_likelihood_sample_mMax.png',format = 'png')
        
        plt.show()


if __name__ == '__main__':
    likelihood_evaluation()







