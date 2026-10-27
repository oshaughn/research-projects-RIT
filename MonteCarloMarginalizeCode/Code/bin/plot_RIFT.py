#!/usr/bin/env python
"""This code is meant to check the health of a RIFT run as it progresses and after it has finished."""
# TO DO: plot_cip_max_lnL needs to be robust.
# JSD convergence for subdags
# Two all.nets when subdags being used
# Usage: In the rundirectory, run plot_RIFT.py --precessing or plot_RIFT.py --rundir-path `pwd`/rundir --save-path `pwd` --precessing 
###########################################################################################
# Import
###########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from collections import namedtuple
import RIFT.lalsimutils as lsu
from argparse import ArgumentParser
import corner
import re
import json
import html
import time
import base64
import io
from scipy.stats import chi2

# Matplotlib configuration
plt.rcParams.update({
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'font.size': 22,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 100
})
plt.style.use('seaborn-v0_8-poster')

__author__ = "A. Jan"

# Avoid printing float type
#np.set_printoptions(legacy='1.25')

# Default colors
default_colors=['black', "#FF0000", "#FF7F00", "#FFFF00", "#7FFF00", "#00FF00", "#00FFFF", "#007FFF", "#0000FF", "#4B0082", "#8B00FF"]

###########################################################################################
# Path and Configuration Setup
###########################################################################################
parser = ArgumentParser()
parser.add_argument("--rundir-path", default = os.getcwd(), type = str, help =  "path to run directory. Default is current directory.")
parser.add_argument("--save-path", default = os.getcwd(), type = str, help =  "Where to create the plots folder. Default is current directory.")
parser.add_argument("--LISA", action = "store_true", help = "Use this argument if analyzing a LISA run")
parser.add_argument("--eccentric", action = "store_true", help = "Use this argument if the run has eccentricity and meanPerAno")
parser.add_argument("--precessing", action = "store_true", help = "Use this argument if the run is precessing")
parser.add_argument("--non-spinning", action = "store_true", help = "Use this argument if the run is non-spinning")
parser.add_argument("--html-max-width", default = 1600, type = int, help = "Images are embedded in Diagnostics.html and downscaled to this width (pixels) if wider, to keep the file small. Use 0 to embed the images at full resolution. Default is 1600.")
parser.add_argument("--html-show-paths", action = "store_true", help = "Keep absolute paths in the ILE, CIP and Visual text of Diagnostics.html. By default they are shortened so the html file can be shared. The Run summary always shows the full paths.")
parser.add_argument("--truth-file", default = None, type = str,  help = "Path to the truth file. If not provided, the code will search for a truth file in the parent directory or the frames directory. The frames directory is expected to be located in the parent directory.")
opts = parser.parse_args()

path = opts.rundir_path
save_path = opts.save_path
LISA = opts.LISA
eccentricity = opts.eccentric
precessing = opts.precessing
non_spinning = opts.non_spinning

# Print kind of analysis
messages = []
if eccentricity:
    messages.append("Eccentric analysis")
if precessing:
    messages.append("Precessing analysis")
if LISA:
    messages.append("LISA analysis")
if non_spinning:
    messages.append("Non spinning analysis")
if messages:
    print("\n" + "\n".join(messages) + "\n")

# Locate corner plot executable
corner_plot_exe = os.popen("which plot_posterior_corner.py").read()[:-1]

# Define file paths
all_net_path = os.path.join(path, "all.net")
corner_all_net_path = all_net_path # all.net used in the corner plots, updated in the generate plots section if there is a subdag

# Determine if truth file should be used
if opts.truth_file is not None:
    truth_file_path = opts.truth_file
else:
    candidate_paths = [
        os.path.join(path, "../mdc.xml.gz"),
        os.path.join(path, "../frames/mdc.xml.gz"),
    ]

    truth_file_path = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            truth_file_path = candidate
            break

use_truths = truth_file_path is not None

if use_truths:
    print(f"Using {truth_file_path} for truth values in corner plots!")


# Initialize diagnostics dictionary
run_diagnostics = {
    "JSD": {},
    "JSD_3": {},
}
JSD_threshold = 0.005

###########################################################################################
# Functions
###########################################################################################
def get_lnL_cut_points(all_net_path, lnL_cut=15, error_threshold=0.4, composite=False, provide_max_lnL_point=False):
    """
    Computes high-likelihood statistics from an all.net or composite file.

    Args:
        all_net_path (str): Path to the all.net or composite file.
        lnL_cut (float): Threshold used to define high-likelihood points. Default is 15.
        error_threshold (float): Maximum allowed Monte Carlo error. Default is 0.4
        composite (bool): Whether the input file is a composite file.
        provide_max_lnL_point (bool): If True, return the maximum-likelihood
            sample and its lnL value.

    Returns:
        tuple:
            If provide_max_lnL_point is True:
                - Maximum-likelihood sample.
                - Maximum lnL value.

            If composite is True:
                - Global maximum lnL.
                - Number of high-likelihood points in the composite.
                - Maximum lnL in the composite file.
                - Total number of points.

            Otherwise:
                - Maximum lnL value.
                - Number of high-likelihood points.
    """
    # Load data from all.net file
    data = np.loadtxt(all_net_path, ndmin=2)
    
    # Extract lnL and error columns
    samples = data[:, :9]
    lnL = data[:, 9]
    error = data[:, 10]
    
    # Adjust columns if LISA is True
    if LISA and not(eccentricity):
        lnL = data[:, 11]
        error = data[:, 12]
        samples = data[:,:11]
    if LISA and eccentricity:
        lnL = data[:,13]
        error = data[:,14]
        samples = data[:,:13]
    if not(LISA) and eccentricity:
        lnL = data[:, 11]
        error = data[:, 12]
        samples = data[:,:11]
    
    # Remove NaN values from lnL (and the same rows from error and samples, so they stay aligned with lnL)
    total_points = len(lnL)
    not_nan = ~np.isnan(lnL)
    lnL = lnL[not_nan]
    error = error[not_nan]
    samples = samples[not_nan]

    # Every evaluation in this composite failed
    if composite and len(lnL) == 0:
        return run_diagnostics["max_lnL"], 0, np.nan, total_points

    # Find high likelihood points based on lnL_cut
    max_lnL = np.max(lnL)
    man_lnL_index = np.argmax(lnL)
    if provide_max_lnL_point:
        return samples[man_lnL_index], max_lnL
    if composite:
        max_lnL_composite = max_lnL
        max_lnL = run_diagnostics["max_lnL"]
    high_lnL_indices = np.argwhere(lnL >= (max_lnL - lnL_cut)).flatten()

    # Thorough split of number of high lnL points
    high_lnL_indices_lnLcut_12 = np.argwhere(lnL >= (max_lnL - 12)).flatten()
    high_lnL_indices_lnLcut_10 = np.argwhere(lnL >= (max_lnL - 10)).flatten()
    high_lnL_indices_lnLcut_5 = np.argwhere(lnL >= (max_lnL - 5)).flatten()
    high_lnL_indices_lnLcut_2 = np.argwhere(lnL >= (max_lnL - 2)).flatten()
    
    high_lnL_points = len(high_lnL_indices)
    lnL = lnL[high_lnL_indices]
    error = error[high_lnL_indices]

    # Filter high lnL points with low Monte Carlo error
    low_error_indices = np.argwhere(error <= error_threshold).flatten()
    lnL = lnL[low_error_indices]
    error = error[low_error_indices]

    # No high likelihood points with low error in this composite (can happen in early iterations)
    if composite and len(lnL) == 0:
        return run_diagnostics["max_lnL"], 0, np.nan, total_points
    
    # Update diagnostics with results
    max_lnL = np.max(lnL)
    no_points = len(lnL[lnL >= (max_lnL - lnL_cut)])
    if composite:
        max_lnL_composite = max_lnL
        max_lnL = run_diagnostics["max_lnL"]
        return np.round(max_lnL, 2), no_points, np.round(max_lnL_composite, 2), total_points
    if not(composite):
        run_diagnostics.update({
            "total_lnL_evaluations":total_points,
            "max_lnL": np.round(max_lnL, 2),
            "high_lnL_points": no_points,
            "high_lnL_points_with_large_error": high_lnL_points - no_points,
            "total_high_lnL_points": high_lnL_points,
            "high_lnL_points_with_lnLcut_12": len(high_lnL_indices_lnLcut_12),
            "high_lnL_points_with_lnLcut_10": len(high_lnL_indices_lnLcut_10),
            "high_lnL_points_with_lnLcut_5": len(high_lnL_indices_lnLcut_5),
            "high_lnL_points_with_lnLcut_2": len(high_lnL_indices_lnLcut_2),
        })

        return np.round(max_lnL, 2), no_points


def lnL_at_credible_interval(ci=0.99, run_diagnostics=run_diagnostics, return_ndim=False):
    """
    Computes the log-likelihood corresponding to the boundary of a
    credible interval, assuming a multivariate Gaussian posterior.

    Args:
        ci (float): Credible interval enclosed by the likelihood contour.
            Default is 0.99.
        run_diagnostics (dict): Dictionary containing run diagnostics.
        return_ndim (bool): If True, return the dimensionality of the
            inferred parameter space instead of the log-likelihood.

    Returns:
        float:
            If return_ndim is True:
                - Number of dimensions used in the calculation.

            Otherwise:
                - Log-likelihood at the credible interval boundary.
"""
    if not (0 < ci < 1):
        raise ValueError("ci must be between 0 and 1, e.g. 0.99")

    ndim = 4
    if opts.non_spinning:
        ndim = 2
    else:
        if opts.precessing:
            ndim += 4
        if opts.eccentric:
            ndim += 2
    if return_ndim:
        return ndim
    max_lnL = run_diagnostics["max_lnL"]

    r2 = chi2.ppf(ci, df=ndim)
    delta_lnL = -0.5 * r2
    lnL = max_lnL + delta_lnL

    return lnL

def create_plots_folder(save_path):
    """
    Creates the plots directory structure if it does not already exist.

    Args:
        save_path (str): Path where the plots directory will be created.

    Returns:
        None
    """
    if not(os.path.exists(save_path + "/plots")):
        print(f"--> plots folder does not exist. Creating one in {save_path}")
        os.mkdir(save_path + "/plots")
        os.mkdir(save_path + "/plots/histograms")
        os.mkdir(save_path + "/plots/corner_plots")
        os.mkdir(save_path + "/plots/1_D_plots")
    else:
        print(f"--> plots folder exists, saving plots in directory {save_path}/plots")

def get_chirpmass_massratio_eta_totalmass_from_componentmasses(m1, m2):
    """
    Computes chirp mass, mass ratio, symmetric mass ratio, and total mass
    from component masses.

    Args:
        m1 (array): Primary masses.
        m2 (array): Secondary masses.

    Returns:
        tuple:
            - Chirp masses.
            - Mass ratios (q = m2 / m1).
            - Symmetric mass ratios.
            - Total masses.
    """
    return np.array((m1*m2)**(3/5) / (m1+m2)**(1/5)).reshape(-1,1), np.array(m2/m1).reshape(-1,1), np.array((m1*m2) / (m1+m2)**(2)).reshape(-1,1), np.array(m1+m2).reshape(-1,1)

def get_index_for_parameter(parameter, extrinsic = False):
    """
    Retrieves the index corresponding to a given parameter name.

    Args:
        parameter (str): The name of the parameter.

    Returns:
        int or None: The index of the parameter if found, otherwise None.
    """
    rift_parameters = ["m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z", "mc", "eta", "indx",
 "Npts", "ra", "dec", "tref", "phiorb", "incl", "psi", "dist", "p", "ps",
 "lnL", "mtot", "q", "eccentricity", "meanPerAno"]
    if extrinsic:
        rift_parameters = ["m1", "m2", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z", "mc", "eta", "ra", "dec", "time", "phiorb", "incl", "psi", "distance", "Npts", "lnL", "p", "ps", "neff",
    "mtot", "q", "chi_eff", "chi_p", "m1_source", "m2_source", "mc_source", "mtotal_source", "redshift", "eccentricity", "meanPerAno"]
    return rift_parameters.index(parameter) if parameter in rift_parameters else None

def get_sample_statistics(samples):
    """
    Computes the 5th, 50th, and 95th percentiles of a sample set.

    Args:
        samples (array-like): Input samples.

    Returns:
        numpy.ndarray:
            - 5th percentile.
            - 50th percentile (median).
            - 95th percentile.
    """
    return np.percentile(samples, [5,50,95])

def get_combination_from_mass_and_spin(posterior, parameter):
    """
    Computes spin combinations and derived spin parameters from posterior
    samples.

    Args:
        posterior (numpy.ndarray): Posterior samples.
        parameter (str): Parameter to compute. Supported values are
            'chi1', 'chi2', 'chi_eff', 'chiMinus', 'chi_p',
            'chi1_perp', and 'chi2_perp'.

    Returns:
        numpy.ndarray:
            Requested spin combination evaluated for all samples.
    """

    parameter_m1 = get_index_for_parameter("m1")
    parameter_m2 = get_index_for_parameter("m2")

    parameter_s1x = get_index_for_parameter("s1x")
    parameter_s1y = get_index_for_parameter("s1y")
    parameter_s1z = get_index_for_parameter("s1z")

    parameter_s2x = get_index_for_parameter("s2x")
    parameter_s2y = get_index_for_parameter("s2y")
    parameter_s2z = get_index_for_parameter("s2z")

    m1 = posterior[:, parameter_m1]
    m2 = posterior[:, parameter_m2]

    s1x = posterior[:, parameter_s1x]
    s1y = posterior[:, parameter_s1y]
    s1z = posterior[:, parameter_s1z]

    s2x = posterior[:, parameter_s2x]
    s2y = posterior[:, parameter_s2y]
    s2z = posterior[:, parameter_s2z]

    if parameter == "chi1":
        return np.sqrt(s1x**2 + s1y**2 + s1z**2)

    elif parameter == "chi2":
        return np.sqrt(s2x**2 + s2y**2 + s2z**2)

    elif parameter == "chi_eff":
        return (m1 * s1z + m2 * s2z) / (m1 + m2)

    elif parameter == "chiMinus":
        return (m1 * s1z - m2 * s2z) / (m1 + m2)

    elif parameter == "chi_p":
        q = m2 / m1  # assumes m1 >= m2

        chi1_perp = np.sqrt(s1x**2 + s1y**2)
        chi2_perp = np.sqrt(s2x**2 + s2y**2)

        return np.maximum(
            chi1_perp,
            ((4 * q + 3) / (4 + 3 * q)) * q * chi2_perp
        )

    elif parameter == "chi1_perp":
        return np.sqrt(s1x**2 + s1y**2)

    elif parameter == "chi2_perp":
        return np.sqrt(s2x**2 + s2y**2)


    else:
        raise ValueError(
            f"Unknown parameter '{parameter}'. "
            "Choose from: chi1, chi2, chi_eff, chiMinus, chi_p"
        )
def convert_all_net_to_posterior_format(all_net_path):
    all_net_data = np.loadtxt(all_net_path)
    chirpmass, massratio, eta, totalmass = get_chirpmass_massratio_eta_totalmass_from_componentmasses(all_net_data[:,1], all_net_data[:,2])
   # m1 m2 a1x a1y a1z a2x a2y a2z mc eta indx  Npts ra dec tref phiorb incl psi  dist p ps lnL mtotal q 
    zeros_for_extrinsic = np.zeros((len(all_net_data), 1)) 
    lnL = np.array(all_net_data[:,9]).reshape(-1,1)
    posterior_format_all_net = np.hstack([all_net_data[:,1:9], chirpmass, eta, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, zeros_for_extrinsic, lnL, totalmass, massratio])
    return posterior_format_all_net

def count_evaluations(file_path):
    """
    Counts the rows (marginalized lnL evaluations) in an all.net or composite file without loading it.

    Args:
        file_path (str): Path to the all.net or composite file.

    Returns:
        int: Number of rows.
    """
    with open(file_path, "r") as f:
        return sum(1 for line in f if line.strip() and not line.startswith("#"))

def choose_all_net_for_corner_plots(main_all_net_path, subdag_folder_path):
    """
    Chooses the all.net used in all the corner plots (the coloured high lnL points in the posterior corner plots and the exploration corner plot).
    Without a subdag this is the main all.net. With a subdag, the all.net with more evaluations (rows) is used, as it covers more of the explored
    parameter space. The main all.net is kept if the subdag all.net is missing, empty, or has the same number of evaluations.

    Args:
        main_all_net_path (str): Path to the main all.net.
        subdag_folder_path (str or None): Path to the subdag folder (output of find_posteriors_in_sub), None if there is no subdag.

    Returns:
        str: Path to the all.net to use in the corner plots.
    """
    if subdag_folder_path is None:
        return main_all_net_path
    subdag_all_net_path = f"{subdag_folder_path}/all.net"
    try:
        if not os.path.exists(subdag_all_net_path) or os.path.getsize(subdag_all_net_path) == 0:
            print(f"\n--> Subdag all.net not found or empty ({subdag_all_net_path}), using the main all.net in the corner plots.")
            return main_all_net_path
        main_evaluations = count_evaluations(main_all_net_path)
        subdag_evaluations = count_evaluations(subdag_all_net_path)
    except Exception as e:
        print(e)
        print("Couldn't compare the main and subdag all.net, using the main all.net in the corner plots.")
        return main_all_net_path
    use_subdag = subdag_evaluations > main_evaluations
    run_diagnostics["corner_plots_all_net"] = {"used": "Subdag" if use_subdag else "Main", "main_evaluations": main_evaluations, "subdag_evaluations": subdag_evaluations}
    print(f"\n--> all.net for the corner plots: main has {main_evaluations} evaluations, subdag has {subdag_evaluations}. Using the {'subdag' if use_subdag else 'main'} all.net.")
    return subdag_all_net_path if use_subdag else main_all_net_path

def find_posteriors_in_main(path_to_main_folder, limit_iterations=None):
    """
    Finds and sorts posterior sample files in the main folder.

    Args:
        path_to_main_folder (str): Path to the main folder containing
            posterior sample files.
        limit_iterations (int, optional): Number of posterior files to
            return.

    Returns:
        tuple:
            - Sorted posterior file paths.
            - Corresponding iteration indices.
    """
    posteriors_in_main = glob.glob(path_to_main_folder + "/posterior_samples*")
    posteriors_in_main.sort(key = os.path.getctime) # sort them according to creation time
    if limit_iterations:
        index = np.linspace(0, len(posteriors_in_main)-1, limit_iterations)
        index = np.array(index, dtype=int)
        return np.array(posteriors_in_main, dtype = str)[index], index + 1
    return posteriors_in_main, np.arange(len(posteriors_in_main)) + 1

def find_posteriors_in_sub(path_to_main_folder, limit_iterations = None, return_subdag_folder_path=False):
    """
    Finds posterior sample files in the subdag directory.

    Args:
        path_to_main_folder (str): Path to the main folder containing
            iteration_*_cip directories.
        limit_iterations (int, optional): Number of posterior files to
            return.
        return_subdag_folder_path (bool): If True, return the path to the
            subdag directory containing posterior samples.

    Returns:
        If return_subdag_folder_path is True:
            str or None:
                Path to the subdag directory, or None if no posterior
                samples are found.

        Otherwise:
            tuple:
                - Posterior file paths.
                - Iteration indices.
    """
    if return_subdag_folder_path:
        subdag_folder_path = None
        subdag_dirs = glob.glob(path_to_main_folder + "/iteration_*_cip")
        for d in subdag_dirs:
            posteriors = glob.glob(f"{d}/posterior_samples*.dat")
            if len(posteriors) > 0:
                subdag_folder_path = d
                break 
        return subdag_folder_path
    posteriors_in_subdag, iterations = find_posteriors_in_main(path_to_main_folder + "/iteration*cip*")
    

    if limit_iterations:
        index = np.linspace(0, len(posteriors_in_subdag)-1, limit_iterations)
        index = np.array(index, dtype=int)
        return np.array(posteriors_in_subdag, dtype = str)[index], index + 1
    else:
        return posteriors_in_subdag, np.arange(len(posteriors_in_subdag)) + 1

def calculate_JS_divergence(data1, data2):
    """
    Calculates the Jensen-Shannon Divergence between two datasets.

    Args:
        data1 (array-like): First dataset.
        data2 (array-like): Second dataset.

    Returns:
        summary (namedtuple): Summary containing median, lower, and upper quantiles of the divergence.
    """
    def calculate_js(data1, data2, ntests=10, xsteps=100):
        js_array = np.zeros(ntests)
        for j in range(ntests):
            nsamples = min([len(data1), len(data2)])
            A = np.random.choice(data1, size=nsamples, replace=False)
            B = np.random.choice(data2, size=nsamples, replace=False)
            xmin = np.min([np.min(A), np.min(B)])
            xmax = np.max([np.max(A), np.max(B)])
            x = np.linspace(xmin, xmax, xsteps)
            A_pdf = gaussian_kde(A)(x)
            B_pdf = gaussian_kde(B)(x)
            js_array[j] = np.nan_to_num(np.power(jensenshannon(A_pdf, B_pdf,  base = 2), 2))
        return calc_median_error(js_array)

    def calc_median_error(jsvalues, quantiles=(0.16, 0.84)):
        quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
        quants = np.percentile(jsvalues, quants_to_compute * 100)
        summary = namedtuple("summary", ["median", "lower", "upper"])
        summary.median = quants[1]
        summary.plus = quants[2] - summary.median
        summary.minus = summary.median - quants[0]
        return summary

    return calculate_js(data1, data2)

def plot_neff_data(path_to_main_folder, plot_title):
    """
    Plot effective number of samples (neff) data from CIP iterations.
    Args:
        path_to_main_folder (str): Path to the folder containing CIP iteration subfolders.
        plot_title (str): suffix for the plots and filenames.
    """
    print(f"\n--> Plotting n-eff for CIP for {plot_title} iterations.")
    # find CIP folders
    cip_iteration_folders = glob.glob(path_to_main_folder + "/iteration*cip*")
    
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("neff")
    # iteration numbers come from the folder names, a folder can be missing in the middle so counting them would drop the last iteration
    folder_iterations = [int(re.search(r"iteration_(\d+)_cip", os.path.basename(folder)).group(1)) for folder in cip_iteration_folders if re.search(r"iteration_(\d+)_cip", os.path.basename(folder))]
    iterations = np.arange(max(folder_iterations)+1) if folder_iterations else np.arange(0)
    
    run_diagnostics[plot_title] = {}
    run_diagnostics[plot_title]["CIP_neff"] = {}

    # read requested neff from CIP sub files
    CIP_file_names = ["CIP_worker.sub", "CIP_worker0.sub", "CIP_worker1.sub", "CIP_worker2.sub", "CIP_worker3.sub"]
    for i, name in enumerate(CIP_file_names):
        filename = f"{path_to_main_folder}/{name}"
        try:
            with open(filename, "r") as f:
                content = f.read()
                matches = re.findall(r"--n-eff\s+([+-]?\d+(?:\.\d+)?)", content)
                if matches:
                    neff_value = float(matches[-1])
                    worker_label = name.replace(".sub", "")
                    ax.axhline(y=neff_value, linestyle="--", color=default_colors[i], alpha=1.0, linewidth=1.0, label=f"{worker_label} neff")
                    run_diagnostics[plot_title]["CIP_neff"][worker_label] = np.round(neff_value, 2)
                else:
                    continue
        except Exception as e:
            pass
    ax.legend(loc="upper left")
    
    # read neff achived for each iteration from each instance of CIP
    run_diagnostics[plot_title]["CIP_neff_achieved"] = {}
    for n in iterations:
        i = path_to_main_folder + f"/iteration_{n}_cip"
        # the neff data is stored in the plots folder, not in the run directory (which might not be writable), and removed once it is read
        neff_file = f"{save_path}/plots/neff_{plot_title}_iteration_{n}.txt"
        # remove existing data file because I append
        os.system(f"rm {neff_file} 2> /dev/null")
        # read neff data from each file and store it in the neff file
        cmd=f"for i in {i}/overlap-grid-*-*ESS* ; do cat $i | tail -n 1 >> {neff_file}; done 2> /dev/null"
        os.system(cmd) 
        # calculate neff statistics
        try:
            if os.path.getsize(neff_file) == 0: 
                continue
            tmp_ESS_data=np.loadtxt(neff_file, usecols=[2])
            low, avg, high = np.percentile(tmp_ESS_data, [2.5,50,97.5]) # 2 std
            low_1_std, avg, high_1_std = np.percentile(tmp_ESS_data, [16,50,84]) # 1 std
            mini, maxi = np.min(tmp_ESS_data), np.max(tmp_ESS_data)
            ax.plot(iterations[n], mini, marker="x", color="black")
            ax.plot(iterations[n], maxi, marker="x", color="black")
            print(f"n-eff summary | iteration {iterations[n]:>2} | median = {avg:8.2f} | 95% interval = [{low:8.2f}, {high:8.2f}]")
            ax.errorbar(iterations[n], avg, yerr=np.array([avg-low,high-avg]).reshape(-1, 1), color = "royalblue", ecolor = "red", fmt ='o')
            ax.errorbar(iterations[n], avg, yerr=np.array([avg-low_1_std,high_1_std-avg]).reshape(-1, 1), color = "royalblue", ecolor = "green", fmt ='.')
            run_diagnostics[plot_title]["CIP_neff_achieved"][f"iteration_{n}_neff"] = np.round(avg, 2)
            iteration_prog = n
        except Exception as e:
            #print(f"Couldn't plot neff for iteration = {iterations[n]}")
            #break
            continue  # try plotting all possible cip's incase there is one folder where subdag exists
        finally:
            # the neff file is not needed anymore
            os.system(f"rm {neff_file} 2> /dev/null")
    # remove any neff file that was left behind
    os.system(f"rm {save_path}/plots/neff_{plot_title}_iteration_*.txt 2> /dev/null")
    
    # read max lnL data from CIP output files
    lnL_files_last_iteration = glob.glob(path_to_main_folder + f"/iteration_{iterations[iteration_prog]}_cip/*lnL*")
    run_diagnostics[plot_title]["latest_grid"] = f"overlap-grid-{iteration_prog+1}.xml.gz"
    run_diagnostics[plot_title]["latest_iteration"] = int(iteration_prog)
    max_lnL, no_points = get_lnL_cut_points(all_net_path)
    ax.set_xticks(np.arange(0, run_diagnostics[plot_title]["latest_iteration"]+1, 1))
    fig.savefig(save_path + f"/plots/Effective_samples_per_CIPworker_{plot_title}.png", bbox_inches='tight')
    plt.close(fig)

def plot_exploration_corner(all_net_path):
    """
    Generates and saves a corner plot for all the points at which marginalized likelihood was evaluated, effectively acting as the exploration plot.

    Args:
        all_net_path (str): File path to all.net
    """
    print('\n--> Plotting exploration corner.')
    use_cols = [1,2,5,8]
    if use_truths:
        P = lsu.xml_to_ChooseWaveformParams_array(truth_file_path)[0]
        truths = [ P.extract_param('m1')/lsu.lsu_MSUN, P.extract_param('m2')/lsu.lsu_MSUN, P.extract_param('s1z'), P.extract_param('s2z')]
    if LISA and not(eccentricity):
        use_cols.append([9,10])
        labels=[r"$m_1$ $(\times 10^6 M_\odot)$", r"$m_2$ $(\times 10^6 M_\odot)$", r"$\chi_{1z}$", r"$\chi_{2z}$", r"$\lambda$", r"$\beta$"]
        if use_truths:
             truths.append([P.extract_param('lambda'),  P.extract_param('beta')])
    if LISA and eccentricity:
        use_cols.append([9,10,11,12])
        labels=[r"$m_1$ $(\times 10^6 M_\odot)$", r"$m_2$ $(\times 10^6 M_\odot)$", r"$\chi_{1z}$", r"$\chi_{2z}$", r"$\lambda$", r"$\beta$", r'$e$', r'$\ell$']
        if use_truths:
            truths.append([P.extract_param('lambda'),  P.extract_param('beta'), P.extract_param('eccentricity'), P.extract_param('meanPerAno')])
    if not(LISA) and eccentricity:
        use_cols.append([9,10])
        labels=[r"$m_1$", r"$m_2$", r"$\chi_{1z}$", r"$\chi_{2z}$",  r'$e$', r'$\ell$']
        if use_truths:
            truths.append([P.extract_param('eccentricity'), P.extract_param('meanPerAno')])
        if precessing:
            use_cols.append([3,4,6,7])
            labels=[r"$m_1$", r"$m_2$", r"$\chi_{1z}$", r"$\chi_{2z}$", r'$e$', r'$\ell$', r"$\chi_{1x}$", r"$\chi_{1y}$", r"$\chi_{2x}$", r"$\chi_{2y}$"]
            if use_truths:
                truths.append([P.extract_param('s1x'), P.extract_param('s1y'), P.extract_param('s2x'), P.extract_param('s2y')])
    if not(LISA) and precessing and not(eccentricity):
        use_cols.append([3,4,6,7])
        labels=[r"$m_1$", r"$m_2$", r"$\chi_{1z}$", r"$\chi_{2z}$", r"$\chi_{1x}$", r"$\chi_{1y}$", r"$\chi_{2x}$", r"$\chi_{2y}$"]
        if use_truths:
            truths.append([P.extract_param('s1x'), P.extract_param('s1y'), P.extract_param('s2x'), P.extract_param('s2y')])
    # Load all.net
    def flatten(arg):
        if not isinstance(arg, list): # if not list
            return [arg]
        return [x for sub in arg for x in flatten(sub)]

    use_cols = flatten(use_cols)
    if use_truths:
        truths = flatten(truths) 
    data = np.loadtxt(all_net_path, usecols = use_cols)
    # If else statement to check if truths are provided are not
    if use_truths:
        P = lsu.xml_to_ChooseWaveformParams_array(truth_file_path)[0]
        fig = corner.corner(data,  truth_color="black", truths=truths, color='cornflowerblue', smooth=None,smooth1d =None, linewidth = 1.0,  plot_datapoints=True, plot_density=False, no_fill_contours=True, contours=False, levels=[0.0], contour_kwargs={"linewidths":1.0},hist_kwargs={"linewidth":1.0, "density": True},labels=labels)
    else:
        fig = corner.corner(data,  color='cornflowerblue', smooth=None,smooth1d =None, linewidth = 1.0,  plot_datapoints=True, plot_density=False, no_fill_contours=True, contours=False, levels=[0.0], contour_kwargs={"linewidths":1.0},hist_kwargs={"linewidth":1.0, "density": True},labels=labels)
    # Save this figure
    fig.savefig(save_path + f'/plots/Exploration_corner_plot.png', bbox_inches='tight')


def plot_lnL_points_in_ranges(all_net_path, lnL_cuts=[2, 5, 10, 12, 15], error_threshold=0.4):
    """
    Plots a histogram of the number of points (marginalized lnL evaluations in all.net) that lie in ranges of lnL below the maximum lnL,
    e.g. 0-2, 2-5, 5-10, 10-12 and 12-15 below the maximum. Only points with Monte Carlo error below the threshold are counted.

    Args:
        all_net_path (str): Path to the all.net file.
        lnL_cuts (list of float): Edges of the ranges, measured below the maximum lnL. Default is [2, 5, 10, 12, 15].
        error_threshold (float): Maximum allowed Monte Carlo error. Default is 0.4.
    """
    print("\n--> Plotting number of points in lnL ranges.")
    data = np.loadtxt(all_net_path)

    # Extract lnL and error columns (same convention as get_lnL_cut_points)
    lnL = data[:, 9]
    error = data[:, 10]
    if LISA and not(eccentricity):
        lnL = data[:, 11]
        error = data[:, 12]
    if LISA and eccentricity:
        lnL = data[:, 13]
        error = data[:, 14]
    if not(LISA) and eccentricity:
        lnL = data[:, 11]
        error = data[:, 12]

    # Remove NaN and inf values and keep only the points with low Monte Carlo error
    finite = np.isfinite(lnL) & np.isfinite(error)
    lnL = lnL[finite]
    error = error[finite]
    lnL = lnL[error <= error_threshold]

    # maximum lnL of the points with low error, this is the maximum lnL reported in the ILE diagnostics
    max_lnL = np.max(lnL)

    # distance below the maximum lnL
    delta_lnL = max_lnL - lnL
    edges = [0] + list(lnL_cuts)
    counts, _ = np.histogram(delta_lnL, bins=edges)

    fig, ax = plt.subplots()
    x = np.arange(len(counts))
    bars = ax.bar(x, counts, width=0.6, color="royalblue")
    ax.bar_label(bars, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{edges[i]}-{edges[i+1]}" for i in range(len(counts))])
    ax.set_xlabel("lnL below max lnL")
    ax.set_ylabel("points")
    ax.set_title(f"max lnL = {max_lnL:0.2f}, points within {edges[-1]} = {np.sum(counts)}")

    fig.savefig(save_path + "/plots/lnL_points_in_ranges.png", bbox_inches='tight')
    plt.close(fig)

def plot_cip_max_lnL(path_to_main_folder, plot_title):
    """
    Plot the maximum log-likelihood (lnL) values sampled from different iterations.
    Args:
        path_to_main_folder (str): The path to main folder containing iteration subfolders with lnL data files.
        plot_title (str): suffix for the plots and filenames.

    """
    print(f"\n--> Plotting sampled lnL by CIP for {plot_title} iterations.")

    iterations = np.arange(0, run_diagnostics[plot_title]["latest_iteration"]+1, 1)

    # get bins for histograms
    all_lnL_hist = []
    for iteration in iterations:
        try:
            files_iteration = glob.glob(path_to_main_folder + f"/iteration_{iteration}_cip/*lnL*")
        except:
            continue
        for file_name in files_iteration:
            try:
                data = np.atleast_1d(np.loadtxt(file_name))
            except Exception:
                continue
            if data.size > 0:
                all_lnL_hist.append(data)
    if not all_lnL_hist:
        raise ValueError(f"No lnL data found in any {path_to_main_folder}/iteration_*_cip folder.")
    all_lnL_hist = np.concatenate(all_lnL_hist)
    bins = np.linspace(np.min(all_lnL_hist), np.max(all_lnL_hist), 40)

    fig, ax = plt.subplots()
    fig_hist, ax_hist = plt.subplots()
    run_diagnostics[plot_title]['cip_sampled_lnL'] = {}
    for iteration in iterations:
        run_diagnostics[plot_title]['cip_sampled_lnL'][iteration] = {}
        try:
            files_iteration = glob.glob(path_to_main_folder + f"/iteration_{iteration}_cip/*lnL*")
        except:
            continue
        collect_lnL = []
        collec_lnL_hist = np.zeros((1,1))
        samples_total = 0
        for j in np.arange(len(files_iteration)):
            try:
                data = np.atleast_1d(np.loadtxt(files_iteration[j]))
            except Exception:
                continue
            if data.size == 0:
                continue
            collect_lnL.append(np.max(data))
            samples_total += len(data)
            collec_lnL_hist = np.vstack([collec_lnL_hist, data[:, None]])
        # missing or empty folder, skip this iteration and keep going with the others
        if len(collect_lnL) == 0:
            print(f"No usable lnL files in {path_to_main_folder}/iteration_{iteration}_cip, skipping iteration {iteration}.")
            continue
        samples_total_per_worker = samples_total/len(collect_lnL)
        collec_lnL_hist = np.delete(collec_lnL_hist, 0, 0)
        collect_lnL = np.array(collect_lnL)
        low_1_std, max_lnL_avg_this_iteration, high_1_std  = np.percentile(collect_lnL, [16,50,84]) # 1 std
        low_2_std, max_lnL_avg_this_iteration, high_2_std  = np.percentile(collect_lnL, [2.5,50,97.5]) # 2 std
        run_diagnostics[plot_title]['cip_sampled_lnL'][iteration].update({
            'avg':np.round(max_lnL_avg_this_iteration, 2),
            '+':np.round(high_2_std, 2),
            '-':np.round(low_2_std, 2)})
        
        ax.errorbar(iteration, max_lnL_avg_this_iteration, yerr = np.array([max_lnL_avg_this_iteration-low_2_std, high_2_std-max_lnL_avg_this_iteration]).reshape(-1,1), color = "royalblue", ecolor = "red", fmt ='.')
        ax.errorbar(iteration, max_lnL_avg_this_iteration, yerr = np.array([max_lnL_avg_this_iteration-low_1_std, high_1_std-max_lnL_avg_this_iteration]).reshape(-1,1), color = "royalblue", ecolor = "green", fmt ='o')
        ax_hist.hist(collec_lnL_hist, label=iteration, histtype='step', linewidth = 1.0, bins=bins, density=True)
    
    # statistics of the latest iteration that has lnL data
    last_iteration_with_data = [iteration for iteration in iterations if run_diagnostics[plot_title]['cip_sampled_lnL'][iteration]][-1]
    run_diagnostics[plot_title]["cip_average_max_lnL_sampled"] = run_diagnostics[plot_title]['cip_sampled_lnL'][last_iteration_with_data]['avg']#np.round(np.mean(collect_lnL), 2)
    run_diagnostics[plot_title]["cip_std_max_lnL_sampled"] = [run_diagnostics[plot_title]['cip_sampled_lnL'][last_iteration_with_data]['+']-run_diagnostics[plot_title]["cip_average_max_lnL_sampled"], run_diagnostics[plot_title]["cip_average_max_lnL_sampled"]-run_diagnostics[plot_title]['cip_sampled_lnL'][last_iteration_with_data]['-']]#np.round(np.std(collect_lnL), 2)
    
    ax.set_xlabel('iteration')
    ax.set_ylabel('lnL')
    ax.axhline(y = run_diagnostics['max_lnL'], linestyle = "--", color="black")
    expected_median = lnL_at_credible_interval(ci=0.5)
    ax.fill_between(iterations, expected_median-1.5, run_diagnostics['max_lnL'], color="green", alpha=0.5)
    ax.set_xticks(iterations)
    fig.savefig(save_path + f"/plots/Maximum_sampled_lnL_per_CIPworker_{plot_title}.png", bbox_inches="tight")

    ax_hist.set_xlabel('lnL')
    ax_hist.legend(loc='upper left', frameon=False)
    ax_hist.axvline(x = run_diagnostics['max_lnL'], linestyle = "--", color="black")
    fig_hist.savefig(save_path + f"/plots/CIP_lnL_distribution_per_iteration_{plot_title}.png", bbox_inches="tight")
    plt.close()

def plot_high_likelihood_expoloration(path_to_main_folder, plot_title):
    """
    Plots high likelihood points over iterations.
    Every iteration up to the latest CIP iteration is checked, plus any consolidated_*.composite file beyond it (ILE can be ahead of CIP).
    An iteration with a missing, empty or unreadable composite file is reported and skipped, the other iterations are still processed.

    Args:
        path_to_main_folder (str): Path to the folder containing the composite files.
        plot_title (str): suffix for the plots and filenames.
    """
    print(f"\n--> Plotting likelihood exploration for {plot_title} iterations.")

    run_diagnostics[plot_title]["composite_information"] = {}

    # iterations to check: up to the latest CIP iteration and any composite file that exists beyond it
    composite_iterations = []
    for composite_file in glob.glob(f"{path_to_main_folder}/consolidated_*.composite"):
        match = re.search(r"consolidated_(\d+)\.composite", os.path.basename(composite_file))
        if match:
            composite_iterations.append(int(match.group(1)))
    iterations = np.arange(0, max([run_diagnostics[plot_title]["latest_iteration"]] + composite_iterations)+1, 1)
    
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("high lnL points")
    ax.set_title(f"Total high lnL points = {run_diagnostics['high_lnL_points']}, max_lnL = {run_diagnostics['max_lnL']}")
    collect_data = []
    collect_iter = []
    print("\nIteration | Global max lnL | High-lnL points | Iteration max lnL | Total lnL points")
    for iteration in iterations:
        composite_file = f"{path_to_main_folder}/consolidated_{iteration}.composite"
        run_diagnostics[plot_title]["composite_information"][iteration] = {}
        note = None
        if not os.path.exists(composite_file):
            note = "file not found"
        elif os.path.getsize(composite_file) == 0:
            note = "file is empty"
        else:
            try:
                max_lnL, no_points, max_lnL_composite, total_points = get_lnL_cut_points(composite_file, composite=True)
            except Exception as e:
                note = f"could not read file: {e}"
        if note is not None:
            print(f"{iteration:9d} | {composite_file}: {note}")
            run_diagnostics[plot_title]["composite_information"][iteration].update({"note": note})
            continue
        percent_high_lnL_points =  np.round(no_points/total_points*100, 2)
        if no_points == 0:
            max_lnL_composite = "NA"
            run_diagnostics[plot_title]["composite_information"][iteration].update({"note": "no points within 15 lnL of the global max with error <= 0.4"})
            print(f"{iteration:9d} | {max_lnL:14.2f} | {no_points:15d} | {'NA':>17} | {total_points:16d}")
        else:
            print(f"{iteration:9d} | {max_lnL:14.2f} | {no_points:15d} | {max_lnL_composite:17.2f} | {total_points:16d}")
        collect_data.append(no_points)
        collect_iter.append(iteration)
        ax.scatter(iteration, no_points, label = f"{max_lnL_composite} ({percent_high_lnL_points}%)", s=25)
        run_diagnostics[plot_title]["composite_information"][iteration].update({
                "max_lnL":max_lnL_composite,
                "high_lnL_points":no_points,
                "percent_high_lnL_points": percent_high_lnL_points,
                "total_evaluations": total_points})
    
    ax.grid(alpha=0.4)
    ax.plot(collect_iter, collect_data, color = "black", linestyle = "--", linewidth = 1.5, alpha = 0.5)
    ax.set_xticks(iterations)
    ax.legend(loc="upper left")
    fig.savefig(save_path + f"/plots/Likelihood_exploration_plot_{plot_title}.png", bbox_inches='tight')
    plt.close(fig)

def plot_histograms(sorted_posterior_file_paths, plot_title, iterations = None, plot_legend = True, JSD = True):
    """
    Plots histograms for specified parameters across different posterior samples.

    Args:
        sorted_posterior_file_paths (list of str): List of file paths to sorted posterior samples.
        plot_title (str): Title for the plots and filenames.
        iterations (list of int or None): Iteration numbers for labeling histograms. Defaults to None, in which case only the final iteration is plotted.
        plot_legend (bool): Whether to include a legend in the histograms. Defaults to True.
        JSD (bool): Whether to calculate and display Jensen-Shannon Divergence between iterations. Defaults to True.
    """
    print(f"\n--> Plotting histograms for {plot_title} iterations.")
    # when you just want to plot final iterations histograms
    if iterations is None: 
        iterations = [-1]
        plot_legend = False
    if use_truths:
        P = lsu.xml_to_ChooseWaveformParams_array(truth_file_path)[0]
    # all_net_data = convert_all_net_to_posterior_format(all_net_path)
    # not_nan_lnL = np.argwhere(all_net_data[:,-3]>=np.max(all_net_data[:,-3]) - 15).flatten()#np.argwhere(~np.isnan(all_net_data[:,-3])).flatten()
    # all_net_data = np.array(all_net_data[not_nan_lnL])
    parameters =  ["mc", "q", "eta", "m1", "m2", "mtot", "s1z", "s2z", "chi_eff", "chiMinus", "chi1", "chi2"]
    # for LISA include skylocation
    if LISA:
        parameters.append("dec")
        parameters.append("ra")
    if eccentricity:
        parameters.append("eccentricity")
        parameters.append("meanPerAno")
    if precessing:
        parameters.append("s1x")
        parameters.append("s1y")
        parameters.append("s2x")
        parameters.append("s2y")
        parameters.append("chi_p")
        parameters.append("chi1_perp")
        parameters.append("chi2_perp")
    
    for parameter in parameters:
        print(f"Plotting histogram for {parameter}")

        fig, ax = plt.subplots()
        ax.set_title(plot_title)
        ax.set_xlabel(parameter)
        ax.set_yticks([])
        
        # get bins
        all_data = []
        for file_path in sorted_posterior_file_paths:
            if parameter in ["chi_eff", "chiMinus", "chi_p", "chi1", "chi2", "chi1_perp", "chi2_perp"]:
                data = get_combination_from_mass_and_spin(np.loadtxt(file_path), parameter)
            else:
                data = np.loadtxt(file_path)[:, get_index_for_parameter(parameter)]
            
            all_data.append(data)
        combined_data = np.concatenate(all_data)
        bins = np.linspace(np.min(combined_data), np.max(combined_data), 40)

        for i, data in enumerate(all_data):
            line_label = str(iterations[i])

            if i > 0 and JSD:
                JS_test = calculate_JS_divergence(data, data_previous)
                line_label += f" ({JS_test.median:0.3f})"

            ax.hist(data, bins=bins, density=True, histtype="step", linewidth=1.0, color=default_colors[i], label=line_label)

            if use_truths:
                factor = lsu.lsu_MSUN if parameter in ["mc", "m1", "m2", "mtot"] else 1
                parameter_extract = {"chi_eff": "xi", "ra": "phi", "dec": "theta"}.get(parameter, parameter)
                ax.axvline(x=P.extract_param(parameter_extract)/factor, linestyle="--", linewidth=1.0, color="black")

            data_previous = data
    
        # don't create legend when only plotting finals iteration's histograms
        if plot_legend: 
            ax.legend(loc = "upper right")
        fig.savefig(save_path + f"/plots/histograms/histogram_{plot_title}_{parameter}.png", bbox_inches='tight')
        plt.close()

def plot_corner(sorted_posterior_file_paths, plot_title, iterations = None, parameters = ["mc", "eta", "chi_eff"], use_truths = False):
    """
    Generates corner plots for posterior samples using a specified plotting executable.

    Args:
        sorted_posterior_file_paths (list of str): List of file paths to sorted posterior samples.
        plot_title (str): Title for the plot, used in filenames.
        iterations (list of int, optional): List of iteration numbers to include in the plot. Defaults to [0] if None.
        parameters (list of str): List of parameters to include in the plot. Defaults to ["mc", "eta", "xi"].
        use_truths (bool): Whether to include truth values in the plot. Defaults to False.
    """
    # for extrinsic, plot ra and dec if not LISA run
    if plot_title == "extrinsic" and not(LISA):
        parameters.append("ra")
        parameters.append("dec")

    print(f"\n--> Plotting corner plot for params ({plot_title}) {parameters}")

    # title, from the all.net used in the corner plots (the extrinsic plots always show the main all.net values)
    if plot_title.startswith("extrinsic") or corner_all_net_path == all_net_path:
        max_lnL, no_points = run_diagnostics["max_lnL"], run_diagnostics["high_lnL_points"]
    else:
        _, no_points, max_lnL, _ = get_lnL_cut_points(corner_all_net_path, composite=True)
    title = f"max_lnL={max_lnL:0.2f},points_cut={no_points}" 

    # Plotting command begins
    plotting_command = f"python {corner_plot_exe} --plot-1d-extra --quantiles None --ci-list [0.9] --use-title {title} --no-special-param-ranges "

    if not plot_title.startswith("extrinsic"):
        plotting_command += f"--composite-file {corner_all_net_path} --lnL-cut 15 --sigma-cut 0.4 "

    # Append iteration-related options to the command
    if iterations is not None:
        plotting_command += "--use-legend "
    else:
        iterations = [0]

    # Include truth file if required
    if use_truths:
        plotting_command += f"--truth-file {truth_file_path} "
    
    # plot grey points (low lnL) when showing multiple iterations
    if "Final" not in plot_title: 
        plotting_command += "--use-all-composite-but-grayscale "
    
    # Add parameter options to the command
    for parameter in parameters:
        plotting_command += f"--parameter {parameter} "

    # Add posterior file paths and labels to the command
    for i, posterior_file in enumerate(sorted_posterior_file_paths):
        plotting_command += f"--posterior-file {posterior_file} --posterior-label {iterations[i]} --posterior-color '{default_colors[i]}' "

    # Append LISA flag if applicable
    if LISA:
        plotting_command += "--LISA "
    
    # Append eccentricity flag if applicable
    if LISA and eccentricity:
        plotting_command += "--eccentricity "

    if not(LISA) and eccentricity:
        plotting_command += "--eccentricity --meanPerAno"
    
    # avoid too much output
    plotting_command += " 2> /dev/null"

    # Execute the plotting command
    os.system(plotting_command)

    # Move and rename output files
    corner_plot_filename = f"corner_{'_'.join(parameters)}.png"
    new_corner_plot_path = f"plots/corner_plots/corner_{'_'.join(parameters)}_{plot_title}.png"
    os.system(f"mv {save_path}/{corner_plot_filename} {save_path}/{new_corner_plot_path}")

    # Move and rename individual parameter plots
    for parameter in parameters:
        os.system(f"mv {save_path}/{parameter}.png {save_path}/plots/1_D_plots/{parameter}_{plot_title}.png")
        os.system(f"mv {save_path}/{parameter}_cum.png {save_path}/plots/1_D_plots/{parameter}_cum_{plot_title}.png") 

def plot_JS_divergence(posterior_1_path, posterior_2_path, posterior_3_path=None, plot_title=None, threshold=JSD_threshold, parameters = ["mc","eta", "m1", "m2"]):
    """
    Plots the Jensen-Shannon divergence between posterior samples for
    selected parameters.

    Args:
        posterior_1_path (str): Path to the first posterior sample file.
        posterior_2_path (str): Path to the second posterior sample file.
        posterior_3_path (str or None): Path to an optional third posterior
            sample file. Default is None.
        plot_title (str or None): Title for the plot and output filename.
            Default is None.
        threshold (float): JSD threshold shown as a horizontal line.
        parameters (list of str): Parameters for which the JSD is computed.
            Default is ["mc", "eta", "m1", "m2"].

    Returns:
        None
    """
    if not(non_spinning):
        parameters.append("s1z")
        parameters.append("s2z")
        parameters.append("chi_eff")
        parameters.append("chiMinus")
    if LISA:
        parameters.append("dec")
        parameters.append("ra")
    if eccentricity:
        parameters.append("eccentricity")
        parameters.append("meanPerAno")
    if precessing:
        parameters.append("s1x")
        parameters.append("s1y")
        parameters.append("s2x")
        parameters.append("s2y")
        parameters.append("chi_p")
        parameters.append("chi1_perp")
        parameters.append("chi2_perp")

    print(f"\n--> Plotting Jensen Shannon Divergence for {parameters} with threshold {threshold}\n")

    posterior_data1 = np.loadtxt(posterior_1_path)
    posterior_data2 = np.loadtxt(posterior_2_path)
    if not(posterior_3_path is None):
        posterior_data3 = np.loadtxt(posterior_3_path)

    JSD_array = [] # collect for last and second-to-last
    JSD_error = []

    JSD_array_third = [] # collect for last and third-to-last
    JSD_error_third = []

    run_diagnostics["JSD"][plot_title] = {}
    run_diagnostics["JSD_3"][plot_title] = {}

    for parameter in parameters:
        if parameter in ["chi_eff", "chiMinus", "chi_p", "chi1", "chi2", "chi1_perp", "chi2_perp"]:
            data1, data2 = get_combination_from_mass_and_spin(posterior_data1, parameter), get_combination_from_mass_and_spin(posterior_data2, parameter)
            JSD = calculate_JS_divergence(data1, data2)
            if not(posterior_3_path is None):
                data3 = get_combination_from_mass_and_spin(posterior_data3, parameter) 
                JSD_3 = calculate_JS_divergence(data1, data3)
        else:
            parameter_n = get_index_for_parameter(parameter)
            JSD = calculate_JS_divergence(posterior_data1[:, parameter_n], posterior_data2[:, parameter_n])
            if not(posterior_3_path is None):
                parameter_n = get_index_for_parameter(parameter)
                JSD_3 = calculate_JS_divergence(posterior_data1[:, parameter_n], posterior_data3[:, parameter_n])
        
        JSD_array.append(JSD.median)
        JSD_error.append([JSD.minus, JSD.plus])
        
        run_diagnostics["JSD"][plot_title][parameter] = np.round(JSD.median, 3)
        
        if not(posterior_3_path is None):
            JSD_array_third.append(JSD_3.median)
            JSD_error_third.append([JSD_3.minus, JSD_3.plus])
            run_diagnostics["JSD_3"][plot_title][parameter] = np.round(JSD_3.median, 3)
    
    fig, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.set_ylabel("JSD")
    ax.axhline( y = threshold, linewidth = 1.0, linestyle = "--", color = "red")
    
    latest_iteration = run_diagnostics[plot_title.split("_")[0]]["latest_iteration"]
    label_second = f"Iter {latest_iteration} vs Iter {latest_iteration-1}"
    
    ax.errorbar(parameters, JSD_array, np.array(JSD_error).T, color="royalblue", ecolor="red", fmt='o', markersize=5, label=label_second)
    if posterior_3_path is not None:
        label_third = f"Iter {latest_iteration} vs Iter {latest_iteration-2}"
        ax.errorbar(parameters, JSD_array_third, np.array(JSD_error_third).T, color="green", ecolor="black", fmt='o', markersize=5, label=label_third)
    #ax.errorbar(parameters, JSD_array, np.array(JSD_error).T,  color = "royalblue", ecolor = "red", fmt ='o', markersize = 5, label='latest-secondlatest')
    #if not(posterior_3_path is None):
    #    ax.errorbar(parameters, JSD_array_third, np.array(JSD_error_third).T,  color = "green", ecolor = "black", fmt ='o', markersize = 5, label='latest-thirdlatest')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', labelrotation=60)
    fig.savefig(save_path+f"/plots/JSD_per_parameter_{plot_title}.png", bbox_inches='tight')
    plt.close(fig)

def write_sample_statistics(posterior, parameters=["mc", "eta", "m1", "m2"], extrinsic = False):
    """
    Computes and writes sample statistics for specified parameters to a file.
    Args:
        posterior (str): Path to the file containing posterior samples.
        parameters (list, optional): List of parameter names for which
            statistics will be computed. Defaults to
            ["mc", "eta", "m1", "m2"].
    """
    if not(non_spinning):
        parameters.append("s1z")
        parameters.append("s2z")
        parameters.append("chi_eff")
        parameters.append("chiMinus")
    if LISA:
        parameters.append("dec")
        parameters.append("ra")
    if eccentricity:
        parameters.append("eccentricity")
        parameters.append("meanPerAno")
    if precessing:
        parameters.append("s1x")
        parameters.append("s1y")
        parameters.append("s2x")
        parameters.append("s2y")
        parameters.append("chi_p")
        parameters.append("chi1_perp")
        parameters.append("chi2_perp")
        parameters.append("chi1")
        parameters.append("chi2")
    if use_truths:
        P = lsu.xml_to_ChooseWaveformParams_array(truth_file_path)[0]
    
    print(f"\n--> Writing sample statistics for parameters: {parameters}")
    
    posterior = np.loadtxt(posterior)
    
    f = open(save_path + f"/plots/sample_statistics.txt", "w")
    f.write("Note: limits are equal-tailed 90th percentile\n")
    run_diagnostics["sample_statistics"] = {}
    for parameter in parameters:
        if parameter in ["chi_eff", "chiMinus", "chi_p", "chi1", "chi2", "chi1_perp", "chi2_perp"]: 
            samples_here = get_combination_from_mass_and_spin(posterior, parameter) 
        else:
            parameter_n = get_index_for_parameter(parameter, extrinsic = extrinsic)
            samples_here = posterior[:,parameter_n]
        statistics = get_sample_statistics(samples_here)
        run_diagnostics["sample_statistics"][parameter] = np.round(statistics, 2)
        line = f"{parameter}: median = {statistics[1]:0.3f}, upper limit = {statistics[2]:0.3f}, lower limit = {statistics[0]:0.3f}"
        line = (f"{parameter:12s} = "f"{statistics[1]:8.3f} "f"[{statistics[0]:8.3f}, {statistics[2]:8.3f}]")

        if use_truths:
            factor = 1
            parameter_extract = parameter
            if parameter in ["mc", "m1", "m2", "mtot"]:
                factor = lsu.lsu_MSUN
            if parameter == "chi_eff":
                parameter_extract = "xi"
            if parameter == "ra":
                parameter_extract = "phi"
            if parameter == "dec":
                parameter_extract = "theta"
            line += f"   truth: {P.extract_param(parameter_extract)/factor:8.3f}"
        f.write(line + "\n")
    max_sample, lnL = get_lnL_cut_points(all_net_path, lnL_cut=15, error_threshold=0.4, composite=False, provide_max_lnL_point=True)
    f.close()

def plot_log_likelihood(extrinsic_path):
    """
    Plots a histogram of the log-likelihood (lnL) distribution from extrinsic samples file.

    Args:
        extrinsic_path (str): File path to the extrinsic data file containing sampled parameters.
                             The log-likelihood values are assumed to be in column index 18.
    """
    extrinsic_data = np.loadtxt(extrinsic_path+"/extrinsic_posterior_samples.dat")
    index_lnL = 18
    plt.cla()
    plt.title('lnL distribution')
    plt.xlabel('lnL')
    plt.ylabel('Points')
    plt.hist(extrinsic_data[:, index_lnL], histtype='step', color='black', bins=40)
    plt.savefig(save_path+f"/plots/lnL_distribution_from_extrinsic_step.png", bbox_inches='tight')
    plt.cla()
    plt.close()

def evaluate_run(run_diagnostics):
    """
    Evaluates the run and writes diagnostics information to a file.

    Args:
        run_diagnostics (dict): Dictionary containing diagnostics data.

    Returns:
        None
    """

    def write_header(f, title):
        f.write("\n")
        f.write("###########################################################################################\n")
        f.write(f"# {title}\n")
        f.write("###########################################################################################\n")

    def write_dict_information(f, title, data):
        f.write(f"\n\t{title}:\n")
        if not data:
            f.write("\tNo data available.\n")
            return

        for key, value in data.items():
            f.write(f"\t\t{key}: {value}\n")

    def write_composite_information(f, plot_title):
        composite_information = run_diagnostics.get(plot_title, {}).get("composite_information", {})

        f.write(f"\n{plot_title} likelihood exploration:\n")
        if not composite_information:
            f.write("\tNo composite information available.\n")
            return

        f.write("\titeration | max lnL | high lnL points | percent high lnL points | total evaluations\n")
        for iteration, values in composite_information.items():
            percent = values.get('percent_high_lnL_points', 'NA')
            percent = f"{percent}%" if percent != 'NA' else 'NA'
            note = f"  ({values['note']})" if values.get('note') else ""
            f.write(f"\t{iteration:>9} | {values.get('max_lnL', 'NA'):>7} | {values.get('high_lnL_points', 'NA'):>15} | {percent:>23} | {values.get('total_evaluations', 'NA'):>17}{note}\n")

    def write_neff_information(f, plot_title):
        requested_neff = run_diagnostics.get(plot_title, {}).get("CIP_neff", {})
        achieved_neff = run_diagnostics.get(plot_title, {}).get("CIP_neff_achieved", {})

        f.write(f"\n{plot_title} CIP n-eff:")
        write_dict_information(f, "Requested n-eff", requested_neff)
        write_dict_information(f, "Achieved n-eff", achieved_neff)

        if not requested_neff or not achieved_neff:
            return True

        first_requested_neff = requested_neff[list(requested_neff.keys())[0]]
        last_requested_neff = requested_neff[list(requested_neff.keys())[-1]]
        last_achieved_neff = achieved_neff[list(achieved_neff.keys())[-1]]

        neff_is_good = True
        
        if last_achieved_neff < last_requested_neff:
            f.write(f"\t--> Latest achieved n-eff ({last_achieved_neff}) is below the requested value ({last_requested_neff}).  To address this, try narrowing the parameter space or switching to a different sampler. Alternatively, you can reduce the neff for each CIP job (>10) and increase the number of CIP jobs submitted per iteration.\n")
            neff_is_good = False
        if first_requested_neff <= last_achieved_neff < last_requested_neff and first_requested_neff != last_requested_neff:
            if run_diagnostics.get("run_is_complete") == False:
                f.write(f"\t--> Latest achieved n-eff ({last_achieved_neff}) is below the final target ({last_requested_neff}), although it exceeds the initial target ({first_requested_neff}). Since the run is incomplete, this warning can be ignored.\n")

        return neff_is_good

    def write_cip_sampled_lnL_information(f, plot_title):
        diagnostics = run_diagnostics.get(plot_title, {})

        latest_iteration = diagnostics.get("latest_iteration", None)
        average_lnL = diagnostics.get("cip_average_max_lnL_sampled", None)
        std_lnL = diagnostics.get("cip_std_max_lnL_sampled", None)
        sampled_lnL = diagnostics.get("cip_sampled_lnL", {})

        #f.write(f"\n{plot_title} CIP sampled lnL:\n")

        #if sampled_lnL:
        #    f.write("\titeration | median max lnL | low | high\n")
        #    for iteration, values in sampled_lnL.items():
        #        f.write(f"\t{iteration:>9} | {values.get('avg', 'NA'):>14} | {values.get('-', 'NA'):>3} | {values.get('+', 'NA'):>4}\n")

        if average_lnL is None or std_lnL is None:
            f.write("\tNo sampled lnL summary available.\n")
            return True

        sampling_is_good = True

        try:
            expected_median = lnL_at_credible_interval(ci=0.5)
            if expected_median - average_lnL > 1.5:
                f.write(f"\t--> The expected median max lnL ({expected_median:0.2f}) is more than 1.5 above the average sampled max lnL ({average_lnL}). CIP may not be sampling the peak well enough, or there may be interpolation error near the peak.\n")
                sampling_is_good = False
            if run_diagnostics.get("run_is_complete") == False:
                f.write(f"\t--> Since the run is incomplete, this warning can be ignored.\n")
        except Exception:
            f.write("\tCould not compute the expected median lnL.\n")

        return sampling_is_good

    def write_jsd_information(f, jsd_labels, title, threshold=JSD_threshold):
        f.write(f"\n{title}:\n")

        found_jsd = False
        jsd_is_good = True

        for jsd_label in jsd_labels:
            jsd_data = run_diagnostics.get("JSD", {}).get(jsd_label, {})
            jsd_data_3 = run_diagnostics.get("JSD_3", {}).get(jsd_label, {})

            if not jsd_data and not jsd_data_3:
                continue

            found_jsd = True
            #f.write(f"\n\t{jsd_label}:\n")

            if jsd_data:
                latest_iteration = run_diagnostics.get(jsd_label.split("_")[0], {}).get("latest_iteration", "latest")
                f.write(f"\t  iteration {latest_iteration} vs iteration {latest_iteration - 1}:\n")

                for parameter, value in jsd_data.items():
                    marker = "  <-- above threshold" if value > threshold else ""
                    f.write(f"\t\t{parameter}: {value}{marker}\n")
                    if value > threshold:
                        jsd_is_good = False

            if jsd_data_3:
                latest_iteration = run_diagnostics.get(jsd_label.split("_")[0], {}).get("latest_iteration", "latest")
                f.write(f"\t  iteration {latest_iteration} vs iteration {latest_iteration - 2}:\n")

                for parameter, value in jsd_data_3.items():
                    marker = "  <-- above threshold" if value > threshold else ""
                    f.write(f"\t\t{parameter}: {value}{marker}\n")
                    if value > threshold:
                        jsd_is_good = False
        if not found_jsd:
            f.write("\tNo JSD information available.\n")

        return jsd_is_good

    diagnostics_path = save_path + "/plots/Diagnostics.txt"

    with open(diagnostics_path, "w") as f:

        write_header(f, "Run summary")
        f.write(f"Run directory = {path}\n")
        f.write(f"Save directory = {save_path}/plots\n")
        if run_diagnostics.get("Subdag"):
            f.write(f"Subdag directory = {find_posteriors_in_sub(path, return_subdag_folder_path=True)}\n")
        f.write(f"LISA analysis = {LISA}\n")
        f.write(f"Eccentric analysis = {eccentricity}\n")
        f.write(f"Precessing analysis = {precessing}\n")
        f.write(f"Non-spinning analysis = {non_spinning}\n")
        f.write(f"Truth file used = {truth_file_path if use_truths else None}\n")
        f.write(f"Run complete = {run_diagnostics.get('run_is_complete', False)}\n")

        write_header(f, "ILE diagnostics")

        ILE_is_good = True

        f.write(f"Total number of marginalized lnL evaluations = {run_diagnostics.get('total_lnL_evaluations', 'NA')}\n")
        f.write(f"Total number of high marginalized lnL points = {run_diagnostics.get('total_high_lnL_points', 'NA')}\n")
        f.write(f"Total number of high marginalized lnL points used = {run_diagnostics.get('high_lnL_points', 'NA')}\n")
        f.write(f"Total number of high marginalized lnL points not used due to large error = {run_diagnostics.get('high_lnL_points_with_large_error', 'NA')}\n")

        if "max_lnL" in run_diagnostics:
            f.write(f"Maximum marginalized lnL = {run_diagnostics['max_lnL']}\n")
            f.write(f"Approximate SNR captured = {np.sqrt(2 * run_diagnostics['max_lnL']):0.2f}\n")

        f.write(f"Number of high lnL points with lnL cuts [12, 10, 5, 2] = [{run_diagnostics.get('high_lnL_points_with_lnLcut_12', 'NA')}, {run_diagnostics.get('high_lnL_points_with_lnLcut_10', 'NA')}, {run_diagnostics.get('high_lnL_points_with_lnLcut_5', 'NA')}, {run_diagnostics.get('high_lnL_points_with_lnLcut_2', 'NA')}]\n")

        total_high_points = run_diagnostics.get("total_high_lnL_points", 0)
        large_error_points = run_diagnostics.get("high_lnL_points_with_large_error", 0)
        high_points_used = run_diagnostics.get("high_lnL_points", None)

        if total_high_points > 0 and large_error_points / total_high_points > 0.5:
            f.write("\t--> More than half of the high-likelihood points have large Monte Carlo error. Consider reducing d-max, increasing d-min, increasing n-max, or changing the sampler.\n")
            ILE_is_good = False

        if high_points_used is not None:
            if high_points_used <= 500:
                f.write("\t--> Number of high-likelihood points is less than 500. The initial grid may not have sufficient resolution. Consider reducing the parameter space or increasing the number of grid points.\n")
                ILE_is_good = False

            elif high_points_used < 5000:
                latest_grid = run_diagnostics.get("Main", {}).get("latest_grid", "latest grid")
                f.write(f"\t--> Number of high-likelihood points is less than 5000. Consider rerunning with {latest_grid} as the starting grid and copying this run's all.net as bonus.composite.\n")
                ILE_is_good = False
        if ILE_is_good ==False and run_diagnostics.get("run_is_complete") == False:
                f.write(f"\t--> Since the run is incomplete, this warning can be ignored.\n")
        write_composite_information(f, "Main")

        if run_diagnostics.get("Subdag"):
            write_composite_information(f, "Subdag")

        f.write("\n")
        if ILE_is_good:
            f.write("\t--> ILE status: GOOD! <--\n")
        else:
            f.write("\t--> ILE status: BAD! <--\n")

        write_header(f, "CIP diagnostics")

        CIP_is_good = True

        for plot_title in ["Main", "Subdag"]:
            if plot_title not in run_diagnostics:
                continue

            if not run_diagnostics[plot_title]:
                continue

            diagnostics = run_diagnostics[plot_title]

            f.write(f"\n{plot_title} CIP summary:\n")
            f.write(f"\tLatest iteration = {diagnostics.get('latest_iteration', 'NA')}\n")
            f.write(f"\tLatest grid = {diagnostics.get('latest_grid', 'NA')}\n")
            
            latest_iteration = diagnostics.get("latest_iteration", None)
            average_lnL = diagnostics.get("cip_average_max_lnL_sampled", None)
            std_lnL = diagnostics.get("cip_std_max_lnL_sampled", None)
            sampled_lnL = diagnostics.get("cip_sampled_lnL", {})
            
            if latest_iteration is None:
                f.write(f"\tAverage max lnL sampled by CIP = {average_lnL:0.2f} (+{std_lnL[0]:0.2f}, -{std_lnL[1]:0.2f})\n")
            else:
                f.write(f"\tAverage max lnL sampled by CIP in iteration {latest_iteration} = {average_lnL:0.2f} (+{std_lnL[0]:0.2f}, -{std_lnL[1]:0.2f})\n")
            if "max_lnL" in run_diagnostics:
                f.write(f"\tMaximum lnL in all.net = {run_diagnostics['max_lnL']}\n")
            
            sampling_is_good = write_cip_sampled_lnL_information(f, plot_title)
            neff_is_good = write_neff_information(f, plot_title)

            if not neff_is_good or not sampling_is_good:
                CIP_is_good = False

        jsd_is_good_main = write_jsd_information(f, ["Main_iteration"], "Main iteration JSD")
        jsd_is_good_subdag = write_jsd_information(f, ["Subdag_iteration"], "Subdag iteration JSD")

        if not jsd_is_good_main or not jsd_is_good_subdag:
            latest_grid = run_diagnostics.get("Main", {}).get("latest_grid", "latest grid")
            f.write(f"\n\t--> At least one JSD value is above {JSD_threshold}. This suggests that the run may not have converged. Consider rerunning with {latest_grid} as the starting grid and copying this run's all.net as bonus.composite.\n")
            if run_diagnostics.get("run_is_complete") is False:
                f.write(f"\t--> Since the run is not complete, this warning can be ignored.\n")
            CIP_is_good = False

        f.write("\n")
        if CIP_is_good:
            f.write("\t--> CIP status: GOOD! <--\n")
        else:
            f.write("\t--> CIP status: BAD! <--\n")

        #if "sample_statistics" in run_diagnostics:
        #    write_header(f, "Posterior sample statistics")
        #    write_dict_information(f, "Equal-tailed 90% intervals", run_diagnostics["sample_statistics"])

        write_header(f, "Visual diagnostics")
        f.write("\t 1) Is the 90% credible interval mostly around the red points? If not, it could be that the run needs more iterations.")
        f.write(f"\n\t 2) Has the parameter space been sufficiently explored? Are there blue points around the red points? Continuing the run will help if this is true with {run_diagnostics['Main']['latest_grid']} as your starting grid and copying this run's all.net as bonus.composite in your new run directory")
        f.write("\n\t 3) Is the approximate SNR captured close to True SNR? A significant difference implies the inference got stuck at a local lnL maxima. Happens rarely")
    print("\n###########################################################################################")
    print("# Run diagnostics")
    print("###########################################################################################")
    print(f"Diagnostics written to {diagnostics_path}")

    for key, value in run_diagnostics.items():
        print(f"{key}: {value}")

def check_extrinsic_present(path):
    """
    Checks whether an extrinsic posterior sample file exists.

    Args:
        path (str): Path to the run directory.

    Returns:
        bool:
            True if extrinsic_posterior_samples.dat exists, otherwise False.
    """
    return os.path.exists(f"{path}/extrinsic_posterior_samples.dat")

def mask_paths(text):
    """
    Shortens absolute paths (e.g. /home/user/runs/run1) to .../last_directory, so the html file can be shared
    without revealing usernames or the directory structure of the cluster.

    Args:
        text (str): Text of a section from Diagnostics.txt.

    Returns:
        str: Text with the absolute paths shortened.
    """
    return re.sub(r"(?<!\S)/(?:[^\s/]+/)*[^\s/]+", lambda m: ".../" + m.group(0).rstrip("/").split("/")[-1], text)

def read_diagnostics_sections(diagnostics_path):
    """
    Splits Diagnostics.txt into its sections using the headers written by evaluate_run.

    Args:
        diagnostics_path (str): Path to Diagnostics.txt.

    Returns:
        dict: Section title (e.g. 'ILE diagnostics') -> text of the section.
    """
    sections = {}
    current = None
    with open(diagnostics_path, "r") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# ") and i > 0 and lines[i-1].startswith("#####"):
            current = line[2:].strip()
            sections[current] = []
            continue
        if line.startswith("#####"):
            continue
        if current is not None:
            sections[current].append(line)
    return {title: "\n".join(text).strip("\n") for title, text in sections.items()}

def diagnostics_text_to_html(text):
    """
    Escapes diagnostics text for html and highlights status lines and warnings.

    Args:
        text (str): Text of a section from Diagnostics.txt.

    Returns:
        str: html-safe text with <span> tags around status lines and warnings.
    """
    lines = []
    for line in text.split("\n"):
        escaped = html.escape(line)
        stripped = line.strip()
        if "status: GOOD" in stripped:
            lines.append(f'<span class="good">{escaped}</span>')
        elif "status: BAD" in stripped:
            lines.append(f'<span class="bad">{escaped}</span>')
        elif stripped.startswith("-->") or "<-- above threshold" in stripped:
            lines.append(f'<span class="warn">{escaped}</span>')
        else:
            lines.append(escaped)
    return "\n".join(lines)

def split_parameter_string(parameter_string):
    """
    Converts the parameter string in a corner plot filename (e.g. 'mc_eta_chi_eff')
    into a readable list (e.g. 'mc, eta, chi_eff'). Falls back to the input string if it cannot be split.

    Args:
        parameter_string (str): Parameters joined by underscores.

    Returns:
        str: Readable, comma separated parameters.
    """
    known = ["mc", "q", "eta", "m1", "m2", "mtot", "s1x", "s1y", "s1z", "s2x", "s2y", "s2z", "a1z", "a2z", "chi_eff", "chiMinus", "chi1", "chi2", "chi_p", "chi1_perp", "chi2_perp",
             "dec", "ra", "eccentricity", "meanPerAno", "distance", "incl", "phiorb", "psi", "time", "m1_source", "m2_source", "mtotal_source"]
    known.sort(key = len, reverse = True) # prefer the longest match (e.g. m1_source over m1)
    tokens = re.findall("|".join(re.escape(k) for k in known), parameter_string)
    if "_".join(tokens) == parameter_string:
        return ", ".join(tokens)
    return parameter_string

def build_plot_manifest(save_path, reference_file=None):
    """
    Searches the plots folder and builds a description of the available plots for the html file.
    Histograms and corner plots are identified from their filenames, so no parameter lists are hard coded.

    Args:
        save_path (str): Path containing the plots folder.
        reference_file (str, optional): Plots modified before this file are not included, so plots left over from an earlier run
            (e.g. with different options) are not shown as current. Defaults to None, which includes all plots.

    Returns:
        dict: Histograms, corner plots and figure groups (per section) that exist in the plots folder.
    """
    plots = save_path + "/plots"
    reference_time = os.path.getmtime(reference_file) if reference_file and os.path.exists(reference_file) else None
    stale_plots = []
    def is_current(file_path):
        if reference_time is not None and os.path.getmtime(file_path) < reference_time:
            stale_plots.append(os.path.relpath(file_path, plots))
            return False
        return True
    run_order = {"Main": 0, "Subdag": 1, "Extrinsic": 2}
    parameter_order = ["mc", "q", "eta", "m1", "m2", "mtot", "s1z", "s2z", "chi_eff", "chiMinus", "chi1", "chi2", "dec", "ra", "eccentricity", "meanPerAno",
                       "s1x", "s1y", "s2x", "s2y", "chi_p", "chi1_perp", "chi2_perp"]
    manifest = {"histograms": [], "corners": [], "figures": {}}

    # 1D histograms: histogram_{plot_title}_{parameter}.png
    for file_path in glob.glob(plots + "/histograms/histogram_*.png"):
        if not is_current(file_path):
            continue
        name = os.path.basename(file_path)[len("histogram_"):-len(".png")]
        for title in ["Main_Final", "Subdag_Final", "Main", "Subdag"]:
            if name.startswith(title + "_"):
                parameter = name[len(title)+1:]
                manifest["histograms"].append({"run": title.split("_")[0], "latest": title.endswith("_Final"), "key": parameter, "file": os.path.relpath(file_path, plots)})
                break
    manifest["histograms"].sort(key = lambda e: (run_order[e["run"]], e["latest"], parameter_order.index(e["key"]) if e["key"] in parameter_order else len(parameter_order), e["key"]))

    # corner plots: corner_{parameters}_{plot_title}.png
    for file_path in glob.glob(plots + "/corner_plots/corner_*.png"):
        if not is_current(file_path):
            continue
        name = os.path.basename(file_path)[len("corner_"):-len(".png")]
        for title in ["Main_Final", "Subdag_Final", "extrinsic_source_mass", "extrinsic", "Main", "Subdag"]:
            if name.endswith("_" + title):
                parameters = name[:-(len(title)+1)]
                run = "Extrinsic" if title.startswith("extrinsic") else title.split("_")[0]
                latest = True if run == "Extrinsic" else title.endswith("_Final")
                manifest["corners"].append({"run": run, "latest": latest, "key": split_parameter_string(parameters), "file": os.path.relpath(file_path, plots)})
                break
    manifest["corners"].sort(key = lambda e: (run_order[e["run"]], e["latest"], len(e["key"]), e["key"]))

    # figure groups: each group is a title and a list of (label, file). A label can have a list of files, which are shown side by side.
    # Only files that exist are kept. A group can have a description (shown above the plot) and a note (small text shown below the plot)
    def figure_group(title, items, description=None, note=None):
        existing = []
        for label, files in items:
            file_list = [files] if isinstance(files, str) else list(files)
            file_list = [file for file in file_list if os.path.exists(f"{plots}/{file}") and is_current(f"{plots}/{file}")]
            if not file_list:
                continue
            existing.append({"label": label, "file": file_list[0]} if isinstance(files, str) else {"label": label, "files": file_list})
        group = {"title": title, "items": existing}
        if description:
            group["description"] = description
        if note:
            group["note"] = note
        return group if existing else None

    figures = {
        "ile": [
            figure_group("Points in lnL ranges", [("all.net", "lnL_points_in_ranges.png")],
                         note="Numbers are for points with Monte Carlo error < 0.4."),
            figure_group("High lnL points per iteration", [("Main", "Likelihood_exploration_plot_Main.png"), ("Subdag", "Likelihood_exploration_plot_Subdag.png")],
                         note="In the legend, the first number is the maximum lnL found in that iteration and the second is the percentage of high lnL points in that iteration."),
        ],
        "cip": [
            figure_group("Effective samples per CIP worker", [("Main", "Effective_samples_per_CIPworker_Main.png"), ("Subdag", "Effective_samples_per_CIPworker_Subdag.png")],
                         description="This shows the effective number of samples (neff) per worker. The total neff is the sum of the neff of all the workers. Earlier iterations may have different neff values than later iterations. Increasing worker numbers typically correspond to workers used in later iterations."),
            figure_group("lnL distribution and maximum lnL sampled per CIP worker",
                         [("Main", ("CIP_lnL_distribution_per_iteration_Main.png", "Maximum_sampled_lnL_per_CIPworker_Main.png")),
                          ("Subdag", ("CIP_lnL_distribution_per_iteration_Subdag.png", "Maximum_sampled_lnL_per_CIPworker_Subdag.png"))],
                         description="The lnL distribution produced by the sampler using the interpolant is compared with the maximum lnL value in the `all.net` training data. This distribution should stabilize across iterations and remain close to the maximum lnL value."),
            figure_group("Jensen-Shannon divergence", [("Main: latest vs previous iterations", "JSD_per_parameter_Main_iteration.png"),
                                                       ("Subdag: latest vs previous iterations", "JSD_per_parameter_Subdag_iteration.png")],
                         description="This metric assesses convergence by comparing the posterior distributions from the previous and latest iterations using the Jensen-Shannon divergence (JSD). Ideally, the JSD should be below 0.005. Convergence can also be assessed visually using the histograms and corner plots below.\nNote: RIFT may use a different internal convergence threshold, but the JSD remains a useful diagnostic."),
        ],
        "visual": [
            figure_group("Exploration corner (all.net)", [("all.net", "Exploration_corner_plot.png")],
                         description="Points at which the marginalized likelihood was evaluated."),
            figure_group("lnL distribution of extrinsic samples", [("extrinsic", "lnL_distribution_from_extrinsic_step.png")]),
        ],
    }
    manifest["figures"] = {section: [g for g in groups if g is not None] for section, groups in figures.items()}
    if stale_plots:
        print(f"Left out {len(stale_plots)} plot(s) older than the start of this run: {', '.join(stale_plots[:5])}{' ...' if len(stale_plots) > 5 else ''}")
    return manifest

def encode_image(image_path, max_width=1600):
    """
    Reads a png file and returns it as a base64 data URI so that it can be embedded in an html file.
    Images wider than max_width are downscaled (keeping the aspect ratio) and the png is optimized, to keep the html file small.

    Args:
        image_path (str): Path to the png file.
        max_width (int): Maximum width in pixels. Use 0 to embed the image as it is. Default is 1600.

    Returns:
        str: data URI of the image.
    """
    try:
        from PIL import Image # installed along with matplotlib
        image = Image.open(image_path)
        if max_width and image.width > max_width:
            image = image.resize((max_width, int(round(image.height * max_width / image.width))), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        raw = buffer.getvalue()
    except Exception:
        with open(image_path, "rb") as f:
            raw = f.read()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

def embed_images_in_manifest(manifest, plots, max_width=1600):
    """
    Replaces the (relative) image paths in the manifest by base64 data URIs, so the html file does not depend on the plots folder.

    Args:
        manifest (dict): Output of build_plot_manifest.
        plots (str): Path to the plots folder, the image paths in the manifest are relative to it.
        max_width (int): Maximum width in pixels of the embedded images. Use 0 for full resolution.
    """
    entries = list(manifest["histograms"]) + list(manifest["corners"])
    for groups in manifest["figures"].values():
        for group in groups:
            entries += group["items"]
    print(f"Embedding {sum(len(entry['files']) if 'files' in entry else 1 for entry in entries)} images in the html file.")
    for entry in entries:
        if "files" in entry:
            entry["files"] = [encode_image(f"{plots}/{file}", max_width=max_width) for file in entry["files"]]
        else:
            entry["file"] = encode_image(f"{plots}/{entry['file']}", max_width=max_width)

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RIFT run diagnostics</title>
<style>
:root { --bg:#ffffff; --fg:#1d2330; --muted:#667085; --card:#f6f7f9; --border:#d9dde3; --accent:#2b5fd9; --good:#1a7f37; --bad:#c62828; --warn:#b26a00; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --fg:#e6e8ec; --muted:#9aa3b2; --card:#1c2027; --border:#2f3541; --accent:#7aa2ff; --good:#4cc27a; --bad:#ff6b6b; --warn:#f0b24a; }
}
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background:var(--bg); color:var(--fg); line-height:1.5; }
header { position:sticky; top:0; z-index:10; background:var(--bg); border-bottom:1px solid var(--border); padding:10px 24px; display:flex; align-items:baseline; gap:24px; flex-wrap:wrap; }
header h1 { font-size:20px; margin:0; }
header .muted { margin:0; }
nav a { margin-right:16px; color:var(--accent); text-decoration:none; font-weight:600; }
main { max-width:1500px; margin:0 auto; padding:16px 24px 64px; }
section { margin-top:32px; scroll-margin-top:70px; }
h2 { font-size:40px; line-height:1.1; margin:0; }
.section-box { display:flex; align-items:center; gap:16px; flex-wrap:wrap; border:2px solid var(--accent); border-radius:12px; background:var(--card); padding:16px 24px; }
.section-box .badge { margin-left:0; }
.intro { margin:16px 2px 12px; font-size:15px; }
h3 { font-size:17px; margin:28px 0 8px; }
.muted { color:var(--muted); font-size:14px; }
pre { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px 16px; overflow-x:auto; white-space:pre-wrap; tab-size:4; font-size:13px; }
.good { color:var(--good); font-weight:600; }
.bad { color:var(--bad); font-weight:600; }
.warn { color:var(--warn); }
.badge { font-size:13px; padding:2px 10px; border-radius:12px; margin-left:10px; vertical-align:middle; color:#fff; }
.badge.good { background:var(--good); } .badge.bad { background:var(--bad); } .badge.unknown { background:var(--muted); }
.controls { display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin:8px 0 12px; }
.btn-row { display:inline-flex; border:1px solid var(--border); border-radius:8px; overflow:hidden; }
.btn-row button { background:var(--card); color:var(--fg); border:0; border-right:1px solid var(--border); padding:6px 14px; cursor:pointer; font-size:14px; }
.btn-row button:last-child { border-right:0; }
.btn-row button.active { background:var(--accent); color:#fff; }
select { background:var(--card); color:var(--fg); border:1px solid var(--border); border-radius:8px; padding:6px 10px; font-size:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(var(--min, 380px), 1fr)); gap:12px; }
figure { margin:0; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:8px; }
figure img { width:100%; height:auto; display:block; background:#fff; border-radius:4px; }
figcaption { font-size:13px; color:var(--muted); margin-top:4px; text-align:center; }
.single img { max-width:1100px; background:#fff; border-radius:4px; border:1px solid var(--border); }
.single .pair { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.single .pair img { max-width:100%; width:100%; }
.desc { font-size:14px; margin:2px 0 10px; white-space:pre-line; }
#figures-ile { display:grid; grid-template-columns:1fr 1fr; gap:24px; align-items:start; }
#figures-ile .single img { width:100%; max-width:100%; }
.note { font-size:12px; color:var(--muted); margin:6px 2px 0; }
code { background:var(--card); border:1px solid var(--border); border-radius:4px; padding:0 4px; font-size:0.92em; }
figure img, .single img { cursor:zoom-in; }
#lightbox { display:none; position:fixed; inset:0; z-index:100; background:rgba(0,0,0,0.85); align-items:center; justify-content:center; cursor:zoom-out; }
#lightbox img { max-width:96vw; max-height:96vh; background:#fff; border-radius:4px; }
details { margin-top:16px; }
summary { cursor:pointer; font-weight:600; }
</style>
</head>
<body>
<header>
  <h1>RIFT run diagnostics</h1>
  <p class="muted">Generated __TIMESTAMP__</p>
  <nav><a href="#ile">ILE</a><a href="#cip">CIP</a><a href="#visual">Visual</a></nav>
</header>
<main>
<details open><summary>Run summary</summary><pre>__SUMMARY__</pre></details>

<section id="ile">
  <div class="section-box"><h2>ILE</h2><span class="badge __ILE_CLASS__">__ILE_STATUS__</span></div>
  <p class="intro">Here, we examine the marginalized likelihood evaluations used to construct the intrinsic posterior distributions. Ideally, there should be thousands of high-lnL points, with good coverage across the lnL ranges. Later iterations should also maintain a high fraction of high-lnL points, indicating continued exploration of the high-likelihood region.</p>
  <pre>__ILE_TEXT__</pre>
  <div id="figures-ile"></div>
</section>

<section id="cip">
  <div class="section-box"><h2>CIP</h2><span class="badge __CIP_CLASS__">__CIP_STATUS__</span></div>
  <p class="intro">Here, we examine the effective number of samples obtained by each worker, assess the health of the interpolant by comparing its likelihood distribution with the underlying lnL values, and evaluate convergence.</p>
  <pre>__CIP_TEXT__</pre>
  <div id="figures-cip"></div>
  <h3>1D posterior histograms</h3>
  <div id="hist-gallery"></div>
</section>

<section id="visual">
  <div class="section-box"><h2>Visual</h2></div>
  <pre>__VISUAL_TEXT__</pre>
  <h3>Corner plots</h3>
  <p class="desc">This provides an excellent visual check of convergence. The posteriors should typically cluster around the red points. If the red regions are surrounded by blue and green points, it indicates that the high-lnL region has been thoroughly explored. The colored points represent the maximum lnL and maximum lnL &minus; 15 levels, while the grey points (if shown) represent lower lnL values.</p>
  <div id="corner-gallery"></div>
  <div id="figures-visual"></div>
</section>
</main>

<script>
const MANIFEST = __MANIFEST__;

// Row of buttons. Calls onSelect(label) when one is clicked.
function buttonRow(labels, selected, onSelect, display) {
  const row = document.createElement("div");
  row.className = "btn-row";
  labels.forEach(label => {
    const b = document.createElement("button");
    b.textContent = display ? display(label) : label;
    if (label === selected) b.className = "active";
    b.onclick = () => onSelect(label);
    row.appendChild(b);
  });
  return row;
}

// Click an image to enlarge it, click again (or press Esc) to close
const lightbox = document.createElement("div");
lightbox.id = "lightbox";
const lightboxImg = document.createElement("img");
lightbox.appendChild(lightboxImg);
document.body.appendChild(lightbox);
function openLightbox(src) { lightboxImg.src = src; lightbox.style.display = "flex"; }
lightbox.onclick = () => { lightbox.style.display = "none"; };
document.addEventListener("keydown", e => { if (e.key === "Escape") lightbox.style.display = "none"; });

// Text where `code` is shown as code
function richText(element, text) {
  text.split("`").forEach((part, i) => {
    if (i % 2) { const c = document.createElement("code"); c.textContent = part; element.appendChild(c); }
    else element.appendChild(document.createTextNode(part));
  });
  return element;
}

// One image, click to enlarge
function plotFigure(file, caption) {
  const fig = document.createElement("figure");
  const img = document.createElement("img");
  img.src = file; img.alt = caption;
  img.onclick = () => openLightbox(file);
  fig.appendChild(img);
  if (caption) {
    const cap = document.createElement("figcaption");
    cap.textContent = caption;
    fig.appendChild(cap);
  }
  return fig;
}

// Gallery with toggles: run (Main/Subdag/...), latest vs all iterations, and which parameter(s) to show.
function makeGallery(rootId, entries, keyName, minWidth, singleDefault) {
  const root = document.getElementById(rootId);
  if (!entries.length) { root.innerHTML = '<p class="muted">No plots found.</p>'; return; }
  const runs = [...new Set(entries.map(e => e.run))];
  const state = {run: runs[0], latest: true, key: null};
  const controls = document.createElement("div");
  controls.className = "controls";
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.style.setProperty("--min", minWidth + "px");
  root.append(controls, grid);

  function render() {
    const forRun = entries.filter(e => e.run === state.run);
    const modes = [...new Set(forRun.map(e => e.latest))];
    if (!modes.includes(state.latest)) state.latest = modes[0];
    const shown = forRun.filter(e => e.latest === state.latest);
    const keys = [...new Set(shown.map(e => e.key))];
    // start on a single parameter when singleDefault is set, otherwise on all of them
    if (state.key === null || (state.key !== "__all__" && !keys.includes(state.key))) state.key = singleDefault ? keys[0] : "__all__";

    controls.innerHTML = "";
    if (runs.length > 1) controls.appendChild(buttonRow(runs, state.run, r => { state.run = r; render(); }));
    if (modes.length > 1) {
      controls.appendChild(buttonRow([true, false], state.latest, m => { state.latest = m; render(); }, m => m ? "Latest iteration" : "All iterations"));
    }
    const select = document.createElement("select");
    const all = document.createElement("option");
    all.value = "__all__"; all.textContent = "All " + keyName;
    select.appendChild(all);
    keys.forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = k; select.appendChild(o); });
    select.value = state.key;
    select.onchange = () => { state.key = select.value; render(); };
    controls.appendChild(select);

    grid.innerHTML = "";
    shown.filter(e => state.key === "__all__" || e.key === state.key).forEach(e => grid.appendChild(plotFigure(e.file, e.key)));
  }
  render();
}

// Figure group: title, optional description above the plot, buttons to switch between the items (e.g. Main/Subdag) when there is more than one,
// the plot (or plots side by side) and an optional small note below
function makeFigureGroups(rootId, groups) {
  const root = document.getElementById(rootId);
  groups.forEach(group => {
    const wrapper = document.createElement("div");
    wrapper.className = "group";
    root.appendChild(wrapper);
    const title = document.createElement("h3");
    title.textContent = group.title;
    wrapper.appendChild(title);
    if (group.description) {
      const description = document.createElement("p");
      description.className = "desc";
      wrapper.appendChild(richText(description, group.description));
    }
    const holder = document.createElement("div");
    holder.className = "single";
    wrapper.appendChild(holder);
    if (group.note) {
      const note = document.createElement("p");
      note.className = "note";
      wrapper.appendChild(richText(note, group.note));
    }
    let selected = group.items[0].label;
    function render() {
      holder.innerHTML = "";
      if (group.items.length > 1) {
        const controls = document.createElement("div");
        controls.className = "controls";
        controls.appendChild(buttonRow(group.items.map(i => i.label), selected, l => { selected = l; render(); }));
        holder.appendChild(controls);
      }
      const item = group.items.find(i => i.label === selected);
      const files = item.files || [item.file];
      const images = document.createElement("div");
      if (files.length > 1) images.className = "pair";
      files.forEach(file => {
        const img = document.createElement("img");
        img.src = file; img.alt = group.title;
        img.onclick = () => openLightbox(file);
        images.appendChild(img);
      });
      holder.appendChild(images);
    }
    render();
  });
}

makeFigureGroups("figures-ile", MANIFEST.figures.ile);
makeFigureGroups("figures-cip", MANIFEST.figures.cip);
makeGallery("hist-gallery", MANIFEST.histograms, "parameters", 380, true);
makeGallery("corner-gallery", MANIFEST.corners, "parameter sets", 560, false);
makeFigureGroups("figures-visual", MANIFEST.figures.visual);
</script>
</body>
</html>
"""

def write_html_diagnostics(save_path, diagnostics_file="Diagnostics.txt", html_file="Diagnostics.html", max_width=1600, show_paths=False):
    """
    Writes an html page (ILE, CIP and Visual sections) in the plots folder, using the text in Diagnostics.txt
    and the plots that exist in the plots folder. The plots are embedded in the html file (base64), so the file is
    self-contained and can be shared without the plots folder.

    Args:
        save_path (str): Path containing the plots folder.
        diagnostics_file (str): Name of the text diagnostics file inside the plots folder.
        html_file (str): Name of the html file to create inside the plots folder.
        max_width (int): Maximum width in pixels of the embedded images, wider images are downscaled. Use 0 for full resolution.
        show_paths (bool): If False (default), absolute paths in the ILE, CIP and Visual text are shortened to the last directory, so the file can be shared. The Run summary always keeps the full paths.
    """
    print("\n--> Writing HTML diagnostics.")
    plots = save_path + "/plots"
    sections = read_diagnostics_sections(f"{plots}/{diagnostics_file}")
    if not show_paths:
        # the Run summary always keeps the full paths
        sections = {title: (text if title == "Run summary" else mask_paths(text)) for title, text in sections.items()}
    manifest = build_plot_manifest(save_path, reference_file=f"{plots}/.run_start")
    embed_images_in_manifest(manifest, plots, max_width=max_width)

    def get_status(text, name):
        if f"{name} status: GOOD" in text:
            return "good", "GOOD"
        if f"{name} status: BAD" in text:
            return "bad", "BAD"
        return "unknown", "UNKNOWN"

    ile_text = sections.get("ILE diagnostics", "No ILE diagnostics available.")
    cip_text = sections.get("CIP diagnostics", "No CIP diagnostics available.")
    visual_text = sections.get("Visual diagnostics", "")
    ile_class, ile_status = get_status(ile_text, "ILE")
    cip_class, cip_status = get_status(cip_text, "CIP")

    page = HTML_TEMPLATE
    replacements = {
        "__TIMESTAMP__": time.strftime("%Y-%m-%d %H:%M:%S"),
        "__SUMMARY__": html.escape(sections.get("Run summary", "")),
        "__ILE_CLASS__": ile_class,
        "__ILE_STATUS__": ile_status,
        "__ILE_TEXT__": diagnostics_text_to_html(ile_text),
        "__CIP_CLASS__": cip_class,
        "__CIP_STATUS__": cip_status,
        "__CIP_TEXT__": diagnostics_text_to_html(cip_text),
        "__VISUAL_TEXT__": diagnostics_text_to_html(visual_text),
        "__MANIFEST__": json.dumps(manifest),
    }
    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)

    with open(f"{plots}/{html_file}", "w") as f:
        f.write(page)
    print(f"HTML diagnostics written to {plots}/{html_file} ({os.path.getsize(f'{plots}/{html_file}')/1e6:0.1f} MB)")
###########################################################################################
# Generate plots
###########################################################################################
# create plots folder
create_plots_folder(save_path)

# marks the start of this run, plots older than this marker are left out of Diagnostics.html
open(save_path + "/plots/.run_start", "w").close()

# finding posterior files
main_posterior_files, main_iterations = find_posteriors_in_main(path)
if len(main_posterior_files) > 7:
    limit_main_iterations = 5
    main_posterior_files, main_iterations = find_posteriors_in_main(path, limit_iterations=limit_main_iterations)
subdag_posterior_files, subdag_iterations = find_posteriors_in_sub(path)

# all.net used in all the corner plots: if there is a subdag, the larger of the main and subdag all.net
corner_all_net_path = choose_all_net_for_corner_plots(all_net_path, find_posteriors_in_sub(path, return_subdag_folder_path=True))

# plot neff
try:
    plot_neff_data(path, plot_title='Main')
except Exception as e:
    print(e)
    # run this function so some information in run_diagnostics dict gets populated.
    get_lnL_cut_points(all_net_path, lnL_cut=15, error_threshold=0.4, composite=False)
    print("Couldn't plot CIP neff per worker for Main iterations.")

# plot exploration corner
try:
    plot_exploration_corner(corner_all_net_path)
except Exception as e:
    print(e)
    print("Couldn't plot exploration corner plot")

# plot number of points in lnL ranges
try:
    plot_lnL_points_in_ranges(all_net_path)
except Exception as e:
    print(e)
    print("Couldn't plot number of points in lnL ranges")

# plot sampled max lnL
try:
    plot_cip_max_lnL(path,  plot_title='Main')
except Exception as e:
    print(e)
    print("Couldn't plot max lnL sampled by CIP for Main iterations.")

# plot likelihood exploration
try:
    plot_high_likelihood_expoloration(path, plot_title='Main')
except Exception as e:
    print(e)
    print("Couldn't plot high likelihod exploration plot for Main iterations.")

# plot histograms
plot_histograms(main_posterior_files, plot_title="Main", iterations=main_iterations, JSD = False)
plot_histograms([main_posterior_files[-1]], plot_title="Main_Final", iterations=None, JSD = False)

# plot corner plots
if LISA:
    plot_corner(main_posterior_files, "Main", iterations = main_iterations, use_truths = use_truths)
    if eccentricity: 
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "eccentricity", "meanPerAno", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mc", "eta", "chi_eff", "eccentricity", "meanPerAno", "dec", "ra"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["m1", "m2", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mtot", "q", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], use_truths = use_truths)
    else:
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "a1z", "a2z", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mc", "eta", "chi_eff", "dec", "ra"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["m1", "m2", "a1z", "a2z", "dec", "ra"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mtot", "q", "a1z", "a2z", "dec", "ra"], use_truths = use_truths)
else:
    # block with no spin and no eccentricity + precession
    if non_spinning:
        plot_corner(main_posterior_files, "Main",  parameters = ["mc", "eta"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final",  parameters = ["mc", "eta"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["m1", "m2"], use_truths = use_truths)
    else:
        plot_corner(main_posterior_files, "Main", iterations = main_iterations, use_truths = use_truths)
    
    # block with spins but split based on precession and eccentricity
    if eccentricity and not(precessing):
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "eccentricity", "meanPerAno"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mc", "eta", "chi_eff", "eccentricity", "meanPerAno"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mtot", "q", "s1z", "s2z", "eccentricity", "meanPerAno"], use_truths = use_truths)
    elif precessing and not(eccentricity):
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "chi_p"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mc", "eta", "chi_eff", "chi_p"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mtot", "q", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], use_truths = use_truths)
    elif precessing and eccentricity:
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "chi_p",  "eccentricity", "meanPerAno"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mc", "eta", "chi_eff", "chi_p", "eccentricity", "meanPerAno"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mtot", "q", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], use_truths = use_truths)
    elif not(precessing) and not(eccentricity) and not(non_spinning):
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "s1z", "s2z"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["m1", "m2", "s1z", "s2z"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Main_Final", parameters = ["mtot", "q", "s1z", "s2z"], use_truths = use_truths)

# plot JS test
try:
    plot_JS_divergence(main_posterior_files[-1], main_posterior_files[-2], main_posterior_files[-3], "Main_iteration") # the last secondlast main iteration and last thirdlast main iteration
except:
    try:
        plot_JS_divergence(main_posterior_files[-1], main_posterior_files[-2], None, "Main_iteration") # the last secondlast main iteration
    except:
        print("Couldn't plot Jensen Shannon Divergence plot")

# is there a subdag? If not, don't plot!
if len(subdag_posterior_files) == 0:
    analyse_subdag = False
else:
    analyse_subdag = True

# if the number of subdag iterations is high, only show five iterations to prevent overcrowding
if len(subdag_posterior_files) > 8 and analyse_subdag == True:
    limit_subdag_iterations = 5 
    subdag_posterior_files, subdag_iterations = find_posteriors_in_sub(path, limit_iterations=limit_subdag_iterations)

if analyse_subdag:
    subdag_path = find_posteriors_in_sub(path, return_subdag_folder_path=True)
    print(f'\t-->Analyzing Subdag iterations found in {subdag_path}<--')
    try:
        plot_neff_data(subdag_path, plot_title='Subdag')
    except Exception as e:
        print(e)
        print("Couldn't plot CIP neff per worker for Subdag iterations.")
    
    # plot sampled max lnL
    try:
        plot_cip_max_lnL(subdag_path,  plot_title='Subdag')
    except Exception as e:
        print(e)
        print("Couldn't plot max lnL sampled by CIP for Subdag iterations.")
    
    # plot likelihood exploration
    try:
        plot_high_likelihood_expoloration(subdag_path, plot_title='Subdag')
    except Exception as e:
        print(e)
        print("Couldn't plot high likelihod exploration plot for Subdag iterations.")


    plot_histograms(subdag_posterior_files, plot_title="Subdag", iterations=subdag_iterations, JSD = False)
    plot_histograms([subdag_posterior_files[-1]], plot_title="Subdag_Final", iterations=None, JSD = False)

    # plot corner plots
    if LISA:
        plot_corner(subdag_posterior_files, "Subdag", iterations=subdag_iterations, use_truths=use_truths)
        if eccentricity:
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "eccentricity", "meanPerAno", "dec", "ra"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mc", "eta", "chi_eff", "eccentricity", "meanPerAno", "dec", "ra"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["m1", "m2", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mtot", "q", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], use_truths=use_truths)
        else:
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "dec", "ra"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "a1z", "a2z", "dec", "ra"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mc", "eta", "chi_eff", "dec", "ra"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["m1", "m2", "a1z", "a2z", "dec", "ra"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mtot", "q", "a1z", "a2z", "dec", "ra"], use_truths=use_truths)
    else:
        if non_spinning:
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mc", "eta"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["m1", "m2"], use_truths=use_truths)
        else:
            plot_corner(subdag_posterior_files, "Subdag", iterations=subdag_iterations, use_truths=use_truths)

        if eccentricity and not(precessing) and not(non_spinning):
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "eccentricity", "meanPerAno"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mc", "eta", "chi_eff", "eccentricity", "meanPerAno"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mtot", "q", "s1z", "s2z", "eccentricity", "meanPerAno"], use_truths=use_truths)
        elif precessing and not(eccentricity) and not(non_spinning):
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "chi_p"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mc", "eta", "chi_eff", "chi_p"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mtot", "q", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], use_truths=use_truths)
        elif precessing and eccentricity and not(non_spinning):
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "chi_p", "eccentricity", "meanPerAno"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mc", "eta", "chi_eff", "chi_p", "eccentricity", "meanPerAno"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mtot", "q", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], use_truths=use_truths)
        elif not(precessing) and not(eccentricity) and not(non_spinning):
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "s1z", "s2z"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["m1", "m2", "s1z", "s2z"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag_Final", parameters=["mtot", "q", "s1z", "s2z"], use_truths=use_truths)

    try:
        plot_JS_divergence(subdag_posterior_files[-1], subdag_posterior_files[-2], None, "Subdag_iteration") # the last two subdag iterations
    except:
        try:
            plot_JS_divergence(subdag_posterior_files[-1], subdag_posterior_files[-2], subdag_posterior_files[-3], "Subdag_iteration")
        except:
           print("Couldn't plot Jensen Shannon Divergence plot.") 

run_diagnostics['run_is_complete'] = False
if check_extrinsic_present(path):
    plot_corner([f"{path}/extrinsic_posterior_samples.dat"], "extrinsic", parameters = ["distance", "incl", "phiorb", "psi", "time"], use_truths = use_truths)
    plot_corner([f"{path}/extrinsic_posterior_samples.dat"], "extrinsic_source_mass", parameters = ["m1_source", "m2_source", "mtotal_source"], use_truths = False)
    write_sample_statistics(f"{path}/extrinsic_posterior_samples.dat", extrinsic = True)
    plot_log_likelihood(path)
    run_diagnostics['run_is_complete'] = True

# run diagnostics
evaluate_run(run_diagnostics)

# html diagnostics (ILE, CIP and Visual sections)
try:
    write_html_diagnostics(save_path, max_width=opts.html_max_width, show_paths=opts.html_show_paths)
except Exception as e:
    print(e)
    print("Couldn't write the HTML diagnostics file.")
