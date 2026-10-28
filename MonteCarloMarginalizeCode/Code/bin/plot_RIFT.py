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
    # puffing corner plots, created also if the plots folder was made by an earlier version
    os.makedirs(save_path + "/plots/Exploration/puffing", exist_ok=True)

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

###########################################################################################
# Puffing: corner plots of the puffed points (puffball-N.xml.gz, output of util_ParameterPuffball.py) and the latest posterior
###########################################################################################
PUFFING_FOLDER = "Exploration/puffing"   # the corner plots are named like the other corner plots: corner_{parameters}_{Main, Main_Final, Subdag or Subdag_Final}.png
PUFFING_THRESHOLDS = {"min_corr_retention": 0.5,    # correlation of eccentricity and meanPerAno in the puffs relative to the grid, checked when the grid has |corr| > 0.5
                      "min_hit_efficiency": 0.3}    # fraction of high lnL points among the puffs relative to the grid
PUFFING_MAX_ITERATIONS = 5               # at most this many puffball iterations (evenly spaced, always including the latest) are shown in the "all iterations" corner plot
PUFFING_LABELS = {"m1": r"$m_1$", "m2": r"$m_2$", "mc": r"$\mathcal{M}_c$", "eta": r"$\eta$", "q": r"$q$", "mtot": r"$M$", "s1z": r"$\chi_{1z}$", "s2z": r"$\chi_{2z}$",
                  "chi_eff": r"$\chi_{\rm eff}$", "chiMinus": r"$\chi_{-}$", "eccentricity": r"$e$", "meanPerAno": r"$\ell$"}

def puffing_parameters(m1, m2, s1z, s2z, eccentric_values=None, mean_anomaly_values=None):
    """
    Parameters used in the puffing corner plots, from component masses (solar masses) and aligned spins. The same function is used
    for the puffball xml.gz files and for the posterior samples, so that both are described identically.
    chiMinus = (m1 s1z - m2 s2z)/M as in the rest of this script.
    """
    parameters = {"m1": m1, "m2": m2, "mc": (m1*m2)**0.6/(m1 + m2)**0.2, "eta": m1*m2/(m1 + m2)**2, "q": m2/m1, "mtot": m1 + m2, "s1z": s1z, "s2z": s2z,
                  "chi_eff": (m1*s1z + m2*s2z)/(m1 + m2), "chiMinus": (m1*s1z - m2*s2z)/(m1 + m2)}
    if eccentric_values is not None:
        parameters["eccentricity"] = np.asarray(eccentric_values, dtype=float)
        parameters["meanPerAno"] = np.mod(np.asarray(mean_anomaly_values, dtype=float), 2*np.pi)
    return parameters

def find_puffing_files(folder):
    """
    Finds the puffball files of a run. puffball-N.xml.gz is the output of util_ParameterPuffball.py in iteration N, made from overlap-grid-N.xml.gz.
    Both are evaluated by ILE in iteration N, so the ILE results are in consolidated_N.composite.

    Args:
        folder (str): Path to the run directory (or the subdag directory).

    Returns:
        dict: iteration -> {"puffball": path, "grid": path or None, "composite": path or None}, sorted by iteration.
    """
    iterations = {}
    for puffball_file in glob.glob(f"{folder}/puffball-*.xml.gz"):
        match = re.search(r"puffball-(\d+)\.xml\.gz$", os.path.basename(puffball_file))
        if not match:
            continue
        n = int(match.group(1))
        grid_file = f"{folder}/overlap-grid-{n}.xml.gz"
        composite_file = f"{folder}/consolidated_{n}.composite"
        iterations[n] = {"puffball": puffball_file, "grid": grid_file if os.path.exists(grid_file) else None,
                         "composite": composite_file if os.path.exists(composite_file) and os.path.getsize(composite_file) > 0 else None}
    return dict(sorted(iterations.items()))

def unwrap_angles(angles, center, period=2*np.pi):
    """Returns the angles within +-period/2 of center (unwrapped about center)."""
    return center + (np.mod(np.asarray(angles) - center + period/2., period) - period/2.)

def circular_summary(angles, period=2*np.pi):
    """
    Describes samples of a periodic parameter. The circle is cut in the middle of the largest empty arc of the samples, and the
    samples are unwrapped about the opposite point.

    Args:
        angles (np.ndarray): Samples of the periodic parameter.
        period (float): Period of the parameter. Default is 2 pi.

    Returns:
        tuple: (dict with center, cut, largest_empty_arc, resultant (mean resultant length), std_raw (std of the angle in [0, period)),
                std_unwrapped, np.ndarray of the unwrapped samples)
    """
    ang = np.mod(angles, period)
    sorted_ang = np.sort(ang)
    gaps = np.diff(np.r_[sorted_ang, sorted_ang[0] + period])
    k = np.argmax(gaps)
    cut = np.mod(sorted_ang[k] + gaps[k]/2., period)
    center = np.mod(cut + period/2., period)
    unwrapped = unwrap_angles(ang, center, period)
    summary = {"center": float(center), "cut": float(cut), "largest_empty_arc": float(gaps[k]),
               "resultant": float(np.abs(np.mean(np.exp(2j*np.pi*ang/period)))),
               "std_raw": float(np.std(ang)), "std_unwrapped": float(np.std(unwrapped))}
    return summary, unwrapped

def read_grid_parameters(xml_file):
    """
    Reads a puffball xml.gz file with lalsimutils and returns the parameters used in the puffing corner plots.

    Args:
        xml_file (str): Path to the xml.gz file.

    Returns:
        dict: arrays from puffing_parameters (m1, m2 in solar masses, mc, eta, q, mtot, s1z, s2z, chi_eff, chiMinus and, for eccentric runs, eccentricity and meanPerAno).
    """
    P_list = lsu.xml_to_ChooseWaveformParams_array(xml_file)
    if len(P_list) == 0:
        raise ValueError(f"No points in {xml_file}")
    m1 = np.array([P.m1 for P in P_list], dtype=float)/lsu.lsu_MSUN
    m2 = np.array([P.m2 for P in P_list], dtype=float)/lsu.lsu_MSUN
    s1z = np.array([P.s1z for P in P_list], dtype=float)
    s2z = np.array([P.s2z for P in P_list], dtype=float)
    if eccentricity:
        return puffing_parameters(m1, m2, s1z, s2z, [P.eccentricity for P in P_list], [P.meanPerAno for P in P_list])
    return puffing_parameters(m1, m2, s1z, s2z)

def composite_column_layout():
    """Column indices of an all.net / consolidated_N.composite file (the same layout as get_lnL_cut_points). None for LISA, whose extra columns are not assumed."""
    if LISA:
        return None
    layout = {"m1": 1, "m2": 2, "s1z": 5, "s2z": 8, "lnL": 9, "err": 10}
    if eccentricity:
        layout.update({"eccentricity": 9, "meanPerAno": 10, "lnL": 11, "err": 12})
    return layout

puffing_file_cache = {}   # loaded composite / all.net files, so each file is read once per run

def load_evaluations(file_path):
    """Loads a consolidated_N.composite or all.net file (cached)."""
    if file_path not in puffing_file_cache:
        puffing_file_cache[file_path] = np.loadtxt(file_path, ndmin=2)
    return puffing_file_cache[file_path]

def match_to_composite(parameters, composite_file, tolerance=5e-3):
    """
    Finds the ILE result (lnL and error) of each point in a consolidated_N.composite (or all.net) file by matching the coordinates
    (m1, m2, s1z, s2z, and eccentricity and meanPerAno for eccentric runs).

    Args:
        parameters (dict): Output of read_grid_parameters.
        composite_file (str): Path to the composite or all.net file.
        tolerance (float): Maximum distance, in units of the standard deviation of each column, for a match.

    Returns:
        dict or None: lnL, err (nan if not matched) and matched_fraction; None if the column layout is not known.
    """
    layout = composite_column_layout()
    if layout is None:
        return None
    from scipy.spatial import cKDTree
    data = load_evaluations(composite_file)
    names = [name for name in ["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"] if name in layout and name in parameters]
    composite_coordinates = np.column_stack([data[:, layout[name]] for name in names])
    points = np.column_stack([parameters[name] for name in names])
    scale = np.maximum(np.std(composite_coordinates, axis=0), 1e-12)
    distance, index = cKDTree(composite_coordinates/scale).query(points/scale)
    matched = distance < tolerance
    return {"lnL": np.where(matched, data[index, layout["lnL"]], np.nan), "err": np.where(matched, data[index, layout["err"]], np.nan),
            "matched_fraction": float(np.mean(matched))}

def diagnose_composite_match(parameters, composite_file, label, tolerance=5e-3):
    """Explains why the puffed points are not (all) found in a composite: wrong file, a column that differs, or only partial coverage."""
    from scipy.spatial import cKDTree
    layout = composite_column_layout()
    data = load_evaluations(composite_file)
    scale = lambda column: max(np.std(data[:, column]), 1e-12)
    masses = np.column_stack([data[:, layout["m1"]]/scale(layout["m1"]), data[:, layout["m2"]]/scale(layout["m2"])])
    distance, index = cKDTree(masses).query(np.column_stack([parameters["m1"]/scale(layout["m1"]), parameters["m2"]/scale(layout["m2"])]))
    found = distance < tolerance
    if found.mean() < 0.5:
        return f"the m1, m2 of the points are not in {label} (is it the file of the iteration that evaluated them?)"
    worst, worst_name, worst_difference = 0., None, 0.
    for name in ["s1z", "s2z", "eccentricity", "meanPerAno"]:
        if name not in layout or name not in parameters:
            continue
        difference = np.abs(parameters[name][found] - data[index[found], layout[name]])
        if name == "meanPerAno":
            difference = np.minimum(difference, 2*np.pi - difference)
        relative = np.median(difference)/scale(layout[name])
        if relative > worst:
            worst, worst_name, worst_difference = relative, name, np.median(difference)
    if worst > 1e-2:
        return f"m1 and m2 of the points are found in {label} but {worst_name} differs (median |point - file| = {worst_difference:.3g}): check the {worst_name} column or parametrization in {label}"
    return f"only part of the points are in {label} (failed or missing ILE evaluations)"

def find_lnL_of_points(parameters, composite_file, composite_label, all_net_file):
    """
    Finds the lnL of the points in the consolidated_N.composite of their iteration, and in all.net (which contains every evaluation) if the composite is missing
    or contains too few of them. Partial matches are used and reported.

    Returns:
        dict: lnL, err (arrays or None), matched_fraction, source (file name) and note (str, "" if everything is fine).
    """
    tries = []   # (label, match result, reason when there is no result)
    for label, file_path in [(composite_label, composite_file), ("all.net", all_net_file)]:
        if file_path is None or not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            tries.append((label, None, f"{label} not found"))
            continue
        try:
            result = match_to_composite(parameters, file_path)
        except Exception as e:
            tries.append((label, None, f"{label} could not be read ({e})"))
            continue
        tries.append((label, result, None))
        if result is not None and result["matched_fraction"] >= 0.5:
            break
    usable = [t for t in tries if t[1] is not None]
    best = max(usable, key=lambda t: t[1]["matched_fraction"]) if usable else None
    path_of = lambda label: composite_file if label == composite_label else all_net_file
    if best is None or best[1]["matched_fraction"] < 0.05:
        reasons = [t[2] for t in tries if t[2]]
        if best is not None:
            reasons.append(diagnose_composite_match(parameters, path_of(best[0]), best[0]))
        return {"lnL": None, "err": None, "matched_fraction": 0. if best is None else best[1]["matched_fraction"], "source": None, "note": "lnL not available: " + "; ".join(reasons)}
    before = tries[:tries.index(best)]
    notes = [t[2] for t in before if t[2]] + [f"only {100*t[1]['matched_fraction']:.0f}% of the points were found in {t[0]}" for t in before if t[1] is not None]
    if best[1]["matched_fraction"] < 0.8:
        notes.append(f"lnL found for {100*best[1]['matched_fraction']:.0f}% of the points in {best[0]}: " + diagnose_composite_match(parameters, path_of(best[0]), best[0]))
    elif notes:
        notes.append(f"lnL taken from {best[0]}")
    return {"lnL": best[1]["lnL"], "err": best[1]["err"], "matched_fraction": best[1]["matched_fraction"], "source": best[0], "note": "; ".join(notes)}

def read_puffball_arguments(folder):
    """
    Reads the puff factor of util_ParameterPuffball.py from the .sub file (e.g. PUFF.sub) of the run.

    Args:
        folder (str): Path to the run directory (or the subdag directory).

    Returns:
        dict: found (bool), file (name of the .sub file) and puff_factor (float, 1 if --puff-factor is not given, the default of util_ParameterPuffball.py).
    """
    info = {"found": False, "file": None, "puff_factor": None}
    for sub_file in sorted(glob.glob(f"{folder}/*.sub")):
        try:
            with open(sub_file, "r") as f:
                content = f.read()
        except Exception:
            continue
        if "util_ParameterPuffball" not in content:
            continue
        factor = re.findall(r"--puff-factor\s+([0-9.eE+-]+)", content)
        info.update({"found": True, "file": os.path.basename(sub_file), "puff_factor": float(factor[-1]) if factor else 1.0})
        break
    return info

def high_lnL_fraction(lnL, err, max_lnL, lnL_cut=15, error_threshold=0.4):
    """Fraction of the evaluated points (lnL not nan) that are high lnL points (within lnL_cut of max_lnL, error <= error_threshold), as in get_lnL_cut_points."""
    evaluated = ~np.isnan(lnL)
    if evaluated.sum() == 0:
        return np.nan
    return float(np.sum(evaluated & (lnL >= max_lnL - lnL_cut) & (err <= error_threshold))/evaluated.sum())

def max_lnL_low_error(all_net_file, lnL_cut=15, error_threshold=0.4):
    """Maximum lnL among the high lnL points (within lnL_cut of the maximum, Monte Carlo error <= error_threshold) of an all.net file, as in get_lnL_cut_points. None if it cannot be read."""
    layout = composite_column_layout()
    if layout is None or not os.path.exists(all_net_file):
        return None
    data = np.loadtxt(all_net_file, ndmin=2)
    lnL, error = data[:, layout["lnL"]], data[:, layout["err"]]
    good = ~np.isnan(lnL)
    lnL, error = lnL[good], error[good]
    if len(lnL) == 0:
        return None
    high = (lnL >= lnL.max() - lnL_cut) & (error <= error_threshold)
    return float(lnL[high].max()) if high.any() else float(lnL.max())

def find_latest_posterior(folder):
    """Path to the latest posterior (the same one as used for the last histograms and corner plots) in the run directory (or the subdag directory)."""
    posteriors, _ = find_posteriors_in_main(folder)
    if len(posteriors) == 0:
        raise FileNotFoundError(f"No posterior_samples* files found in {folder}")
    return posteriors[-1]

def puffing_parameter_sets():
    """Parameter sets of the puffing corner plots, shown in the menu of the html file: the cartesian parameters (first, shown by default) and the chirp mass based ones."""
    spin = [] if non_spinning else ["s1z", "s2z"]
    eccentric_parameters = ["eccentricity", "meanPerAno"] if eccentricity else []
    return [["m1", "m2"] + spin + eccentric_parameters, ["mc", "eta"] + ([] if non_spinning else ["chi_eff"]) + eccentric_parameters]

def read_posterior_parameters(posterior_file):
    """Reads the posterior samples (same column layout as in the rest of this script) and returns the parameters of puffing_parameters."""
    data = np.loadtxt(posterior_file, ndmin=2)
    column = lambda name: data[:, get_index_for_parameter(name)]
    if eccentricity:
        return puffing_parameters(column("m1"), column("m2"), column("s1z"), column("s2z"), column("eccentricity"), column("meanPerAno"))
    return puffing_parameters(column("m1"), column("m2"), column("s1z"), column("s2z"))

def puffing_footprint(puff, posterior, quantile=0.99, max_posterior_points=20000):
    """
    Finds the puffed points that lie outside the footprint of the posterior samples. Distances are measured in m1, m2, s1z, s2z (and eccentricity and
    meanPerAno, unwrapped about the posterior's center) after whitening with the covariance of the posterior samples, so the thin correlated
    directions of the posterior count. A puff is outside the footprint if its nearest posterior sample is farther away than the nearest-neighbour
    distance exceeded by only (1 - quantile) of the posterior samples themselves.

    Args:
        puff (dict): Puffed points, output of read_grid_parameters.
        posterior (dict): Posterior samples, output of read_posterior_parameters.
        quantile (float): Quantile of the posterior's own nearest-neighbour distances used as the footprint size. Default is 0.99.
        max_posterior_points (int): Larger posteriors are randomly thinned to this number of distinct samples.

    Returns:
        dict or None: outside (bool array for the puffed points), threshold and distance (whitened units); None if there are too few distinct posterior samples.
    """
    from scipy.spatial import cKDTree
    names = ["m1", "m2", "s1z", "s2z"] + (["eccentricity", "meanPerAno"] if eccentricity else [])
    posterior_values = np.column_stack([posterior[name] for name in names])
    puff_values = np.column_stack([puff[name] for name in names])
    if eccentricity:
        summary, posterior_unwrapped = circular_summary(posterior["meanPerAno"])
        posterior_values[:, -1] = posterior_unwrapped
        puff_values[:, -1] = unwrap_angles(puff["meanPerAno"], summary["center"])
    mean = posterior_values.mean(axis=0)
    covariance = np.cov(posterior_values.T)
    covariance = covariance + 1e-6*np.diag(np.diag(covariance))   # keeps the whitening finite for almost degenerate directions
    L = np.linalg.cholesky(covariance)
    whiten = lambda values: np.linalg.solve(L, (values - mean).T).T
    posterior_white = np.unique(np.round(whiten(posterior_values), 6), axis=0)   # posterior samples can contain duplicates
    if len(posterior_white) < 200:
        return None
    if len(posterior_white) > max_posterior_points:
        posterior_white = posterior_white[np.random.default_rng(0).choice(len(posterior_white), max_posterior_points, replace=False)]
    tree = cKDTree(posterior_white)
    own_distance = tree.query(posterior_white, k=2)[0][:, 1]
    threshold = float(np.quantile(own_distance, quantile))
    distance = tree.query(whiten(puff_values))[0]
    return {"outside": distance > threshold, "threshold": threshold, "distance": distance}

def posterior_contours(ax, x, y, x_limits, y_limits, credible_levels=(0.5, 0.9), bins=100):
    """
    Draws the 50% and 90% highest-density contours of the posterior samples (smoothed 2D histogram) in black on an axis.

    Returns:
        bool: False if the contours could not be drawn (too few distinct samples).
    """
    from scipy.ndimage import gaussian_filter
    inside = (x >= x_limits[0]) & (x <= x_limits[1]) & (y >= y_limits[0]) & (y <= y_limits[1])
    if inside.sum() < 50:
        return False
    histogram, x_edges, y_edges = np.histogram2d(x[inside], y[inside], bins=bins, range=[x_limits, y_limits])
    histogram = gaussian_filter(histogram, 1.5)
    if histogram.max() <= 0:
        return False
    ordered = np.sort(histogram.ravel())[::-1]
    mass = np.cumsum(ordered)/ordered.sum()
    thresholds = [ordered[min(np.searchsorted(mass, level), len(ordered) - 1)] for level in credible_levels]   # density above which the credible mass is enclosed
    if len(set(thresholds)) < len(thresholds):
        return False
    levels = sorted(thresholds)
    ax.contour(0.5*(x_edges[1:] + x_edges[:-1]), 0.5*(y_edges[1:] + y_edges[:-1]), histogram.T, levels=levels, colors="black", linewidths=[1.8, 1.0][-len(levels):] if len(levels) == 2 else 1.5, zorder=10)
    return True

def draw_puffing_corner(parameters, scatter_layers, hist_layers, posterior, posterior_name, handles, info_lines, out_file):
    """
    Corner plot (matplotlib). Scatter layers are drawn in the order given, then the posterior samples (black) on top.

    Args:
        parameters (list): Parameters on the axes (keys of PUFFING_LABELS).
        scatter_layers (list): dicts with data (parameters dict), mask (bool array or None), color, size, alpha, zorder.
        hist_layers (list): dicts with data, mask (or None) and color, drawn as step histograms on the diagonal.
        posterior (dict): Posterior samples.
        posterior_name (str): File name of the posterior, for the legend.
        handles (list): Legend handles for the layers (the posterior is added).
        info_lines (list): Text written in the upper right corner of the figure.
        out_file (str): Path of the png.
    """
    from matplotlib.lines import Line2D
    n = len(parameters)
    rng = np.random.default_rng(1)
    def pick(layer, count=4000):
        index = np.arange(len(next(iter(layer["data"].values())))) if layer["mask"] is None else np.flatnonzero(layer["mask"])
        return np.sort(rng.choice(index, min(count, len(index)), replace=False)) if len(index) else index
    ipost = np.sort(rng.choice(len(posterior["m1"]), min(5000, len(posterior["m1"])), replace=False))
    limits = []
    for name in parameters:
        if name == "meanPerAno":
            limits.append((0., 2*np.pi)); continue
        values = np.concatenate([posterior[name]] + [layer["data"][name] for layer in scatter_layers])
        low, high = np.percentile(values, [0.1, 99.9]); pad = 0.05*(high - low) if high > low else 1.
        limits.append((low - pad, high + pad))
    picks = [pick(layer) for layer in scatter_layers]
    with plt.rc_context({"font.size": 12, "axes.labelsize": 15, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, axes = plt.subplots(n, n, figsize=(2.9*n + 1, 2.9*n + 1), squeeze=False)
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if j > i:
                    ax.axis("off"); continue
                if i == j:
                    name = parameters[i]; bins = np.linspace(limits[i][0], limits[i][1], 50)
                    for layer in hist_layers:
                        values = layer["data"][name] if layer["mask"] is None else layer["data"][name][layer["mask"]]
                        if len(values):
                            ax.hist(values, bins=bins, density=True, histtype="step", color=layer["color"], linewidth=1.5)
                    ax.hist(posterior[name], bins=bins, density=True, histtype="step", color="black", linewidth=1.5)
                    ax.set_xlim(*limits[i]); ax.set_yticks([])
                else:
                    xname, yname = parameters[j], parameters[i]
                    for layer, index in zip(scatter_layers, picks):
                        ax.scatter(layer["data"][xname][index], layer["data"][yname][index], s=layer["size"], color=layer["color"], alpha=layer["alpha"], linewidths=0, rasterized=True, zorder=layer["zorder"])
                    if not posterior_contours(ax, posterior[xname], posterior[yname], limits[j], limits[i]):   # too few samples for contours: show the samples
                        ax.scatter(posterior[xname][ipost], posterior[yname][ipost], s=2, c="black", alpha=0.7, linewidths=0, rasterized=True, zorder=10)
                    ax.set_xlim(*limits[j]); ax.set_ylim(*limits[i])
                if i == n - 1:
                    ax.set_xlabel(PUFFING_LABELS.get(parameters[j], parameters[j])); ax.tick_params(axis="x", labelrotation=45)
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(PUFFING_LABELS.get(parameters[i], parameters[i]))
                elif j > 0 and i != j:
                    ax.set_yticklabels([])
        handles = handles + [Line2D([], [], color="black", linewidth=1.5, label=f"latest posterior, 50% and 90% contours ({posterior_name})")]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.80), fontsize=13, frameon=True)
        fig.text(0.97, 0.97, "\n".join(info_lines), ha="right", va="top", fontsize=14)
        fig.savefig(out_file, bbox_inches="tight", dpi=75)
        plt.close(fig)

def plot_puffing_corners(folder, plot_title="Main", run_folder=None):
    """
    Corner plots of the puffed points (puffball-N.xml.gz, read with lalsimutils) and the latest posterior, saved in plots/Exploration/puffing and
    named like the other corner plots: corner_{parameters}_{plot_title}_Final.png is the latest puffball (high lnL puffs in red, frontier points in orange)
    and corner_{parameters}_{plot_title}.png shows the puffs of the iterations read, coloured by iteration. The posterior is drawn as contours. The html file shows
    them like the other corner plots. The numbers of puffs, the high lnL puffs and the checks of the correlation and of the puff efficiency are stored in
    run_diagnostics["puffing"][plot_title] for Diagnostics.txt.

    Args:
        folder (str): Path to the run directory (plot_title "Main") or to the subdag directory (plot_title "Subdag").
        plot_title (str): "Main" or "Subdag", used in the file names.
        run_folder (str, optional): Run directory, where the puffball .sub file is searched if it is not in folder (for the subdag).
    """
    print(f"\n--> Plotting puffing corner plots for {plot_title} iterations.")
    files = find_puffing_files(folder)
    if not files:
        print(f"No puffball-N.xml.gz files found in {folder}, skipping the puffing corner plots.")
        return
    arguments = read_puffball_arguments(folder)
    if not arguments["found"] and run_folder is not None:
        arguments = read_puffball_arguments(run_folder)
    if arguments["found"]:
        puff_factor, source = arguments["puff_factor"], arguments["file"]
    else:
        puff_factor, source = 2.0, "assumed, the puffball .sub file was not found"
    chosen = sorted(files)
    if len(chosen) > PUFFING_MAX_ITERATIONS:
        chosen = sorted(set(np.array(chosen)[np.unique(np.linspace(0, len(chosen) - 1, PUFFING_MAX_ITERATIONS).astype(int))]))
    all_net_file = f"{folder}/all.net"
    max_lnL = max_lnL_low_error(all_net_file)
    puffs, lnL_status, table = {}, {}, {}
    for n in chosen:
        try:
            puffs[n] = read_grid_parameters(files[n]["puffball"])
        except Exception as e:
            print(f"puffball-{n}.xml.gz could not be read ({e})")
            continue
        row = {"n_puff": int(len(puffs[n]["m1"])), "n_high": None, "hit_puff": np.nan, "hit_grid": np.nan, "efficiency": np.nan, "corr_grid": None, "corr_puff": None, "notes": []}
        lnL_status[n] = None
        composite_label = f"consolidated_{n}.composite"
        puff_lnL = find_lnL_of_points(puffs[n], files[n]["composite"], composite_label, all_net_file)
        if puff_lnL["lnL"] is not None and max_lnL is not None:
            evaluated = ~np.isnan(puff_lnL["lnL"])
            high = evaluated & (puff_lnL["lnL"] >= max_lnL - 15) & (puff_lnL["err"] <= 0.4)
            lnL_status[n] = {"evaluated": evaluated, "high": high, "matched": puff_lnL["matched_fraction"]}
            row["n_high"], row["hit_puff"] = int(high.sum()), high_lnL_fraction(puff_lnL["lnL"], puff_lnL["err"], max_lnL)
        if puff_lnL["note"]:
            row["notes"].append("puffs: " + puff_lnL["note"])
        if max_lnL is None:
            row["notes"].append("lnL not available: the maximum lnL could not be read from all.net")
        grid = None
        if files[n]["grid"] is None:
            row["notes"].append(f"overlap-grid-{n}.xml.gz not found, so the puff efficiency and the correlation cannot be computed")
        else:
            try:
                grid = read_grid_parameters(files[n]["grid"])
            except Exception as e:
                row["notes"].append(f"overlap-grid-{n}.xml.gz could not be read ({e})")
        if grid is not None:
            if lnL_status[n] is not None:     # efficiency: high lnL fraction of the puffs relative to the grid they were made from
                grid_lnL = find_lnL_of_points(grid, files[n]["composite"], composite_label, all_net_file)
                if grid_lnL["lnL"] is None:
                    row["notes"].append("puff efficiency not available, grid points: " + grid_lnL["note"])
                else:
                    row["hit_grid"] = high_lnL_fraction(grid_lnL["lnL"], grid_lnL["err"], max_lnL)
                    if row["hit_grid"] > 0:
                        row["efficiency"] = row["hit_puff"]/row["hit_grid"]
                    else:
                        row["notes"].append(f"puff efficiency not available: none of the {int(np.sum(~np.isnan(grid_lnL['lnL'])))} evaluated grid points is a high lnL point")
            if eccentricity:                  # correlation of eccentricity and meanPerAno (unwrapped about the grid's center) in the grid and in the puffs
                summary, grid_unwrapped = circular_summary(grid["meanPerAno"])
                puff_unwrapped = unwrap_angles(puffs[n]["meanPerAno"], summary["center"])
                row["corr_grid"] = float(np.corrcoef(grid["eccentricity"], grid_unwrapped)[0, 1])
                row["corr_puff"] = float(np.corrcoef(puffs[n]["eccentricity"], puff_unwrapped)[0, 1])
        table[n] = row
    if not puffs:
        return
    iterations = sorted(puffs)
    latest = iterations[-1]
    # checks on the latest iteration
    warnings = []
    last = table[latest]
    if last["corr_grid"] is not None and abs(last["corr_grid"]) > 0.5 and last["corr_puff"]/last["corr_grid"] < PUFFING_THRESHOLDS["min_corr_retention"]:
        warnings.append(f"Iteration {latest}: the correlation between eccentricity and meanPerAno (unwrapped) is {last['corr_grid']:+.2f} in the grid but {last['corr_puff']:+.2f} in the puffs: the puffs do not follow the posterior ridge.")
    if not np.isnan(last["efficiency"]) and last["efficiency"] < PUFFING_THRESHOLDS["min_hit_efficiency"]:
        warnings.append(f"Iteration {latest}: only {last['hit_puff']:.1%} of the puffs are high lnL points against {last['hit_grid']:.1%} of the grid points (efficiency {last['efficiency']:.2f}); most puffs are evaluated in regions of low likelihood.")
    run_diagnostics.setdefault("puffing", {})[plot_title] = {"puff_factor": puff_factor, "source": source, "table": table, "latest": latest, "warnings": warnings}

    posterior_file = find_latest_posterior(folder)
    posterior = read_posterior_parameters(posterior_file)
    posterior_name = os.path.basename(posterior_file)
    puff = puffs[latest]
    footprint = None
    try:
        footprint = puffing_footprint(puff, posterior)
    except Exception as e:
        print(f"Couldn't compute the footprint of the posterior ({e}), frontier points are not shown.")
    high = lnL_status[latest]["high"] if lnL_status[latest] else None
    outside = footprint["outside"] if footprint else None
    frontier = (high & outside) if (high is not None and outside is not None) else None
    from matplotlib.lines import Line2D
    dot = lambda color, label: Line2D([], [], marker="o", linestyle="", color=color, markersize=7, label=label)
    # layers of the latest iteration
    if high is not None:
        latest_layers = [{"data": puff, "mask": ~high, "color": "#9ecae1", "size": 3, "alpha": 0.45, "zorder": 1},
                         {"data": puff, "mask": high & ~(frontier if frontier is not None else np.zeros(len(high), dtype=bool)), "color": "#d62728", "size": 3, "alpha": 0.45, "zorder": 2}]
        latest_handles = [dot("#9ecae1", "puffs"), dot("#d62728", "high lnL puffs")]
        latest_hist = [{"data": puff, "mask": None, "color": "#3182bd"}, {"data": puff, "mask": high, "color": "#d62728"}]
        if frontier is not None:
            latest_layers.append({"data": puff, "mask": frontier, "color": "#ff9900", "size": 4, "alpha": 0.45, "zorder": 3}); latest_handles.append(dot("#ff9900", "frontier points"))
    else:
        mask_outside = outside if outside is not None else np.zeros(len(puff["m1"]), dtype=bool)
        latest_layers = [{"data": puff, "mask": ~mask_outside, "color": "#9ecae1", "size": 3, "alpha": 0.45, "zorder": 1}]
        latest_handles = [dot("#9ecae1", "puffs")]
        latest_hist = [{"data": puff, "mask": None, "color": "#3182bd"}]
        if outside is not None:
            latest_layers.append({"data": puff, "mask": mask_outside, "color": "#ff9900", "size": 4, "alpha": 0.45, "zorder": 3}); latest_handles.append(dot("#ff9900", "outside the posterior footprint (lnL unknown)"))
    latest_lines = [f"{plot_title}: puffed points of iteration {latest}, {len(puff['m1'])} points (puffball-{latest}.xml.gz)", f"Latest posterior: {posterior_name}, {len(posterior['m1'])} samples"]
    if high is not None:
        n_evaluated = int(lnL_status[latest]["evaluated"].sum())
        latest_lines += [f"High lnL puffs: {int(high.sum())} of {n_evaluated} evaluated ({100.*high.sum()/max(n_evaluated, 1):.1f}%)", "(lnL within 15 of the maximum, error <= 0.4)"]
        if not np.isnan(last["efficiency"]):
            latest_lines.append(f"Puff efficiency: {last['efficiency']:.2f} ({last['hit_puff']:.1%} of the puffs vs {last['hit_grid']:.1%} of the grid points are high lnL)")
        else:
            latest_lines.append("Puff efficiency: not available (see the Puffing diagnostics)")
    else:
        latest_lines += ["lnL of the puffs not available, so no high lnL puffs or puff efficiency", "(see the Puffing diagnostics)"]
    if frontier is not None:
        latest_lines += [f"Frontier points: {int(frontier.sum())}", "(high lnL puffs outside the footprint of the posterior samples)"]
    elif outside is not None:
        latest_lines += [f"Puffs outside the posterior footprint: {int(outside.sum())}", "(farther from the posterior samples than 99% of them are from each other)"]
    # layers of all the iterations read: coloured by iteration
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, max(len(iterations), 2)))
    all_layers = [{"data": puffs[n], "mask": None, "color": colors[k], "size": 3, "alpha": 0.4, "zorder": 1 + k} for k, n in enumerate(iterations)]
    all_hist = [{"data": puffs[n], "mask": None, "color": colors[k]} for k, n in enumerate(iterations)]
    all_handles = [dot(colors[k], f"puffball-{n}" + (f" ({int(lnL_status[n]['high'].sum())} high lnL of {int(lnL_status[n]['evaluated'].sum())})" if lnL_status[n] else "")) for k, n in enumerate(iterations)]
    all_lines = [f"{plot_title}: puffed points of iterations {', '.join(str(n) for n in iterations)}", f"Latest posterior: {posterior_name}, {len(posterior['m1'])} samples"]
    out = f"{save_path}/plots/{PUFFING_FOLDER}"
    os.makedirs(out, exist_ok=True)
    for parameters in puffing_parameter_sets():
        parameters = [name for name in parameters if name in posterior]
        print(f"Plotting puffing corner plots ({plot_title}) {parameters}")
        draw_puffing_corner(parameters, latest_layers, latest_hist, posterior, posterior_name, latest_handles, latest_lines, f"{out}/corner_{'_'.join(parameters)}_{plot_title}_Final.png")
        if len(iterations) > 1:
            draw_puffing_corner(parameters, all_layers, all_hist, posterior, posterior_name, all_handles, all_lines, f"{out}/corner_{'_'.join(parameters)}_{plot_title}.png")

def write_puffing_information(f):
    """Writes the puffing diagnostics (see plot_puffing_corners) of the main run and of the subdag to the open Diagnostics.txt file."""
    puffing = run_diagnostics.get("puffing", {})
    if not puffing:
        f.write("No puffball-N.xml.gz files found, the puffing was not checked.\n")
        return
    for title in ["Main", "Subdag"]:
        if title not in puffing:
            continue
        s = puffing[title]
        f.write(f"\n{title}:\n")
        f.write(f"Puff factor used for the expected widths = {s['puff_factor']:g} ({s['source']}); each puffed parameter should be about sqrt(1+f^2) = {np.sqrt(1. + s['puff_factor']**2):.2f} times wider than the grid.\n")
        f.write("\n\titeration | puffball points | high lnL puffs | puff efficiency\n")
        for n, row in s["table"].items():
            high_puffs = f"{row['n_high']} ({100*row['hit_puff']:.1f}%)" if row["n_high"] is not None else "NA"
            efficiency = f"{row['efficiency']:.2f}" if not np.isnan(row["efficiency"]) else "NA"
            f.write(f"\t{n:>9} | {row['n_puff']:>15} | {high_puffs:>14} | {efficiency:>15}\n")
        for n, row in s["table"].items():
            for note in row["notes"]:
                f.write(f"\t--> Iteration {n}: {note}\n")
        last = s["table"][s["latest"]]
        if last["corr_grid"] is not None:
            note = "" if abs(last["corr_grid"]) > 0.5 else " (no strong correlation in the grid, the correlation check does not apply)"
            f.write(f"\n\tcorr(eccentricity, meanPerAno) in iteration {s['latest']}: {last['corr_grid']:+.2f} (grid), {last['corr_puff']:+.2f} (puffs){note}\n")
        for warning in s["warnings"]:
            f.write(f"\t--> {warning}\n")

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

        write_header(f, "Puffing diagnostics")
        write_puffing_information(f)

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
    manifest = {"histograms": [], "corners": [], "puffing": [], "figures": {}}

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

    # puffing corner plots: Exploration/puffing/corner_{parameters}_{plot_title}.png, named like the corner plots above
    for file_path in glob.glob(plots + "/" + PUFFING_FOLDER + "/corner_*.png"):
        if not is_current(file_path):
            continue
        name = os.path.basename(file_path)[len("corner_"):-len(".png")]
        for title in ["Main_Final", "Subdag_Final", "Main", "Subdag"]:
            if name.endswith("_" + title):
                parameters = name[:-(len(title)+1)]
                manifest["puffing"].append({"run": title.split("_")[0], "latest": title.endswith("_Final"), "key": split_parameter_string(parameters), "file": os.path.relpath(file_path, plots)})
                break
    puffing_order = {", ".join(parameters): i for i, parameters in enumerate(puffing_parameter_sets())}   # cartesian parameters first
    manifest["puffing"].sort(key = lambda e: (run_order[e["run"]], e["latest"], puffing_order.get(e["key"], len(puffing_order)), e["key"]))

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
    entries = list(manifest["histograms"]) + list(manifest["corners"]) + list(manifest["puffing"])
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
  <nav><a href="#ile">ILE</a><a href="#cip">CIP</a><a href="#puffing">Puffing</a><a href="#visual">Visual</a></nav>
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
  <h3 id="puffing">Puffing health</h3>
  <p class="desc">Puffing drives exploration during a run. Visually check whether the puffed points follow the same correlations as the posterior and form a slightly broadened version of it. If puffing is not working, there will be no exploration.</p>
  <pre>__PUFFING_TEXT__</pre>
  <div id="puffing-gallery"></div>
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
if (MANIFEST.puffing.length) makeGallery("puffing-gallery", MANIFEST.puffing, "parameter sets", 560, true);
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
    puffing_text = sections.get("Puffing diagnostics", "No puffing diagnostics available.")
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
        "__PUFFING_TEXT__": diagnostics_text_to_html(puffing_text),
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

# puffing corner plots (puffball-N.xml.gz) of the main run, saved in plots/Exploration/puffing and shown last in the CIP section of the html
try:
    plot_puffing_corners(path, plot_title='Main')
except Exception as e:
    print(e)
    print("Couldn't plot the puffing corner plots for Main iterations.")

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

    # puffing corner plots (puffball-N.xml.gz) of the subdag
    try:
        plot_puffing_corners(subdag_path, plot_title='Subdag', run_folder=path)
    except Exception as e:
        print(e)
        print("Couldn't plot the puffing corner plots for Subdag iterations.")

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
