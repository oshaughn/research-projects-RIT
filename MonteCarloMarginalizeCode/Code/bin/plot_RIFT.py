#!/usr/bin/env python
"""This code is meant to check the health of a RIFT run as it progresses and after it has finished."""
# TO DO: plot_cip_max_lnL needs to be robust.
# Usage: In the rundirectory, run plot_RIFT.py --precessing or plot_RIFT.py --precessing --eccentric or plot_RIFT.py --eccentric --LISA 
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
parser.add_argument("--path", default = os.getcwd(), help =  "path to run directory")
parser.add_argument("--LISA", action = "store_true", help = "Use this argument if analyzing a LISA run")
parser.add_argument("--eccentric", action = "store_true", help = "Use this argument if the run has eccentricity and meanPerAno")
parser.add_argument("--precessing", action = "store_true", help = "Use this argument if the run is precessing")
parser.add_argument("--non-spinning", action = "store_true", help = "Use this argument if the run is non-spinning")
parser.add_argument("--truth-file", default = None, type = str,  help = "Path to the truth file. If not provided, the code will search for a truth file in the parent directory or the frames directory. The frames directory is expected to be located in the parent directory.")
opts = parser.parse_args()
path = opts.path
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


###########################################################################################
# Functions
###########################################################################################
def get_lnL_cut_points(all_net_path, lnL_cut=15, error_threshold=0.4, composite=False, provide_max_lnL_point=False):
    """
    Analyzes the lnL values from an all.net file to find high likelihood points
    and assess their Monte Carlo error.

    Args:
        all_net_path (str): Path to the all.net file.
        lnL_cut (float): Cutoff for determining high likelihood points.
        error_threshold (float): Maximum allowed error for high likelihood points.
        composite (bool): is the file a composite file or an all.net file.

    Returns:
        tuple: Maximum lnL value (rounded) and the number of high likelihood points with low error.
    """
    # Load data from all.net file
    data = np.loadtxt(all_net_path)
    
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
    
    # Remove NaN values from lnL
    total_points = len(lnL)
    lnL = lnL[~np.isnan(lnL)]

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
    
    # Update diagnostics with results
    max_lnL = np.max(lnL)
    no_points = len(lnL[lnL >= (max_lnL - lnL_cut)])
    if composite:
        max_lnL_composite = max_lnL
        max_lnL = run_diagnostics["max_lnL"]
        return np.round(max_lnL, 3), no_points, np.round(max_lnL_composite, 2), total_points
    if not(composite):
        run_diagnostics.update({
            "total_lnL_evaluations":total_points,
            "max_lnL": np.round(max_lnL, 3),
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
    specified credible interval, assuming a multivariate Gaussian posterior.

    Args:
        ci (float): Credible interval enclosed by the likelihood contour
            (e.g., 0.90, 0.95, 0.99).
        run_diagnostics (dict): Dictionary containing run diagnostics,
            including the maximum log-likelihood stored under the key
            ``"max_lnL"``.

    Returns:
            - Log-likelihood at the credible interval boundary (float)
    """
    if not (0 < ci < 1):
        raise ValueError("ci must be between 0 and 1, e.g. 0.99")

    ndim = 4
    if opts.non_spinning:
        ndim = 2
    elif opts.precessing:
        ndim += 4
    elif opts.eccentric:
        ndim += 2
    if return_ndim:
        return ndim
    max_lnL = run_diagnostics["max_lnL"]

    r2 = chi2.ppf(ci, df=ndim)
    delta_lnL = -0.5 * r2
    lnL = max_lnL + delta_lnL

    return lnL

def create_plots_folder(base_dir_path):
    """
    Creates a 'plots' folder in the specified base directory if it does not already exist.

    Args:
        base_dir_path (str): Path to the base directory where the 'plots' folder will be created.
    """
    if not(os.path.exists(base_dir_path + "/plots")):
        print(f"--> plots folder does not exist. Creating one in {base_dir_path}")
        os.mkdir(base_dir_path + "/plots")
        os.mkdir(base_dir_path + "/plots/histograms")
        os.mkdir(base_dir_path + "/plots/corner_plots")
        os.mkdir(base_dir_path + "/plots/1_D_plots")
    else:
        print(f"--> plots folder exists, saving plots in directory {base_dir_path}/plots")

def get_chirpmass_massratio_eta_totalmass_from_componentmasses(m1, m2):
    """
    Computes chirp mass, mass ratio, symmetric mass ratio, and total mass from component masses.

    Args:
        m1 (array): Array of primary masses.
        m2 (array): Array of secondary masses.

    Returns:
        tuple: A tuple containing:
            - Chirp mass (array)
            - Mass ratio (array)
            - Symmetric mass ratio (array)
            - Total mass (array)
    """
    return np.array((m1*m2)**(3/5) / (m1+m2)**(1/5)).reshape(-1,1), np.array(m2/m1).reshape(-1,1), np.array((m1*m2) / (m1+m2)**(2)).reshape(-1,1), np.array(m1+m2).reshape(-1,1)

def get_index_for_parameter(parameter):
    """
    Retrieves the index corresponding to a given parameter name.

    Args:
        parameter (str): The name of the parameter.

    Returns:
        int or None: The index of the parameter if found, otherwise None.
    """
    # m1 m2 a1x a1y a1z a2x a2y a2z mc eta indx  Npts ra dec tref phiorb incl psi  dist p ps lnL mtotal q  eccentricity meanPerAno 
    parameter_indices = {
        "mc": 8,
        "mtot": 22,
        "a1x":2,
        "s1x":2,
        "a1y":3,
        "s1y":3,
        "a1z": 4,
        "s1z": 4,
        "a2x":5,
        "s2x":5,
        "a2y":6,
        "s2y":6,
        "a2z": 7,
        "s2z": 7,
        "eta": 9,
        "m1": 0,
        "m2": 1,
        "q": 23,
        "dec": 13,
        "ra": 12,
        "eccentricity":24,
        "meanPerAno":25,
    }
    
    return parameter_indices.get(parameter, None)  # Return None if parameter is not found

def get_sample_statistics(samples):
    """
    Computes statistical percentiles (5th, 50th, and 95th) for a given sample set.
    Args:
        samples (array-like): A list or NumPy array of samples.

    Returns:
        sample statistics (numpy.ndarray): An array containing the 16th, 50th, and 84th percentiles 
        of the input samples.
    """
    return np.percentile(samples, [5,50,95])

def get_combination_from_mass_and_spin(posterior, parameter):
    """
    Computes spin combinations from posterior samples.

    Args:
        posterior (numpy.ndarray): Posterior samples.
        parameter (str): One of
            'chi1', 'chi2', 'chi_eff', 'chi_minus', 'chi_p'

    Returns:
        numpy.ndarray: Requested spin combination.
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

def find_posteriors_in_main(path_to_main_folder, limit_iterations=None):
    """
    Finds and sorts posterior sample files in the main folder.

    Args:
        path_to_main_folder (str): Path to the main folder containing posterior sample files.
        limit_iterations (int, optional): Number of files to limit the results to.

    Returns:
        posteriors (list of str): Sorted list of paths to posterior sample files.
        indices (numpy array): Indices of the selected files.
    """
    posteriors_in_main = glob.glob(path_to_main_folder + "/posterior_samples*")
    posteriors_in_main.sort(key = os.path.getctime) # sort them according to creation time
    if limit_iterations:
        index = np.linspace(0, len(posteriors_in_main)-1, limit_iterations)
        index = np.array(index, dtype=int)
        return np.array(posteriors_in_main, dtype = str)[index], index + 1
    return posteriors_in_main, np.arange(len(posteriors_in_main)) + 1

def find_posteriors_in_sub(path_to_main_folder, limit_iterations = None):
    """
    Finds posterior sample files in the sub-directory specified by the path.

    Args:
        path_to_main_folder (str): Path to the main folder containing sub-directory with posterior files.
        limit_iterations (int, optional): Number of files to limit the results to.

    Returns:
        posteriors (list of str): List of paths to posterior sample files in the sub-directory.
        indices (numpy array): Indices of the selected files.
    """
    posteriors_in_subdag, iterations = find_posteriors_in_main(path_to_main_folder + "/iteration*cip*")
    if limit_iterations:
        index = np.linspace(0, len(posteriors_in_subdag)-1, limit_iterations)
        index = np.array(index, dtype=int)
        return np.array(posteriors_in_subdag, dtype = str)[index], index + 1
    else:
        return posteriors_in_subdag, np.arange(len(posteriors_in_subdag))

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

def plot_high_likelihood_expoloration(path_to_main_folder):
    """
    Plots high likelihood points over iterations.

    Args:
        path_to_main_folder (str): Path to the main folder containing the composite files.
    """
    print("\n--> Plotting likelihood exploration.")
    run_diagnostics["composite_information"] = {}
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("high lnL points")
    ax.set_title(f"Total high lnL points = {run_diagnostics['high_lnL_points']}, max_lnL = {run_diagnostics['max_lnL']}")
    collect_data = []
    collect_iter = []
    print("iteration, max lnL (global), high lnL points, max lnL(iteration), total lnL points(iteration)")
    for iteration in np.arange(0, run_diagnostics["latest_iteration"]+1, 1):
        run_diagnostics["composite_information"][iteration] = {}
        try:
            max_lnL, no_points, max_lnL_composite, total_points = get_lnL_cut_points(f"{path_to_main_folder}/consolidated_{iteration}.composite", composite=True)
        except Exception as e:
            print(f"Error loading file {path_to_main_folder}/consolidated_{iteration}.composite: {e}")
            continue
        print(iteration, max_lnL, no_points, max_lnL_composite, total_points)
        percent_high_lnL_points =  np.round(no_points/total_points*100, 2)
        collect_data.append(no_points)
        collect_iter.append(iteration)
        ax.scatter(iteration, no_points, label = f"{max_lnL_composite} ({percent_high_lnL_points})", s=25)
        run_diagnostics["composite_information"][iteration].update({
                "max_lnL":max_lnL_composite,
                "high_lnL_points":no_points,
                "percent_high_lnL_points": percent_high_lnL_points})
    ax.grid(alpha=0.4)
    ax.plot(collect_iter, collect_data, color = "black", linestyle = "--", linewidth = 1.5, alpha = 0.5)
    ax.set_xticks(np.arange(0, run_diagnostics["latest_iteration"]+1, 1))
    ax.legend(loc="upper left")
    fig.savefig(path+f"/plots/Likelihood_exploration_plot.png", bbox_inches='tight')
    plt.close(fig)

def plot_neff_data(path_to_main_folder):
    """
    Plot effective number of samples (neff) data from CIP iterations.
    Args:
        path_to_main_folder (str): Path to the main folder containing CIP iteration subfolders.
    """
    print("\n--> Plotting n-eff for CIP.")
    # find CIP folders
    cip_iteration_folders= glob.glob(path_to_main_folder + "/iteration*cip*")
    
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("neff")
    iterations=np.arange(len(cip_iteration_folders)-1) # last folders don't usually have anything
    
    # read requested neff from CIP sub files
    run_diagnostics["CIP_neff"] = {}
    for i in range(3):
        filename = f"{path}/CIP_worker{i}.sub"
        try:
            with open(filename, "r") as f:
                content = f.read()
                matches = re.findall(r"--n-eff\s+([+-]?\d+(?:\.\d+)?)", content)
                if matches:
                    neff_value = float(matches[-1])
                    ax.axhline(
                        y=neff_value,
                        linestyle="--",
                        color=default_colors[i],
                        alpha=1.0,
                        linewidth=1.0,
                        label=f"worker {i} neff"
                    )
                    run_diagnostics["CIP_neff"][f"CIP_worker{i}"] = np.round(neff_value, 2)
                else:
                    continue
        except Exception as e:
            pass
    ax.legend(loc="upper left")
    
    # read neff achived for each iteration from each instance of CIP
    run_diagnostics["CIP_neff_achieved"] = {}
    for n in iterations:
        i = path_to_main_folder + f"/iteration_{n}_cip"
        # remove existing data file because I append
        os.system(f"rm {i}/neff_data.txt 2> /dev/null")
        # read neff data from each file and store it in neff_data.txt
        cmd=f"for i in {i}/overlap-grid-*-*ESS* ; do cat $i | tail -n 1 >> {i}/neff_data.txt; done 2> /dev/null"
        os.system(cmd) 
        # calculate neff statistics
        try:
            tmp_ESS_data=np.loadtxt(f"{i}/neff_data.txt", usecols=[2])
            low, avg, high = np.percentile(tmp_ESS_data, [2.5,50,97.5])
            low_1_std, avg, high_1_std = np.percentile(tmp_ESS_data, [16,50,84])
            mini, maxi = np.min(tmp_ESS_data), np.max(tmp_ESS_data)
            ax.plot(iterations[n], mini, marker="x", color="black")
            ax.plot(iterations[n], maxi, marker="x", color="black")
            print(f"neff detail iteration = {iterations[n]}: Average={avg:0.2f}, low std={low:0.2f}, high std={high:0.2f}")
            ax.errorbar(iterations[n], avg, yerr=np.array([avg-low,high-avg]).reshape(-1, 1), color = "royalblue", ecolor = "red", fmt ='o')
            ax.errorbar(iterations[n], avg, yerr=np.array([avg-low_1_std,high_1_std-avg]).reshape(-1, 1), color = "royalblue", ecolor = "green", fmt ='.')
            run_diagnostics["CIP_neff_achieved"][f"iteration_{n}_neff"] = np.round(avg, 2)
            iteration_prog = n
        except Exception as e:
            print(f"Couldn't plot neff for iteration = {iterations[n]}")
            break
    # read max lnL data from CIP output files
    lnL_files_last_iteration = glob.glob(path_to_main_folder + f"/iteration_{iterations[iteration_prog]}_cip/*lnL*")
    run_diagnostics["latest_grid"] = f"overlap-grid-{iteration_prog+1}.xml.gz"
    run_diagnostics["latest_iteration"] = int(iteration_prog)
    max_lnL, no_points = get_lnL_cut_points(all_net_path)
    print(f"Total number of workers in final iteration = {len(lnL_files_last_iteration)-1}")
    ax.set_title(f"Workers = {len(lnL_files_last_iteration)-1}")
    ax.set_xticks(np.arange(0, run_diagnostics["latest_iteration"]+1, 1))
    fig.savefig(path+f"/plots/Effective_samples_per_CIPworker.png", bbox_inches='tight')
    plt.close(fig)


def plot_cip_max_lnL(path_to_main_folder):
    """
    Plot the maximum log-likelihood (lnL) values sampled from different iterations.

    This function iterates over all available iterations, collects maximum lnL values from files in each iteration's directory,
    calculates the mean and standard deviation (using percentiles) of these values, and plots them with error bars. It also 
    adds a horizontal line indicating the maximum lnL value in all.net.

    Args:
        path_to_main_folder (str): The path to the main folder containing iteration subfolders with lnL data files.

    """
    print("\n--> Plotting sampled lnL by CIP")
    iterations = np.arange(0, run_diagnostics["latest_iteration"]+1, 1)
    run_diagnostics['cip_sampled_lnL'] = {}
    fig, ax = plt.subplots()
    fig_hist, ax_hist = plt.subplots()
    for iteration in iterations:
        run_diagnostics['cip_sampled_lnL'][iteration] = {}
        try:
            files_iteration = glob.glob(path_to_main_folder + f"/iteration_{iteration}_cip/*lnL*")
        except:
            continue
        collect_lnL = []
        collec_lnL_hist = np.zeros((1,1))
        samples_total = 0
        for j in np.arange(len(files_iteration)):
            data = np.loadtxt(files_iteration[j])
            collect_lnL.append(np.max(data))
            samples_total += len(data)
            collec_lnL_hist = np.vstack([collec_lnL_hist, data[:, None]])
        samples_total_per_worker = samples_total/len(files_iteration)
        collec_lnL_hist = np.delete(collec_lnL_hist, 0, 0)
        collect_lnL = np.array(collect_lnL)
        low_1_std, max_lnL_avg_this_iteration, high_1_std  = np.percentile(collect_lnL, [16,50,84])
        low_2_std, max_lnL_avg_this_iteration, high_2_std  = np.percentile(collect_lnL, [2.5,50,97.5])
        run_diagnostics['cip_sampled_lnL'][iteration].update({
            'avg':max_lnL_avg_this_iteration,
            '+':high_1_std,
            '-':low_1_std})
        
        ax.errorbar(iteration, max_lnL_avg_this_iteration, yerr = np.array([max_lnL_avg_this_iteration-low_2_std, high_2_std-max_lnL_avg_this_iteration]).reshape(-1,1), color = "royalblue", ecolor = "red", fmt ='.')
        ax.errorbar(iteration, max_lnL_avg_this_iteration, yerr = np.array([max_lnL_avg_this_iteration-low_1_std, high_1_std-max_lnL_avg_this_iteration]).reshape(-1,1), color = "royalblue", ecolor = "green", fmt ='o')
        ax_hist.hist(collec_lnL_hist, label=iteration, histtype='step', linewidth = 1.0, bins=30, density=True)
    run_diagnostics["cip_average_max_lnL_sampled"] = np.round(np.mean(collect_lnL), 2)
    run_diagnostics["cip_std_max_lnL_sampled"] = np.round(np.std(collect_lnL), 3)
    ndim = lnL_at_credible_interval(return_ndim=True)
    ax.set_xlabel('iteration')
    ax.set_ylabel('lnL')
    ax.axhline(y = run_diagnostics['max_lnL'], linestyle = "--", color="black")
    expected_median = lnL_at_credible_interval(ci=0.5)
    ax.fill_between(iterations, expected_median-1.5, run_diagnostics['max_lnL'], color="green", alpha=0.5)
    ax.set_xticks(iterations)
    fig.savefig(path+f"/plots/Maximum_sampled_lnL_per_CIPworker.png", bbox_inches="tight")

    ax_hist.set_xlabel('lnL')
    ax_hist.legend(loc='upper left', frameon=False)
    ax_hist.axvline(x = run_diagnostics['max_lnL'], linestyle = "--", color="black")
    fig_hist.savefig(path+f"/plots/CIP_lnL_distribution_per_iteration", bbox_inches="tight")
    plt.close()

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
    print("\n--> Plotting histograms")
    # when you just want to plot final iterations histograms
    if iterations is None: 
        iterations = [-1]
        plot_legend = False
    if use_truths:
        P = lsu.xml_to_ChooseWaveformParams_array(truth_file_path)[0]
    # all_net_data = convert_all_net_to_posterior_format(all_net_path)
    # not_nan_lnL = np.argwhere(all_net_data[:,-3]>=np.max(all_net_data[:,-3]) - 15).flatten()#np.argwhere(~np.isnan(all_net_data[:,-3])).flatten()
    # all_net_data = np.array(all_net_data[not_nan_lnL])
    parameters =  ["mc", "q", "eta", "m1", "m2", "mtot","s1z", "s2z", "chi_eff","chiMinus","chi1", "chi2"]
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
        for i in np.arange(len(sorted_posterior_file_paths)):
            line_label = str(iterations[i])
            if parameter in ["chi_eff", "chiMinus", "chi_p", "chi1", "chi2", "chi1_perp", "chi2_perp"]:
                posterior_data = np.loadtxt(sorted_posterior_file_paths[i])
                #data = get_chi_eff_from_mass_and_spins(posterior_data)
                data = get_combination_from_mass_and_spin(posterior_data, parameter)
            else:
                parameter_index = get_index_for_parameter(parameter)
                data = np.loadtxt(sorted_posterior_file_paths[i])[:,parameter_index]
            if i > 0 and JSD:
                JS_test = calculate_JS_divergence(data, data_previous)
                line_label +=f" ({calculate_JS_divergence(data, data_previous).median:0.3f})"
            ax.hist(data, label = line_label, histtype="step", bins = 25, density=True, linewidth=1.0, color=default_colors[i])
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
                ax.axvline(x = P.extract_param(parameter_extract)/factor, linestyle="--", linewidth=1.0, color="black")
            data_previous = data
        #try: (this isn't really helpful, so commenting it out)
        #    likelihood = np.exp(np.array(all_net_data[:,-3]))
        #    reweighted_all_net = np.random.choice(all_net_data[:,parameter_index], p = likelihood /np.sum(likelihood), size = 1000, replace = True)
        #    ax.hist(reweighted_all_net, label = "Likelihood", histtype="step", bins = 50, density=True, alpha = 0.7, linewidth=1.0, color = "grey")
        #except:
        #    print("Couldn't plot likelihood distribution")
        # don't create legend when only plotting finals iteration's histograms
        if plot_legend: 
            ax.legend(loc = "upper right")
        fig.savefig(path+f"/plots/histograms/histogram_{plot_title}_{parameter}.png", bbox_inches='tight')
        plt.close()

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
    plt.savefig(path+f"/plots/lnL_distribution_from_extrinsic_step.png", bbox_inches='tight')
    plt.cla()
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
    max_lnL, no_points = run_diagnostics["max_lnL"], run_diagnostics["high_lnL_points"]  
    title = f"max_lnL={max_lnL:0.2f},points_cut={no_points}" 
    plotting_command = f"python {corner_plot_exe} --plot-1d-extra --quantiles None --ci-list [0.9] --use-title {title} "
    if plot_title != "extrinsic":
        plotting_command += f"--composite-file {all_net_path} --lnL-cut 15 --sigma-cut 0.4 "
     # Append iteration-related options to the command
    if iterations is not None:
        plotting_command += "--use-legend "
    else:
        iterations = [0]

    # Include truth file if required
    if use_truths:
        plotting_command += f"--truth-file {truth_file_path} "
    
    # plot grey points (low lnL) when showing multiple iterations
    if plot_title != "Final":
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
    os.system(f"mv {corner_plot_filename} {new_corner_plot_path}")

    # Move and rename individual parameter plots
    for parameter in parameters:
        os.system(f"mv {parameter}.png plots/1_D_plots/{parameter}_{plot_title}.png")
        os.system(f"mv {parameter}_cum.png plots/1_D_plots/{parameter}_cum_{plot_title}.png") 

def plot_JS_divergence(posterior_1_path, posterior_2_path, posterior_3_path=None, plot_title=None, threshold=0.005, parameters = ["mc","eta", "m1", "m2"]):
    """
    Plots Jensen-Shannon Divergence (JSD) between two posterior datasets for specified parameters.

    Args:
        posterior_1_path (str): File path to the first posterior dataset.
        posterior_2_path (str): File path to the second posterior dataset.
        plot_title (str): Title for the plot and filename.
        parameters (list of str): List of parameters to calculate and plot JSD for.
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
    ax.errorbar(parameters, JSD_array, np.array(JSD_error).T,  color = "royalblue", ecolor = "red", fmt ='o', markersize = 5, label='latest-secondlatest')
    if not(posterior_3_path is None):
        ax.errorbar(parameters, JSD_array_third, np.array(JSD_error_third).T,  color = "green", ecolor = "black", fmt ='o', markersize = 5, label='latest-thirdlatest')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', labelrotation=60)
    fig.savefig(path+f"/plots/JSD_per_parameter_{plot_title}.png", bbox_inches='tight')
    plt.close(fig)

def write_sample_statistics(posterior, parameters=["mc","eta", "m1", "m2"]):
    """
    Computes and writes sample statistics for specified parameters to a file.
    Args:
        posterior (str): Path to the file containing posterior samples.
        parameters (list, optional): List of parameter names for which
            statistics will be computed. Defaults to
            ["mc", "eta", "m1", "m2", "s1z", "s2z", "chi_eff"].
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
    f = open(path+f"/plots/sample_statistics.txt", "w")
    f.write("Note: limits are equal-tailed 90th percentile\n")
    run_diagnostics["sample_statistics"] = {}
    for parameter in parameters:
        if parameter in ["chi_eff", "chiMinus", "chi_p", "chi1", "chi2", "chi1_perp", "chi2_perp"]: 
            samples_here = get_combination_from_mass_and_spin(posterior, parameter) 
        else:
            parameter_n = get_index_for_parameter(parameter)
            samples_here = posterior[:,parameter_n]
        statistics = get_sample_statistics(samples_here)
        run_diagnostics["sample_statistics"][parameter] = np.round(statistics,3)
        line = f"{parameter}: median = {statistics[1]:0.3f}, upper limit = {statistics[2]:0.3f}, lower limit = {statistics[0]:0.3f}"
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
            line += f", truth here = {P.extract_param(parameter_extract)/factor:0.3f}"
        f.write(line + "\n")
    max_sample, lnL = get_lnL_cut_points(all_net_path, lnL_cut=15, error_threshold=0.4, composite=False, provide_max_lnL_point=True)
    f.close()

def plot_exploration_corner(all_net_path):
    """
    Generates and saves a corner plot for all the points at which marginalized likelihood was evaluated, effectively acting as the exploration plot.

    Args:
        all_net_path (str): File path to all.net
    """
    print('\n--> Plotting exploration corner')
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
        labels=[r"$m_1$ $(\times 10^6 M_\odot)$", r"$m_2$ $(\times 10^6 M_\odot)$", r"$\chi_{1z}$", r"$\chi_{2z}$", r"$\lambda$", r"$\beta$", r'$e$', '$\ell$']
        if use_truths:
            truths.append([P.extract_param('lambda'),  P.extract_param('beta'), P.extract_param('eccentricity'), P.extract_param('meanPerAno')])
    if not(LISA) and eccentricity:
        use_cols.append([9,10])
        labels=[r"$m_1$", r"$m_2$", r"$\chi_{1z}$", r"$\chi_{2z}$",  r'$e$', '$\ell$']
        if use_truths:
            truths.append([P.extract_param('eccentricity'), P.extract_param('meanPerAno')])
        if precessing:
            use_cols.append([3,4,6,7])
            labels=[r"$m_1$", r"$m_2$", r"$\chi_{1z}$", r"$\chi_{2z}$", r'$e$', '$\ell$', r"$\chi_{2z}$", r"$\chi_{1x}$", r"$\chi_{1y}$", r"$\chi_{2x}$", r"$\chi_{2y}$"]
            if use_truths:
                truths.append([P.extract_param('s1x'), P.extract_param('s1y'), P.extract_param('s2x'), P.extract_param('s2y')])
    if not(LISA) and precessing and not(eccentricity):
        use_cols.append([3,4,6,7])
        labels=[r"$m_1$", r"$m_2$", r"$\chi_{1z}$", r"$\chi_{2z}$", r"$\chi_{2z}$", r"$\chi_{1x}$", r"$\chi_{1y}$", r"$\chi_{2x}$", r"$\chi_{2y}$"]
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
    fig.savefig(f'plots/Exploration_corner_plot.png')


def evaluate_run(run_diagnostics):
    """
    Evaluates and writes diagnostics information to a file.

    Args:
        run_diagnostics (dict): Dictionary containing diagnostics data.
    """
    f = open(path+f"/plots/Diagnostics.txt", "w")
    # ILE
    f.write("###########################################################################################\n")
    f.write("# ILE diagnostics\n")
    f.write("###########################################################################################\n")
    # Monte carlo error and number of high lnL points
    f.write(f"Total number of marginalized lnL evaluations = {run_diagnostics['total_lnL_evaluations']}\n")
    f.write(f"Total number of high marginalized lnL points = {run_diagnostics['total_high_lnL_points']}\n")
    f.write(f"Total number of high marginalized lnL points used = {run_diagnostics['high_lnL_points']}\n")
    f.write(f"Total number of high marginalized lnL points not used due to large error = {run_diagnostics['high_lnL_points_with_large_error']}\n")
    f.write(f"Approximate SNR captured = {np.sqrt(2*run_diagnostics['max_lnL']):0.2f}\n")
    f.write(f"Number of high lnL points with lnL cut [12, 10, 5, 2] = [{run_diagnostics['high_lnL_points_with_lnLcut_12'], run_diagnostics['high_lnL_points_with_lnLcut_10'], run_diagnostics['high_lnL_points_with_lnLcut_5'], run_diagnostics['high_lnL_points_with_lnLcut_2']}]\n")
    print(run_diagnostics['composite_information'])
    f.write(f"Likelihood exploration data per iteration: \n{run_diagnostics['composite_information']}\n")
    
    ILE_is_good = True
    if run_diagnostics['high_lnL_points_with_large_error']/run_diagnostics['total_high_lnL_points'] > 0.5:
        f.write(f"\t--> Large number of points have a high Monte Carlo error (sigma = 0.4). Consider reducing d-max, increasing d-min, increasing n-max and/or changing the sampler.\n")
        ILE_is_good = False
    if run_diagnostics['high_lnL_points'] <= 500:
        f.write(f"\t--> Number of high likelihood points is less than 500, which could be caused due to initial grid not having sufficient resolution. Considering reducing the parameter space and/or increasing the number of points on the grid.\n")
        ILE_is_good = False
    if 500 < run_diagnostics['high_lnL_points'] < 5000:
        f.write(f"\t--> Number of high likelihood points is less than 5000, consider rerunning with {run_diagnostics['latest_grid']} as your starting grid and copying this run's all.net as bonus.composite in your new run directory.\n")
        ILE_is_good = False
    f.write("\n")
    if ILE_is_good:
        f.write("\t--> ILE status: GOOD! <--\n")
    else:
        f.write("\t--> ILE status: BAD! <--\n")
    f.write("\n\n")
    # CIP
    f.write("###########################################################################################\n")
    f.write("# CIP diagnostics\n")
    f.write("###########################################################################################\n")
    # CIP neff
    f.write(f"CIP neff requested = {run_diagnostics['CIP_neff']}]\n")
    f.write(f"CIP neff achieved = {run_diagnostics['CIP_neff_achieved']}\n")
    CIP_is_good = True
    first_iter_neff = run_diagnostics['CIP_neff'][list(run_diagnostics['CIP_neff'].keys())[0]]
    last_iter_neff = run_diagnostics['CIP_neff'][list(run_diagnostics['CIP_neff'].keys())[-1]]
    last_iter_neff_achieved = run_diagnostics['CIP_neff_achieved'][list(run_diagnostics['CIP_neff_achieved'].keys())[-1]]
    if first_iter_neff <= last_iter_neff_achieved <= last_iter_neff and first_iter_neff!=last_iter_neff:
        f.write(f"\t--> neff has not been reached, the posterior distribution may be wider and/or less smooth. To address this, try narrowing the parameter space or switching to a different sampler. Alternatively, you can reduce the neff for each CIP job (>10) and increase the number of CIP jobs submitted per iteration. However, it seems like it has reached neff as set by the CIP for earlier iterations (CIP_worker0.sub), so if the run is ongoing let it continue.\n")
    elif last_iter_neff > last_iter_neff_achieved:
        f.write(f"\t--> neff has not been reached, the posterior distribution may be wider and/or less smooth. To address this, try narrowing the parameter space or switching to a different sampler. Alternatively, you can reduce the neff for each CIP job (>10) and increase the number of CIP jobs submitted per iteration.\n")
        CIP_is_good = False
    # CIP JSD
    f.write(f"\nCIP Jensen-Shannon divergence:\n{run_diagnostics['JSD']}\n")
    JSD_not_good = {}
    JSD_is_good = True
    for iteration_type in run_diagnostics['JSD']:
        JSD_not_good[iteration_type] = {}
        for param in run_diagnostics['JSD'][iteration_type]:
            if run_diagnostics['JSD'][iteration_type][param] > 0.005:
                JSD_not_good[iteration_type][param] = run_diagnostics['JSD'][iteration_type][param]
    for iteration_type in run_diagnostics['JSD']:
        if len(JSD_not_good[iteration_type]) > 0:
            JSD_is_good = False
    if JSD_is_good is False:
        f.write(f"\n\t   If the Jensen-Shannon Divergence for any parameter between the last and second-to-last iterations is greater than 0.005, it means the run has not yet converged. In this case, you should rerun the analysis using {run_diagnostics['latest_grid']} as the starting grid. Additionally, copy this run's all.net file to a new file named bonus.composite in your new run directory. \n") 
        CIP_is_good = False
    # CIP sampling
    f.write(f"\nAverage max lnL sampled by CIP in iteration {run_diagnostics['latest_iteration']} is: {run_diagnostics['cip_average_max_lnL_sampled']} +- {run_diagnostics['cip_std_max_lnL_sampled']}. Max lnL in all.net is {run_diagnostics['max_lnL']}.\n")
    expected_median = lnL_at_credible_interval(ci=0.5)
    if  expected_median - run_diagnostics['cip_average_max_lnL_sampled'] > 1.5: 
        f.write(f"\t--> The difference between the expected max lnL {expected_median} and the average maximum lnL value sampled by CIP {run_diagnostics['cip_average_max_lnL_sampled']} is more than 1.5, which could cause the peak to be slightly shifted. This discrepancy might be because CIP hasn't sampled the peak well enough or due to interpolation errors. If the issue is inadequate sampling (which is more likely in high signal-to-noise ratio cases), you should increase the neff parameter in the CIP sub file, reduce the number of samples requested, and run the CIP script more times (cip-explode-jobs option) (this can be done without setting up a new run). This will help ensure that the peak is accurately sampled and that the number of samples is close to neff. If the problem is due to interpolation errors, consider running additional iterations to have sufficient lnL evaluations around the peak. \n") 
        CIP_is_good = False

    f.write("\n")
    if CIP_is_good:
        f.write("\t--> CIP status: GOOD! <--\n")
    else:
        f.write("\t--> CIP status: BAD! <--\n")
    f.write("\n###########################################################################################\n")
    f.write("# Visual diagnostics\n")
    f.write("###########################################################################################\n")
    f.write("\t 1) Is the 90% credible interval mostly around the red points? If not, it could be that the run needs more iterations. If the SNR < 30, then the prior might impact it and the shift is expected.")
    f.write(f"\n\t 2) Has the parameter space been sufficiently explored? Are there blue points around the red points? Continuing the run will help if this is true with {run_diagnostics['latest_grid']} as your starting grid and copying this run's all.net as bonus.composite in your new run directory")
    f.write("\n\t 3) Is the approximate SNR captured close to True SNR? A significant difference implies the inference got stuck at a local lnL maxima. Happens rarely")

    f.close()
    print("###########################################################################################")
    print("# Run diagnositcs")
    print("###########################################################################################")

    for key in run_diagnostics:
        print(f"{key}: {run_diagnostics[key]}")

def check_extrinsic_present(path):
    return os.path.exists(f"{path}/extrinsic_posterior_samples.dat")
###########################################################################################
# Generate plots
###########################################################################################
# create plots folder
create_plots_folder(path)

# finding posterior files
main_posterior_files, main_iterations = find_posteriors_in_main(path)
if len(main_posterior_files) > 7:
    limit_main_iterations = 5
    main_posterior_files, main_iterations = find_posteriors_in_main(path, limit_iterations=limit_main_iterations)
subdag_posterior_files, subdag_iterations = find_posteriors_in_sub(path)

# plot neff
try:
    plot_neff_data(path)
except:
    # run this function so some information in run_diagnostics dict gets populated.
    get_lnL_cut_points(all_net_path, lnL_cut=15, error_threshold=0.4, composite=False)
    print("Couldn't plot CIP neff per worker for each iteration.")

# plot exploration corner
try:
    plot_exploration_corner(all_net_path)
except Exception as e:
    print(e)
    print("Couldn't plot exploration corner plot")

# plot sampled max lnL
try:
    plot_cip_max_lnL(path)
except Exception as e:
    print(e)
    print("Couldn't plot max lnL sampled by CIP per iteration.")

# plot likelihood exploration
try:
    plot_high_likelihood_expoloration(path)
except:
    print("Couldn't plot high likelihod exploration plot.")

# write sample statistics
write_sample_statistics(main_posterior_files[-1])

# plot histograms
plot_histograms(main_posterior_files, plot_title="Main", iterations=main_iterations, JSD = False)

# plot corner plots
if LISA:
    plot_corner(main_posterior_files, "Main", iterations = main_iterations, use_truths = use_truths)
    if eccentricity: 
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "eccentricity", "meanPerAno", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mc", "eta", "chi_eff", "eccentricity", "meanPerAno", "dec", "ra"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mtot", "q", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], use_truths = use_truths)
    else:
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "a1z", "a2z", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mc", "eta", "chi_eff", "dec", "ra"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "a1z", "a2z", "dec", "ra"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mtot", "q", "a1z", "a2z", "dec", "ra"], use_truths = use_truths)
else:
    # block with no spin and no eccentricity + precession
    if non_spinning:
        plot_corner(main_posterior_files, "Main",  parameters = ["mc", "eta"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final",  parameters = ["mc", "eta"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2"], use_truths = use_truths)
    else:
        plot_corner(main_posterior_files, "Main", iterations = main_iterations, use_truths = use_truths)
    
    # block with spins but split based on precession and eccentricity
    if eccentricity and not(precessing):
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "eccentricity", "meanPerAno"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mc", "eta", "chi_eff", "eccentricity", "meanPerAno"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mtot", "q", "s1z", "s2z", "eccentricity", "meanPerAno"], use_truths = use_truths)
    elif precessing and not(eccentricity):
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "chi_p"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mc", "eta", "chi_eff", "chi_p"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mtot", "q", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], use_truths = use_truths)
    elif precessing and eccentricity:
        plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "chi_p",  "eccentricity", "meanPerAno"], iterations = main_iterations, use_truths = use_truths)
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mc", "eta", "chi_eff", "chi_p", "eccentricity", "meanPerAno"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mtot", "q", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], use_truths = use_truths)
    elif not(precessing) and not(eccentricity) and not(non_spinning):
        plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "s1z", "s2z"], iterations = main_iterations, use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "s1z", "s2z"], use_truths = use_truths)
        plot_corner([main_posterior_files[-1]], "Final", parameters = ["mtot", "q", "s1z", "s2z"], use_truths = use_truths)

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
    plot_histograms(subdag_posterior_files, plot_title="Subdag", iterations=subdag_iterations, JSD = False)
    # plot corner plots
    if LISA:
        plot_corner(subdag_posterior_files, "Subdag", iterations=subdag_iterations, use_truths=use_truths)
        if eccentricity:
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "eccentricity", "meanPerAno", "dec", "ra"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mc", "eta", "chi_eff", "eccentricity", "meanPerAno", "dec", "ra"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["m1", "m2", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mtot", "q", "a1z", "a2z", "eccentricity", "meanPerAno", "dec", "ra"], use_truths=use_truths)
        else:
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "dec", "ra"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "a1z", "a2z", "dec", "ra"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mc", "eta", "chi_eff", "dec", "ra"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["m1", "m2", "a1z", "a2z", "dec", "ra"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mtot", "q", "a1z", "a2z", "dec", "ra"], use_truths=use_truths)
    else:
        if non_spinning:
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mc", "eta"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["m1", "m2"], use_truths=use_truths)
        else:
            plot_corner(subdag_posterior_files, "Subdag", iterations=subdag_iterations, use_truths=use_truths)

        if eccentricity and not(precessing):
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "eccentricity", "meanPerAno"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mc", "eta", "chi_eff", "eccentricity", "meanPerAno"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["m1", "m2", "s1z", "s2z", "eccentricity", "meanPerAno"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mtot", "q", "s1z", "s2z", "eccentricity", "meanPerAno"], use_truths=use_truths)
        elif precessing and not(eccentricity):
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "chi_p"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mc", "eta", "chi_eff", "chi_p"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mtot", "q", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y"], use_truths=use_truths)
        elif precessing and eccentricity:
            plot_corner(subdag_posterior_files, "Subdag", parameters=["mc", "eta", "chi_eff", "chi_p", "eccentricity", "meanPerAno"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mc", "eta", "chi_eff", "chi_p", "eccentricity", "meanPerAno"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["m1", "m2", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mtot", "q", "s1z", "s2z", "s1x", "s1y", "s2x", "s2y", "eccentricity", "meanPerAno"], use_truths=use_truths)
        elif not(precessing) and not(eccentricity) and not(non_spinning):
            plot_corner(subdag_posterior_files, "Subdag", parameters=["m1", "m2", "s1z", "s2z"], iterations=subdag_iterations, use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["m1", "m2", "s1z", "s2z"], use_truths=use_truths)
            plot_corner([subdag_posterior_files[-1]], "Subdag", parameters=["mtot", "q", "s1z", "s2z"], use_truths=use_truths)

    try:
        plot_JS_divergence(subdag_posterior_files[-1], subdag_posterior_files[-2], None, "Subdag_iteration") # the last two subdag iterations
    except:
        try:
            plot_JS_divergence(subdag_posterior_files[-1], subdag_posterior_files[-2], subdag_posterior_files[-3], "Subdag_iteration")
        except:
           print("Couldn't plot Jensen Shannon Divergence plot.") 
    plot_JS_divergence(main_posterior_files[-1], subdag_posterior_files[-1], None, "Main_subdag_iteration") # the last main and subdag iteration

if check_extrinsic_present(path):
    plot_corner([f"{path}/extrinsic_posterior_samples.dat"], "extrinsic", parameters = ["distance", "incl", "phiorb", "psi", "time"], use_truths = use_truths)
    plot_corner([f"{path}/extrinsic_posterior_samples.dat"], "extrinsic_source_mass", parameters = ["m1_source", "m2_source", "mtotal_source"], use_truths = False)
    plot_log_likelihood(path)

# run diagnostics
evaluate_run(run_diagnostics)
