#!/usr/bin/env python
"""This code is meant to check the health of a RIFT run as it progresses and after it has finished. python plot_RIFT.py path/to/rundir/"""
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
import sys

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

###########################################################################################
# Path and Configuration Setup
###########################################################################################
path = sys.argv[1]
LISA = True

# Locate corner plot executable
corner_plot_exe = os.popen("which plot_posterior_corner.py").read()[:-1]

# Define file paths
all_net_path = os.path.join(path, "all.net")
truth_file_path = os.path.join(path, "../mdc.xml.gz")

# Determine if truth file should be used
use_truths = os.path.exists(truth_file_path)
if use_truths:
    print(f"Using {truth_file_path} for truth values in corner plots!")

# Initialize diagnostics dictionary
run_diagnostics = {
    "JSD": {}
}

###########################################################################################
# Functions
###########################################################################################
def get_lnL_cut_points(all_net_path, lnL_cut=15, error_threshold=0.4, composite=False):
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
    lnL = data[:, 9]
    error = data[:, 10]
    
    # Adjust columns if LISA is True
    if LISA:
        lnL = data[:, 11]
        error = data[:, 12]
    
    # Remove NaN values from lnL
    total_points = len(lnL)
    lnL = lnL[~np.isnan(lnL)]

    # Find high likelihood points based on lnL_cut
    max_lnL = np.max(lnL)
    if composite:
        max_lnL_composite = max_lnL
        max_lnL = run_diagnostics["max_lnL"]
    high_lnL_indices = np.argwhere(lnL >= (max_lnL - lnL_cut)).flatten()
    high_lnL_points = len(high_lnL_indices)
    lnL = lnL[high_lnL_indices]
    error = error[high_lnL_indices]

    # Filter high lnL points with low Monte Carlo error
    low_error_indices = np.argwhere(error < error_threshold).flatten()
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
            "total_high_lnL_points": high_lnL_points
        })

        return np.round(max_lnL, 3), no_points

def create_plots_folder(base_dir_path):
    """
    Creates a 'plots' folder in the specified base directory if it does not already exist.

    Args:
        base_dir_path (str): Path to the base directory where the 'plots' folder will be created.
    """
    if not(os.path.exists(base_dir_path + "/plots")):
        os.mkdir(base_dir_path + "/plots")

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
    parameter_indices = {
        "mc": 8,
        "mtot": -2,
        "a1z": 4,
        "s1z": 4,
        "a2z": 7,
        "s2z": 7,
        "eta": 9,
        "m1": 0,
        "m2": 1,
        "q": -1,
        "dec": 13,
        "ra": 12
    }
    
    return parameter_indices.get(parameter, None)  # Return None if parameter is not found

def get_chi_eff_from_mass_and_spins(posterior):
    """
    Computes the effective spin parameter (χ_eff) from the posterior data.

    Args:
        posterior (numpy.ndarray): Array where each row represents a set of parameters. 

    Returns:
        numpy.ndarray: Array of χ_eff values computed from the posterior data.
    """
    parameter_m1, parameter_m2 = get_index_for_parameter("m1"), get_index_for_parameter("m2")
    parameter_s1z, parameter_s2z = get_index_for_parameter("s1z"), get_index_for_parameter("s2z")
    return (posterior[:,parameter_m1]*posterior[:,parameter_s1z] + posterior[:,parameter_m2]*posterior[:,parameter_s2z]) / (posterior[:,parameter_m1] + posterior[:,parameter_m2])

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
    run_diagnostics["composite_information"] = {}
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("high lnL points")
    ax.set_title(f"Total high lnL points = {run_diagnostics['high_lnL_points']}")
    collect_data = []
    for iteration in np.arange(0, run_diagnostics["latest_iteration"]+1, 1):
        run_diagnostics["composite_information"][iteration] = {}
        try:
            max_lnL, no_points, max_lnL_composite, total_points = get_lnL_cut_points(f"{path_to_main_folder}/consolidated_{iteration}.composite", composite=True)
        except Exception as e:
            print(f"Error loading file {path_to_main_folder}/consolidated_{iteration}.composite: {e}")
            continue
        print(iteration, max_lnL, no_points, max_lnL_composite, total_points)
        percent_high_lnL_points =  np.round(no_points/total_points, 2)
        collect_data.append(no_points)
        ax.scatter(iteration, no_points, label = f"{max_lnL_composite} ({percent_high_lnL_points})", s=15, color="red")
        run_diagnostics["composite_information"][iteration].update({
                "max_lnL":max_lnL_composite,
                "high_lnL_points":no_points,
                "percent_high_lnL_points": percent_high_lnL_points})
    ax.grid(alpha=0.4)
    ax.plot(collect_data, color = "black", linestyle = "--", linewidth = 1.5, alpha = 0.5)
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
    # find CIP folders
    cip_iteration_folders= glob.glob(path_to_main_folder + "/iteration*cip*")
    
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("neff")
    iterations=np.arange(len(cip_iteration_folders)-1) # last folders don't usually have anything
    # read requested neff from CIP sub files
    try:
        run_diagnostics["CIP_neff"] = {}
        neff_requested_0 = os.popen('cat CIP_worker0.sub | grep -Eo "\-\-n-eff [+-]?[0-9]+([.][0-9]+)?"').read()[:-1].split(" ")[-1]
        ax.axhline(y = float(neff_requested_0), linestyle = "--", color = "black", alpha = 0.8, linewidth = 1.0, label = "worker 0 neff")
        run_diagnostics["CIP_neff"]["CIP_worker0"] = np.round(float(neff_requested_0), 2)
        neff_requested_1 = os.popen('cat CIP_worker1.sub | grep -Eo "\-\-n-eff [+-]?[0-9]+([.][0-9]+)?"').read()[:-1].split(" ")[-1]
        ax.axhline(y = float(neff_requested_1), linestyle = "--", color = "blue", alpha = 0.8, linewidth = 1.0, label = "worker 1 neff")
        run_diagnostics["CIP_neff"]["CIP_worker1"] = np.round(float(neff_requested_1), 2)
        neff_requested_2 = os.popen('cat CIP_worker2.sub | grep -Eo "\-\-n-eff [+-]?[0-9]+([.][0-9]+)?"').read()[:-1].split(" ")[-1] # could find a better way to do this
        ax.axhline(y = float(neff_requested_2), linestyle = "--", color = "red", alpha = 0.8, linewidth = 1.0, label = "worker 2 neff")
        run_diagnostics["CIP_neff"]["CIP_worker2"] = np.round(float(neff_requested_2), 2)
    except Exception as e:
        print(e)
        print("Couldn't plot requested neff.")
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
    # read max lnL data from CIP output files
    print(f"READING lnL FILES FROM iteration_{iterations[iteration_prog]}_cip")
    lnL_files_last_iteration = glob.glob(path_to_main_folder + f"/iteration_{iterations[iteration_prog]}_cip/*lnL*")
    run_diagnostics["latest_grid"] = f"overlap-grid-{iteration_prog+1}.xml.gz"
    run_diagnostics["latest_iteration"] = int(iteration_prog)
    collect_lnL = []
    for j in np.arange(len(lnL_files_last_iteration)):
        data = np.loadtxt(lnL_files_last_iteration[j])
        collect_lnL.append(np.max(data))
    collect_lnL = np.array(collect_lnL)
    max_lnL, no_points = get_lnL_cut_points(all_net_path)
    index = np.argwhere(max_lnL - collect_lnL >= 2)
    # print, save diagnostics and plot
    print(f"Max lnL  = {max_lnL}, average max lnL from workers = {np.mean(collect_lnL)} with std = {np.std(collect_lnL)}")
    print(f"Total number of worker in final iteration = {len(lnL_files_last_iteration)}, number of them which didn't capture max_lnL = {len(index)}")
    run_diagnostics["cip_average_max_lnL_sampled"] = np.round(np.mean(collect_lnL), 2)
    run_diagnostics["cip_std_max_lnL_sampled"] = np.round(np.std(collect_lnL), 3)
    ax.set_title(f"{len(index)} / {len(lnL_files_last_iteration)}")
    ax.set_xticks(np.arange(0, run_diagnostics["latest_iteration"]+1, 1))
    fig.savefig(path+f"/plots/Neff_plot.png", bbox_inches='tight')

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
    # when you just want to plot final iterations histograms
    if iterations is None: 
        iterations = [-1]
        plot_legend = False
    # all_net_data = convert_all_net_to_posterior_format(all_net_path)
    # not_nan_lnL = np.argwhere(all_net_data[:,-3]>=np.max(all_net_data[:,-3]) - 15).flatten()#np.argwhere(~np.isnan(all_net_data[:,-3])).flatten()
    # all_net_data = np.array(all_net_data[not_nan_lnL])
    parameters =  ["mc", "q", "eta", "m1", "m2", "s1z", "s2z", "chi_eff"]
    # for LISA include skylocation
    if LISA:
        parameters.append("dec")
        parameters.append("ra")
    for parameter in parameters:
        print(f"Plotting histogram for {parameter}")
        fig, ax = plt.subplots()
        ax.set_title(plot_title)
        ax.set_xlabel(parameter)
        ax.set_yticks([])
        for i in np.arange(len(sorted_posterior_file_paths)):
            line_label = str(iterations[i])
            if parameter == "chi_eff":
                posterior_data = np.loadtxt(sorted_posterior_file_paths[i])
                data = get_chi_eff_from_mass_and_spins(posterior_data)
            else:
                parameter_index = get_index_for_parameter(parameter)
                data = np.loadtxt(sorted_posterior_file_paths[i])[:,parameter_index]
            if i > 0 and JSD:
                JS_test = calculate_JS_divergence(data, data_previous)
                line_label +=f" ({calculate_JS_divergence(data, data_previous).median:0.3f})"
            ax.hist(data, label = line_label, histtype="step", bins = 50, density=True, linewidth=1.0)
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
        fig.savefig(path+f"/plots/historgam_{plot_title}_{parameter}.png", bbox_inches='tight')
        plt.close()

def plot_corner(sorted_posterior_file_paths, plot_title, iterations = None, parameters = ["mc", "eta", "xi"], use_truths = False):
    """
    Generates corner plots for posterior samples using a specified plotting executable.

    Args:
        sorted_posterior_file_paths (list of str): List of file paths to sorted posterior samples.
        plot_title (str): Title for the plot, used in filenames.
        iterations (list of int, optional): List of iteration numbers to include in the plot. Defaults to [0] if None.
        parameters (list of str): List of parameters to include in the plot. Defaults to ["mc", "eta", "xi"].
        use_truths (bool): Whether to include truth values in the plot. Defaults to False.
    """
    max_lnL, no_points = run_diagnostics["max_lnL"], run_diagnostics["high_lnL_points"]  
    title = f"max_lnL={max_lnL:0.2f},points_cut={no_points}" 
    plotting_command = f"python {corner_plot_exe} --plot-1d-extra --lnL-cut 15 --use-all-composite-but-grayscale --composite-file {all_net_path} --quantiles None --ci-list [0.9] --use-title {title} --sigma-cut 0.4 "
     # Append iteration-related options to the command
    if iterations is not None:
        plotting_command += "--use-legend "
    else:
        iterations = [0]

    # Include truth file if required
    if use_truths:
        plotting_command += f"--truth-file {truth_file_path} "

    # Add parameter options to the command
    for parameter in parameters:
        plotting_command += f"--parameter {parameter} "

    # Add posterior file paths and labels to the command
    for i, posterior_file in enumerate(sorted_posterior_file_paths):
        plotting_command += f"--posterior-file {posterior_file} --posterior-label {iterations[i]} "

    # Append LISA flag if applicable
    if LISA:
        plotting_command += "--LISA "

    # Execute the plotting command
    os.system(plotting_command)

    # Move and rename output files
    corner_plot_filename = f"corner_{'_'.join(parameters)}.png"
    new_corner_plot_path = f"plots/corner_{'_'.join(parameters)}_{plot_title}.png"
    os.system(f"mv {corner_plot_filename} {new_corner_plot_path}")

    # Move and rename individual parameter plots
    for parameter in parameters:
        os.system(f"mv {parameter}.png plots/{parameter}_{plot_title}.png")
        os.system(f"mv {parameter}_cum.png plots/{parameter}_cum_{plot_title}.png") 

def plot_JS_divergence(posterior_1_path, posterior_2_path, plot_title, parameters = ["mc","eta", "m1", "m2", "s1z", "s2z", "chi_eff"]):
    """
    Plots Jensen-Shannon Divergence (JSD) between two posterior datasets for specified parameters.

    Args:
        posterior_1_path (str): File path to the first posterior dataset.
        posterior_2_path (str): File path to the second posterior dataset.
        plot_title (str): Title for the plot and filename.
        parameters (list of str): List of parameters to calculate and plot JSD for.
    """
    if LISA:
        parameters.append("dec")
        parameters.append("ra")
    posterior_data1 = np.loadtxt(posterior_1_path)
    posterior_data2 = np.loadtxt(posterior_2_path)
    JSD_array = []
    JSD_error = []
    run_diagnostics["JSD"][plot_title] = {}
    for parameter in parameters:
        if parameter == "chi_eff":
            data1, data2 = get_chi_eff_from_mass_and_spins(posterior_data1), get_chi_eff_from_mass_and_spins(posterior_data2)
            JSD = calculate_JS_divergence(data1, data2)
        else:
            parameter_n = get_index_for_parameter(parameter)
            JSD = calculate_JS_divergence(posterior_data1[:, parameter_n], posterior_data2[:, parameter_n])
        JSD_array.append(JSD.median)
        JSD_error.append([JSD.minus, JSD.plus])
        run_diagnostics["JSD"][plot_title][parameter] = np.round(JSD.median, 3)
    fig, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.set_ylabel("JSD")
    ax.axhline( y =0.05, linewidth = 1.0, linestyle = "--", color = "red")
    ax.errorbar(parameters, JSD_array, np.array(JSD_error).T,  color = "royalblue", ecolor = "red", fmt ='o', markersize = 5)
    fig.savefig(path+f"/plots/JSD_{plot_title}.png", bbox_inches='tight')

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
    f.write(f"Total number of lnL evaluations = {run_diagnostics['total_lnL_evaluations']}\n")
    f.write(f"Total number of high lnL points = {run_diagnostics['total_high_lnL_points']}\n")
    f.write(f"Total number of high lnL points used = {run_diagnostics['high_lnL_points']}\n")
    f.write(f"Total number of high lnL points not used due to large error = {run_diagnostics['high_lnL_points_with_large_error']}\n")
    f.write(f"\nLikelihood exploration data per iteration: \n{run_diagnostics[composite_information]}\n")
    ILE_is_good = True
    f.write("\n")
    if run_diagnostics['high_lnL_points_with_large_error']/run_diagnostics['total_high_lnL_points'] > 0.5:
        f.write(f"\t--> Large number of points have a high Monte Carlo error (sigma = 0.4). Consider reducing d-max, increasing d-min, increasing n-max and/or changing the sampler.\n")
        ILE_is_good = False
    if run_diagnostics['high_lnL_points'] <= 500:
        f.write(f"\t--> Number of high likelihood points is less than 500, which could be caused due to initial grid not having sufficient resolution. Considering reducing the parameter space and/or increasing the number of points on the grid.\n")
        ILE_is_good = False
    if 500 < run_diagnostics['high_lnL_points'] < 5000:
        f.write(f"\t--> Number of high likelihood points is less than 5000, consider rerunning with {run_diagnostics['latest_grid']} as your starting grid and copying this run's all.net as bonus.composite in your new run directory.\n")
        ILE_is_good = False
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
    last_iter_neff = run_diagnostics['CIP_neff'][list(run_diagnostics['CIP_neff'].keys())[-1]]
    last_iter_neff_achieved = run_diagnostics['CIP_neff_achieved'][list(run_diagnostics['CIP_neff_achieved'].keys())[-1]]
    if last_iter_neff > last_iter_neff_achieved:
        f.write(f"\t--> neff has not been reached, the posterior distribution may be wider and/or irregular. To address this, try narrowing the parameter space or switching to a different sampler. Alternatively, you can reduce the neff for each CIP job (>10) and increase the number of CIP jobs submitted per iteration.\n")
        CIP_is_good = False
    # CIP JSD
    f.write(f"\nCIP Jensen-Shannon divergence: {run_diagnostics['JSD']}\n")
    JSD_not_good = {}
    JSD_is_good = True
    for iteration_type in run_diagnostics['JSD']:
        JSD_not_good[iteration_type] = {}
        for param in run_diagnostics['JSD'][iteration_type]:
            if run_diagnostics['JSD'][iteration_type][param] > 0.05:
                JSD_not_good[iteration_type][param] = run_diagnostics['JSD'][iteration_type][param]
    for iteration_type in run_diagnostics['JSD']:
        print(JSD_not_good[iteration_type])
        if len(JSD_not_good[iteration_type]) > 0:
            JSD_is_good = False
    if JSD_is_good is False:
        f.write(f"\t--> Following parameters have Jensen-Shannon Divergence values greater than 0.05: {JSD_not_good}\n")
        f.write(f"\t--> If the Jensen-Shannon Divergence for any parameter between the last and second-to-last iterations is greater than 0.05, it means the run has not yet converged. In this case, you should rerun the analysis using {run_diagnostics['latest_grid']} as the starting grid. Additionally, copy this run's all.net file to a new file named bonus.composite in your new run directory. \n") 
        CIP_is_good = False
    # CIP sampling
    f.write(f"\nAverage max lnL sampled by CIP in iteration {run_diagnostics['latest_iteration']} is: {run_diagnostics['cip_average_max_lnL_sampled']} +- {run_diagnostics['cip_std_max_lnL_sampled']}. Max lnL in all.net is {run_diagnostics['max_lnL']}.\n")
    if run_diagnostics['max_lnL'] - run_diagnostics['cip_average_max_lnL_sampled'] > 2:
        f.write(f"\t--> The difference between the maximum lnL value from all.net and the average maximum lnL value sampled by CIP is more than 2, which could cause the peak to be slightly shifted. This discrepancy might be because CIP hasn't sampled the peak well enough or due to interpolation errors. If the issue is inadequate sampling (which is more likely in high signal-to-noise ratio cases), you should increase the neff parameter in the CIP sub file, reduce the number of samples requested, and run the CIP script more times (this can be done without setting up a run). This will help ensure that the peak is accurately sampled and that the number of samples is close to neff. If the problem is due to interpolation errors, consider running additional iterations to have sufficient lnL evaluations around the peak. \n") 
        CIP_is_good = False

    f.write("\n")
    if CIP_is_good:
        f.write("\t--> CIP status: GOOD! <--\n")
    else:
        f.write("\t--> CIP status: BAD! <--\n")
    f.close()

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
plot_neff_data(path)

# plot likelihood exploration
plot_high_likelihood_expoloration(path)

# plot histograms
plot_histograms(main_posterior_files, plot_title="Main", iterations=main_iterations, JSD = False)

# plot corner plots
if LISA:
    plot_corner(main_posterior_files, "Main", iterations = main_iterations, use_truths = use_truths)
    plot_corner(main_posterior_files, "Main", parameters = ["mc", "eta", "chi_eff", "dec", "ra"], iterations = main_iterations, use_truths = use_truths)
    plot_corner([main_posterior_files[-1]], "Final", parameters = ["mc", "eta", "chi_eff", "dec", "ra"], use_truths = use_truths)
    plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "a1z", "a2z", "dec", "ra"], use_truths = use_truths)
    plot_corner([main_posterior_files[-1]], "Final", parameters = ["mtot", "q", "a1z", "a2z", "dec", "ra"], use_truths = use_truths)
else:
    plot_corner(main_posterior_files, "Main", iterations = main_iterations, use_truths = use_truths)
    plot_corner(main_posterior_files, "Main", parameters = ["m1", "m2", "a1z", "a2z"], iterations = main_iterations, use_truths = use_truths)
    plot_corner(main_posterior_files, "Main", parameters = ["mtot", "q", "a1z", "a2z"], iterations = main_iterations, use_truths = use_truths)
    plot_corner([main_posterior_files[-1]], "Final", use_truths = use_truths)
    plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "a1z", "a2z"], use_truths = use_truths)
    plot_corner([main_posterior_files[-1]], "Final", parameters = ["mtot", "q", "a1z", "a2z"], use_truths = use_truths)

# plot JS test
plot_JS_divergence(main_posterior_files[-1], main_posterior_files[-2], "Main_iteration") # the last two main iterations


# is there a subdag? If not, don't plot!
if len(subdag_posterior_files) == 0:
    analyse_subdag = False
else:
    analyse_subdag = True

if len(subdag_posterior_files) > 8 and analyse_subdag == True:
    limit_subdag_iterations = 5 # if the number of subdag iterations is high, only show five iterations to prevent overcrowding
    subdag_posterior_files, subdag_iterations = find_posteriors_in_sub(path, limit_iterations=limit_subdag_iterations)

if analyse_subdag:
    plot_histograms(subdag_posterior_files, plot_title="Subdag", iterations=subdag_iterations, JSD = False)
    plot_corner(subdag_posterior_files, "Subdag", iterations = subdag_iterations, use_truths = use_truths)
    plot_JS_divergence(subdag_posterior_files[-1], subdag_posterior_files[-2], "Subdag") # the last two subdag iterations
    plot_JS_divergence(main_posterior_files[-1], subdag_posterior_files[-1], "Main") # the last main and subdag iteration

# run diagnostics
print(run_diagnostics)
evaluate_run(run_diagnostics)
