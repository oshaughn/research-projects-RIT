#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from collections import namedtuple
import sys
plt.rcParams.update({
'axes.labelsize': 22,
'axes.titlesize': 22,
'font.size': 22,
'legend.fontsize': 14,
'xtick.labelsize': 18,
'ytick.labelsize': 18,
'font.family': 'serif',
'font.sans-serif': ['Bitstream Vera Sans'],
'font.serif': ['Times New Roman'],
'figure.dpi':100
}
)

# TO DO:
# PE summary
# plot injection + high lnL waveform
path = sys.argv[1]
corner_plot_exe = os.popen("which plot_posterior_corner.py").read()[:-1]
all_net_path = path + "/all.net"

truth_file_path = path + "/../mdc.xml.gz"
use_truths = False
if os.path.exists(truth_file_path):
    use_truths = True
    print(f"Using {truth_file_path} for truth values in corner plots!")

#######
def get_lnL_cut_points(all_net_path, lnL_cut = 15):
    data= np.loadtxt(all_net_path)
    lnL = data[:,9]
    max_lnL=np.max(lnL)
    no_points=len(lnL[lnL>=(max_lnL - lnL_cut)])
    return max_lnL, no_points

def create_plots_folder(base_dir_path):
    if not(os.path.exists(base_dir_path + "/plots")):
        os.mkdir(base_dir_path + "/plots")

def get_chirpmass_massratio_eta_totalmass_from_componentmasses(m1, m2):
    return np.array((m1*m2)**(3/5) / (m1+m2)**(1/5)).reshape(-1,1), np.array(m2/m1).reshape(-1,1), np.array((m1*m2) / (m1+m2)**(2)).reshape(-1,1), np.array(m1+m2).reshape(-1,1)

def get_index_for_parameter(parameter):
    if parameter == "mc":
        parameter_n = 8
    if parameter == "mtot":
        parameter_n = -2
    if parameter in ("a1z","s1z"):
        parameter_n = 4
    if parameter in ("a2z","s2z"):
        parameter_n = 7
    if parameter == "eta":
        parameter_n = 9
    if parameter == "m1":
        parameter_n = 0
    if parameter == "m2":
        parameter_n = 1
    if parameter == "q":
        parameter_n = -1
    return parameter_n

def get_chi_eff_from_mass_and_spins(posterior):
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
    posteriors_in_main = glob.glob(path_to_main_folder + "/posterior_samples*")
    posteriors_in_main.sort(key = os.path.getctime) # sort them according to creation time
    if limit_iterations:
        index = np.linspace(0, len(posteriors_in_main)-1, limit_iterations)
        index = np.array(index, dtype=int)
        return np.array(posteriors_in_main, dtype = str)[index], index + 1
    return posteriors_in_main, np.arange(len(posteriors_in_main)) + 1

def find_posteriors_in_sub(path_to_main_folder, limit_iterations = None):
    posteriors_in_subdag, iterations = find_posteriors_in_main(path_to_main_folder + "/iteration*cip*")
    if limit_iterations:
        index = np.linspace(0, len(posteriors_in_subdag)-1, limit_iterations)
        index = np.array(index, dtype=int)
        return np.array(posteriors_in_subdag, dtype = str)[index], index + 1
    else:
        return posteriors_in_subdag, np.arange(len(posteriors_in_subdag))

def calculate_JS_divergence(data1, data2):
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

def plot_ESS_data(path_to_main_folder):
    cip_iteration_folders= glob.glob(path_to_main_folder + "/iteration*cip*")
    cip_iteration_folders.sort(key = os.path.getctime)
    print(cip_iteration_folders)
    fig, ax = plt.subplots()
    ax.set_xlabel("iteration")
    ax.set_ylabel("neff")
    iterations=np.arange(len(cip_iteration_folders))
    n=0 #keep track of which iteration we are plotting for. Added this, with the above line so the iterations are int instead of floats
    try:
        neff_requested = os.popen('cat CIP_worker2.sub | grep -Eo "n-eff [+-]?[0-9]+([.][0-9]+)?"').read()[:-1].split(" ")[-1] # could find a better way to do this
        ax.axhline(y = float(neff_requested), linestyle = "--", color = "red", linewidth = 1.0, label = "Final iter neff")
        ax.legend(loc="upper left")
    except Exception as e:
        print(e)
        print("Couldn't plot requested neff.")
    for i in cip_iteration_folders:
        os.system(f"rm {i}/ESS_data.txt")
        cmd=f"for i in {i}/overlap-grid-*-*ESS* ; do cat $i | tail -n 1 >> {i}/ESS_data.txt; done"
        os.system(cmd)
        tmp_ESS_data=np.loadtxt(f"{i}/ESS_data.txt", usecols=[2])
        try:
            low, avg, high = np.percentile(tmp_ESS_data, [2.5,50,97.5])
            low_1_std, avg, high_1_std = np.percentile(tmp_ESS_data, [16,50,84])
            mini, maxi = np.min(tmp_ESS_data), np.max(tmp_ESS_data)
            ax.plot(iterations[n], mini, marker="x", color="black")
            ax.plot(iterations[n], maxi, marker="x", color="black")
            print(f"ESS detail iteration = {iterations[n]}: Average={avg:0.2f}, low std={low:0.2f}, high std={high:0.2f}")
            ax.errorbar(iterations[n], avg, yerr=np.array([avg-low,high-avg]).reshape(-1, 1), color = "royalblue", ecolor = "red", fmt ='o')
            ax.errorbar(iterations[n], avg, yerr=np.array([avg-low_1_std,high_1_std-avg]).reshape(-1, 1), color = "royalblue", ecolor = "green", fmt ='.')
        except Exception as e:
            print(e)
            print(f"Couldn't plot ESS for iteration = {iterations[n]}")
        n+=1
    fig.savefig(path+f"/plots/ESS_plot.png", bbox_inches='tight')

def plot_histograms(sorted_posterior_file_paths, plot_title, iterations = None, plot_legend = True, JSD = True):
    if iterations is None: # when you just want to plot final iterations histograms
        iterations = [-1]
        plot_legend = False
    all_net_data = convert_all_net_to_posterior_format(all_net_path)
    not_nan_lnL = np.argwhere(all_net_data[:,-3]>=np.max(all_net_data[:,-3]) - 15).flatten()#np.argwhere(~np.isnan(all_net_data[:,-3])).flatten()
    all_net_data = np.array(all_net_data[not_nan_lnL])
    for parameter in ["mc", "q", "eta", "m1", "m2", "s1z", "s2z", "chi_eff"]:
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
        if plot_legend: # don't create legend when only plotting finals iteration's historgrams
            ax.legend(loc = "upper right")
        fig.savefig(path+f"/plots/historgam_{plot_title}_{parameter}.png", bbox_inches='tight')
        plt.close()

def plot_corner(sorted_posterior_file_paths, plot_title, iterations = None, parameters = ["mc", "eta", "xi"], use_truths = False):
    max_lnL, no_points = get_lnL_cut_points(all_net_path)
    title = f"{plot_title},max_lnL={max_lnL:0.3f},points_cut={no_points}" 
    plotting_command = f"python {corner_plot_exe} --plot-1d-extra --lnL-cut 15 --use-all-composite-but-grayscale --composite-file {all_net_path} --quantiles None --ci-list [0.9] --use-title {title} --sigma-cut 0.5 "
    if iterations is not None:
        plotting_command += "--use-legend "
    else:
        iterations = [0]
    if use_truths:
        plotting_command += f"--truth-file {truth_file_path} "
    for i, parameter in enumerate(parameters):
        plotting_command += f"--parameter {parameter} "
    for iteration in np.arange(len(iterations)):
        plotting_command += f"--posterior-file {sorted_posterior_file_paths[iteration]} --posterior-label {iterations[iteration]} "
    os.system(plotting_command)
    os.system(f"mv corner_" + "_".join(parameters) + ".png" + " plots/corner_" + "_".join(parameters) +"_" + plot_title + ".png")
    for i, parameter in enumerate(parameters):
        os.system(f"mv {parameter}.png plots/{parameter}_{plot_title}.png")
        os.system(f"mv {parameter}_cum.png plots/{parameter}_cum_{plot_title}.png")
  
def plot_JS_divergence(posterior_1_path, posterior_2_path, plot_title, parameters = ["mc","eta", "m1", "m2", "s1z", "s2z", "chi_eff"]):
    posterior_data1 = np.loadtxt(posterior_1_path)
    posterior_data2 = np.loadtxt(posterior_2_path)
    JSD_array = []
    JSD_error = []
    for parameter in parameters:
        if parameter == "chi_eff":
            data1, data2 = get_chi_eff_from_mass_and_spins(posterior_data1), get_chi_eff_from_mass_and_spins(posterior_data2)
            JSD = calculate_JS_divergence(data1, data2)
        else:
            parameter_n = get_index_for_parameter(parameter)
            JSD = calculate_JS_divergence(posterior_data1[:, parameter_n], posterior_data2[:, parameter_n])
        JSD_array.append(JSD.median)
        JSD_error.append([JSD.minus, JSD.plus])
    fig, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.set_ylabel("JSD")
    ax.axhline( y =0.05, linewidth = 1.0, linestyle = "--", color = "red")
    ax.errorbar(parameters, JSD_array, np.array(JSD_error).T,  color = "royalblue", ecolor = "red", fmt ='o', markersize = 5)
    fig.savefig(path+f"/plots/JSD_{plot_title}.png", bbox_inches='tight')

# create plots folder
create_plots_folder(path)

# finding posterior files
main_posterior_files, main_iterations = find_posteriors_in_main(path)
if len(main_posterior_files) > 7:
    limit_main_iterations = 5
    main_posterior_files, main_iterations = find_posteriors_in_main(path, limit_iterations=limit_main_iterations)
subdag_posterior_files, subdag_iterations = find_posteriors_in_sub(path)

# plot ESS
plot_ESS_data(path)

# plot histograms
plot_histograms(main_posterior_files, plot_title="Main", iterations=main_iterations, JSD = False)

# plot corner plots
plot_corner(main_posterior_files, "Main", iterations = main_iterations, use_truths = use_truths)
plot_corner([main_posterior_files[-1]], "Final", use_truths = use_truths)
plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "a1z", "a2z"], use_truths = use_truths)
plot_corner([main_posterior_files[-1]], "Final", parameters = ["m1", "m2", "mtot", "mc"], use_truths = use_truths)
plot_corner([main_posterior_files[-1]], "Final", parameters = ["chi_eff", "a1z", "a2z"], use_truths = use_truths)
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

