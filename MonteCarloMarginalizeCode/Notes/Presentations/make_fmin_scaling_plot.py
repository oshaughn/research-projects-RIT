import numpy as np
from matplotlib import pylab as plt

#ILE_fmins = [10., 15., 20., 25., 30., 35., 40.]
ILE_data = [8664., 2172., 2674., 2174., 2162., 1892., 1890.] # runtimes in s (avg of 2 runs)
ILE_fmins = [15., 20., 25., 30., 35., 40.]
T1_data = [2195., 2679., 2124., 2189., 1885., 1861.]
EOB_data = [2202., 2407., 1859., 1847., 1584., 1564.]

rt_40 = 1861. # runtime of Bayesian PE using fmin=40.

def Bayes_runtime(fmin, rt_40):
    """
    Compute runtime of Bayesian PE for a min. freq. fmin
    given the rt_40 as the runtime when fmin=40 Hz.
    """
    return rt_40 * (fmin / 40.)**(-8./3.)

fmins = np.arange(10., 40., 0.1)

plt.semilogy(fmins, Bayes_runtime(fmins, rt_40), 'k-', label="Waveform-limited scaling")
plt.scatter(ILE_fmins, T1_data, label="rapid PE (T1)", c='b')
plt.scatter(ILE_fmins, EOB_data, label="rapid PE (EOBNRv2)", marker='x', c='r')
plt.title('PE Runtimes for $(1.55+1.23) M_\odot$ BNS signal')
plt.legend()
plt.xlabel("min. freq. (Hz)")
plt.ylabel("runtime (s)")
plt.show()
