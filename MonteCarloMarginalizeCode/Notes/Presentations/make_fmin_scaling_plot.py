import numpy as np
from matplotlib import pylab as plt

ILE_fmins = [20., 25., 30., 35., 40.]
ILE_data = [2674., 2174., 2162., 1892., 1890.] # runtimes in s (avg of 2 runs)

rt_40 = 1890. # runtime of Bayesian PE using fmin=40.

def Bayes_runtime(fmin, rt_40):
    """
    Compute runtime of Bayesian PE for a min. freq. fmin
    given the rt_40 as the runtime when fmin=40 Hz.
    """
    return rt_40 * (fmin / 40.)**(-8./3.)

fmins = np.arange(10., 40., 0.1)

plt.semilogy(fmins, Bayes_runtime(fmins, rt_40), label="Waveform-limited scaling")
plt.scatter(ILE_fmins, ILE_data, label="rapid PE")
plt.legend()
plt.xlabel("min. freq. (Hz)")
plt.ylabel("runtime (s)")
plt.show()
