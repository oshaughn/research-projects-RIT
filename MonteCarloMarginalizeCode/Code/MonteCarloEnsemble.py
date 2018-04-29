import numpy as np
import weighted_gmm
#import matplotlib.pyplot as plt
#from multiprocessing import cpu_count, Pool  # switch to 'multiprocess' instead of 'multiprocessing'  https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
#from multiprocess import cpu_count, Pool  # switch to 'multiprocess' instead of 'multiprocessing'  https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
from multiprocessing.dummy import cpu_count, Pool  # switch to 'multiprocess' instead of 'multiprocessing'  https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
from time import time

import warnings

import dill # so I can pickle lambda functions: https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

class integrator(object):
    '''
    
    A class for evaluating multivariate integrals using an adaptive monte carlo method.  
    It is assumed that the integrand can be well-approximated by a mixture of Gaussians.
    
    Parameters
    
    d: Number of dimensions
    
    bounds: A (d x 2) array where each row is [lower_bound, upper_bound] for its
    correspondind dimension
    
    gmm_dict: dictionary where each key is a tuple of dimension indicies and each value
    is a either a mixture model object or None.  Must contain all dimensions with no repeats.
    
    n_comp: number of gaussian components for each dimension
    
    After creating an instance of this object and defining func to be a vectorized python
    function, calling monte_carlo_integrator(func) will return the integral.
    
    '''

    def __init__(self, d, bounds, gmm_dict, n_comp):
        self.d = d
        self.bounds = bounds
        self.gmm_dict = gmm_dict
        self.n_comp = n_comp
        self.t = 1.05
        self.n_draw = (5000 * (self.d**2)) # this is a VERY arbitrary choice about the number of 
                                 # samples that are necessary
        
    def evaluate_function_over_sample_array(self, sample_array, func):
        '''
        Uses a multiprocessing pool to split up function evaluation over an array of samples.
        
        Note: func must be a function that takes a numpy array as its parameter and
              returns a vertical numpy array of function values (e.g. numpy vectorized, etc.)
        '''
        n = self.n_draw
        max_procs = cpu_count()
        array_list = []
        length = int(n / max_procs) # length of each split array
        a = 0
        # split the sample array
        while a < max_procs - 1:
            array_list.append(sample_array[a * length : (a + 1) * length])
            a += 1
        # take care of the weird-sized last piece (in case it didn't split evenly)
        array_list.append(sample_array[a * length : n + 1])
        p = Pool(max_procs) # create pool
        value_array_list = p.map(func, array_list)
        # put results back into one array to return
        p.close()
        value_array = np.nan_to_num(np.concatenate(value_array_list))
        return value_array

    def uniform(self, func):
        '''
        Uniformly samples the function, returning n d-dimensional samples and the function
        value at those samples.
        '''
        n = self.n_draw
        d = self.d
        bounds = self.bounds
        llim_array = np.rot90(bounds[:,[0]])
        rlim_array = np.rot90(bounds[:,[1]])
        sample_array = np.random.uniform(llim_array, rlim_array, (n, d))
        value_array = self.evaluate_function_over_sample_array(sample_array, func)
        return sample_array, value_array

    def sample_from_gmm(self, gmm_dict, func):
        '''
        Samples each dimension according to the Gaussian Mixture Model for that dimension.
        If no Mixture model exists, samples uniformly. Returns the n x d array of samples,
        the responsibility of each sample according to the model, and the function value
        for each sample.
        '''
        n = self.n_draw
        d = self.d
        bounds = self.bounds
        llim_array = np.rot90(bounds[:,[0]])
        rlim_array = np.rot90(bounds[:,[1]])
        sample_array = np.empty((n, d))
        p_array = np.ones((n, 1))
        for dim_group in gmm_dict:
            clf = gmm_dict[dim_group]
            if clf:
                # get samples for this set of dimensions
                sample_column = clf.sample(n_samples=n)
                index = 0
                for dim in dim_group:
                    # replace zeros with samples
                    sample_array[:,[dim]] = sample_column[:,[index]]
                    index += 1
                # get responsibilities for samples and take care of orientation, multiply
                # by existing p_array so that the end product is the total responsibility 
                # for each d-dimensional sample
                p_array *= np.rot90([np.exp(clf.score(sample_column))], -1)
            else:
                # clf doesn't exist yet, sample uniformly
                for dim in dim_group:
                    llim = llim_array[0][dim]
                    rlim = rlim_array[0][dim]
                    sample_column = np.random.uniform(llim, rlim, (n, 1))
                    sample_array[:,[dim]] = sample_column
                    p_array *= 1 / (rlim - llim) 
        # get function values
        value_array = self.evaluate_function_over_sample_array(sample_array, func)
        return sample_array, p_array, value_array

    def calc_integral(self, sample_array, value_array, p_array):
        '''
        Performs the monte carlo integral for the given function values and responsibilities.
        
        Note: Ignores samples outside user-specified bounds
        '''
        n = self.n_draw
        d = self.d
        bounds = self.bounds
        
        # take care of points outside domain
        
        for dim in range(d):
            llim = bounds[dim][0]
            rlim = bounds[dim][1]
            sample_column = sample_array[:,[dim]]
            value_array[sample_column < llim] = 0
            value_array[sample_column > rlim] = 0
        
        # do integration
        
        value_array /= p_array
        i = np.sum(value_array)
        return (1.0 / n) * i

    def fit_gmm(self, sample_array, value_array, gmm_dict):
        '''
        Attempts to fit a Gaussian Mixture Model to the data.
        '''
        n = self.n_draw
        d = self.d
        n_comp = self.n_comp
        t = time()
        for dim_group in gmm_dict:
            clf = gmm_dict[dim_group]
            if not clf:
                clf = weighted_gmm.WeightedGMM(n_components=n_comp)
            else:
                clf.warm_start = True
            samples_to_fit = np.empty((n, len(dim_group)))
            index = 0
            for dim in dim_group:
                samples_to_fit[:,[index]] = sample_array[:,[dim]]
                index += 1
                #clf.warm_start = True put this somewhere
            try:
                clf.fit(X=samples_to_fit, w=abs(value_array))
                gmm_dict[dim_group] = clf
            except KeyboardInterrupt:
                return False
            except:
                # mixture model failed to fit, revert to uniform sampling
                gmm_dict[dim_group] = None
            if clf:
                if not clf.converged_:
                    # mixture model failed to fit, revert to uniform sampling
                    gmm_dict[dim_group] = None
        #print('time to train:', round(time() - t, 2), 'seconds')
        return gmm_dict

    def integrate(self, func,*args, **kwargs):
        '''
        Main function to sample the integrand, model each dimension's distribution, and
        iteratively integrate the function and re-train the model until convergence is reached.
        
        func: integrand function (must be able to take numpy array as parameter)
        '''
        d = self.d
        t = self.t
        nmax = kwargs["nmax"] if kwargs.has_key("nmax") else 1e5 #float("inf")
        neff = kwargs["neff"] if kwargs.has_key("neff") else np.float128("inf")
        n = kwargs["n"] if kwargs.has_key("n") else min(10000, self.n_draw)
        self.n_draw = n   # RESET THIS NUMBER: It controls the draw size
        gmm_dict = self.gmm_dict
        sample_array, value_array = self.uniform(func)
        count = 0
        total_integral = 0
        target_count = np.max([int(nmax*1.0/n),1])
        previous = False
        total_iters = 0

        ntotal =n  # we already drew n samples already !
        maxval = -float("Inf")
        maxlnL = -float("Inf")
        eff_samp =0
        var=0
        while count < target_count:
            total_iters += 1
            
            # the version of the scikit-learn mixture model in use here calls a deprecated
            # function in scikit-learn, which causes it to print a bunch of annoying warnings.
            # we don't need this for now, so we ignore the warnings.
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # fit the model
                gmm_dict = self.fit_gmm(sample_array, value_array, gmm_dict)
                if not gmm_dict:
                    break
                # sample from newly-fitted GMM
                sample_array, p_array, value_array = self.sample_from_gmm(gmm_dict, func)
                ntotal += n
            # convert value_array to float, if needed
            value_array  = value_array.astype(float)
            integral = self.calc_integral(sample_array, value_array, p_array)
            maxlnL = np.max([maxlnL, np.log(np.max(sample_array))])
            if True: #verbose:
                print " :",  ntotal, eff_samp, np.sqrt(2*maxlnL), np.sqrt(2*np.log(integral/ntotal)), np.log(integral/ntotal)-maxlnL, np.sqrt(var*ntotal)/integral
                print "    - integral ", integral
            if previous: # check if there is a previous integral for comparison
                # (rough) method to check for convergence
                if (integral / previous) < t and (previous / integral) < t:
                    total_integral += integral
                    count += 1
                else: # something is weird, restart everything. change this
                    total_integral = 0
                    count = 0
            previous = integral
        #plt.scatter(sample_array[:, [0]], value_array)
        #plt.show()
        #plt.hist(sample_array[:, [0]], 100)
        #plt.show()
        return total_integral / target_count
