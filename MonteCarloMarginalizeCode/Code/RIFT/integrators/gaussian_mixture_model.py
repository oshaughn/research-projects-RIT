# -*- coding: utf-8 -*-
'''
Gaussian Mixture Model
----------------------
Fit a Gaussian Mixture Model (GMM) to data and draw samples from it. Uses the
Expectation-Maximization algorithm.

Weighted data GMM formulae: different from framework in eg. https://arxiv.org/pdf/1509.01509.pdf
'''


from six.moves import range

import numpy as np
from scipy.stats import multivariate_normal,norm

try:
    import cupy
    import cupyx.scipy.special
    xpy_default = cupy
    xpy_special_default = cupyx.scipy.special
    identity_convert = cupy.asnumpy
    identity_convert_togpu = cupy.asarray
    cupy_ok = True
except ImportError:
    xpy_default = np
    xpy_special_default = None # scipy.special is used via scipy if needed
    identity_convert = lambda x: x
    identity_convert_togpu = lambda x: x
    cupy_ok = False

# 1. Try to find the legacy mvnun in known locations
try:
    from scipy.stats.mvn import mvnun
    _ORIGINAL_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        from scipy.stats._mvn import mvnun
        _ORIGINAL_AVAILABLE = True
    except (ImportError, AttributeError):
        _ORIGINAL_AVAILABLE = False

if not _ORIGINAL_AVAILABLE:
    def mvnun(lower, upper, mean, cov, maxpts=None, abseps=1e-5, releps=1e-5):
        """
        Modern fallback for scipy.stats.mvn.mvnun using multivariate_normal.cdf.
        Requires SciPy 1.10.0+ for the 'lower_limit' parameter.
        """
        dim = len(mean)
        if maxpts is None:
            maxpts = 2000 * dim
            
        p = multivariate_normal.cdf(
            x=upper,
            mean=mean,
            cov=cov,
            lower_limit=lower,
            maxpts=maxpts,
            abseps=abseps,
            releps=releps
        )
        # Return probability and a '0' for success (mimicking legacy API)
        return p, 0

        
from scipy.special import logsumexp
from . import multivariate_truncnorm as truncnorm
import itertools


def _xpy_logsumexp(a, axis=None):
    """Portable logsumexp.

    cupyx.scipy.special.logsumexp is only available in newer cupy releases;
    the CUDA 10.2 cupy build required by older (sm_30/Kepler) cards does not
    ship it. Implement the reduction directly with cupy primitives so the GPU
    path works regardless of cupy version, and fall back to scipy on CPU.
    """
    if cupy_ok:
        a = cupy.asarray(a)
        a_max = cupy.amax(a, axis=axis, keepdims=True)
        a_max = cupy.where(cupy.isfinite(a_max), a_max, cupy.zeros_like(a_max))
        out = cupy.log(cupy.sum(cupy.exp(a - a_max), axis=axis, keepdims=True)) + a_max
        if axis is None:
            return out.reshape(())
        return cupy.squeeze(out, axis=axis)
    return logsumexp(a, axis=axis)


# Symmetric (Hermitian) eigen-routines. cupy.linalg only provides the Hermitian
# variants (eigh/eigvalsh), not the general eig/eigvals. The matrices fed to
# _near_psd below are covariance/correlation matrices and hence symmetric, so
# the Hermitian routines are both correct and the only ones available on GPU.
if cupy_ok:
    _xpy_eigvals = cupy.linalg.eigvalsh
    _xpy_eig = cupy.linalg.eigh
else:
    _xpy_eigvals = np.linalg.eigvals
    _xpy_eig = np.linalg.eig


def gpu_logpdf(x, mean, cov, xpy):
    """
    GPU-compatible multivariate normal log-pdf.
    x: (n, d) array
    mean: (d,) array
    cov: (d, d) array

    Uses Cholesky + a generic linear solve (xpy.linalg.solve) so the same
    code path works for both numpy and cupy. Note: solve_triangular is
    NOT in numpy.linalg or cupy.linalg (only in scipy.linalg /
    cupyx.scipy.linalg), so we deliberately use the generic solver here.
    """
    d = mean.shape[0]
    diff = x - mean
    # Use cholesky for efficiency and stability. cupy.linalg has no LinAlgError
    # attribute (and cupy.linalg.cholesky returns NaN rather than raising on a
    # non-PSD input), so catch the numpy error type and also treat a NaN factor
    # as failure, falling back to an epsilon-regularized diagonal in both cases.
    eps = 1e-6 * xpy.eye(d)
    try:
        L = xpy.linalg.cholesky(cov)
        if bool(xpy.any(xpy.isnan(L))):
            L = xpy.linalg.cholesky(cov + eps)
    except np.linalg.LinAlgError:
        L = xpy.linalg.cholesky(cov + eps)

    # Solve L*y = diff^T => y = L^-1 * diff^T
    # diff is (n, d), so diff.T is (d, n)
    y = xpy.linalg.solve(L, diff.T)

    # quad_form = sum(y^2, axis=0)
    quad_form = xpy.sum(y**2, axis=0)

    # log_det = 2 * sum(log(diag(L)))
    log_det = 2.0 * xpy.sum(xpy.log(xpy.diag(L)))

    log_prob = -0.5 * (d * xpy.log(2 * xpy.pi) + log_det + quad_form)
    return log_prob

class estimator:
    '''
    Base estimator class for GMM

    Parameters
    ----------
    k : int
        Number of Gaussian components
    max_iters : int
        Maximum number of Expectation-Maximization iterations
    '''

    def __init__(self, k, max_iters=100, tempering_coeff=1e-8,adapt=None):
        self.k = k # number of gaussian components
        self.max_iters = max_iters # maximum number of iterations to convergence
        self.means = [None] * k
        self.covariances =[None] * k
        self.weights = [None] * k
        self.adapt = [None] * k
        if adapt:
            self.adapt = adapt
        self.d = None
        self.p_nk = None
        self.log_prob = None
        self.cov_avg_ratio = 0.05
        self.epsilon = 1e-4
        self.tempering_coeff = tempering_coeff
        self.xpy = xpy_default
        self.identity_convert = identity_convert
        self.identity_convert_togpu = identity_convert_togpu

    def _initialize(self, n, sample_array, log_sample_weights=None):
        p_weights = self.xpy.exp(log_sample_weights - self.xpy.max(log_sample_weights)).flatten()
        p_weights[self.xpy.isnan(p_weights)] = 0 # zero out the nan weights
        p_weights /= self.xpy.sum(p_weights)
        self.means = sample_array[self.xpy.random.choice(n, self.k, p=p_weights.astype(sample_array.dtype)), :]
        self.covariances = [self.xpy.identity(self.d)] * self.k
        self.weights = self.xpy.ones(self.k) / self.k
        self.adapt = [True] * self.k

    def _e_step(self, n, sample_array, log_sample_weights=None):
        '''
        Expectation step
        '''
        if log_sample_weights is None:
            log_sample_weights = self.xpy.zeros(n)
        p_nk = self.xpy.empty((n, self.k))
        for index in range(self.k):
            mean = self.means[index]
            cov = self.covariances[index]
            log_p = self.xpy.log(self.weights[index])
            
            if cupy_ok:
                log_pdf = gpu_logpdf(sample_array, mean, cov, self.xpy)
            else:
                log_pdf = multivariate_normal.logpdf(x=sample_array, mean=mean, cov=cov, allow_singular=True)
                
            p_nk[:,index] = log_pdf + log_p # (16.1.5)
            
        # Use cupy or scipy for logsumexp
        p_xn = _xpy_logsumexp(p_nk, axis=1)

        self.p_nk = p_nk - p_xn[:,self.xpy.newaxis] # (16.1.5)
        # normalize log sample weights as well, before modifying things with them
        ls_sum = _xpy_logsumexp(log_sample_weights)

        self.p_nk += log_sample_weights[:,self.xpy.newaxis]  - ls_sum

        self.log_prob = self.xpy.sum(p_xn + log_sample_weights)

    def _m_step(self, n, sample_array):
        '''
        Maximization step
        '''
        p_nk = self.xpy.exp(self.p_nk)
        weights = self.xpy.sum(p_nk, axis=0)   # weight of a single component
        for index in range(self.k):
          if self.adapt[index]:
            # (16.1.6)
            w = weights[index]   # should be 1 for a single component, note
            p_k = p_nk[:,index]
            mean = self.xpy.sum(self.xpy.multiply(sample_array, p_k[:,self.xpy.newaxis]), axis=0)
            mean /= w
            self.means[index] = mean
            # (16.1.6)
            diff = sample_array - mean
            cov = self.xpy.dot((p_k[:,self.xpy.newaxis] * diff).T, diff) / w
            self.covariances[index] = self._near_psd(cov)
            # (16.17)
        weights /= self.xpy.sum(p_nk[:,self.adapt])
        weights /= self.xpy.sum(weights)
        self.weights = weights


    def _tol(self, n):
        '''
        Scale tolerance with number of dimensions, number of components, and
        number of samples
        '''
        return (self.d * self.k * n) * 10e-4

    def _near_psd(self, x):
        '''
        Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix
        '''
        n = x.shape[0]
        var_list = self.xpy.array([self.xpy.sqrt(x[i,i]) for i in range(n)])
        # Use broadcasting for y instead of nested list comprehension
        y = x / (var_list[:, None] * var_list[None, :])
        while True:
            epsilon = self.epsilon
            if self.xpy.min(_xpy_eigvals(y)) > epsilon:
                return x

            var_list = self.xpy.array([self.xpy.sqrt(x[i,i]) for i in range(n)])
            y = x / (var_list[:, None] * var_list[None, :])

            eigval, eigvec = _xpy_eig(y)
            val = self.xpy.maximum(eigval, epsilon)
            vec = eigvec

            # Standard PSD projection:
            val_psd = self.xpy.maximum(eigval, epsilon)
            near_corr = vec @ self.xpy.diag(val_psd) @ vec.T
            
            # Re-scale back to covariance
            near_cov = near_corr * (var_list[:, None] * var_list[None, :])
            
            if self.xpy.isreal(near_cov).all():
                break
            else:
                x = near_cov.real
        return near_cov
    
    def fit(self, sample_array, log_sample_weights):
        '''
        Fit the model to data
        '''
        n, self.d = sample_array.shape
        self._initialize(n, sample_array, log_sample_weights)
        prev_log_prob = 0
        self.log_prob = float('inf')
        count = 0
        while abs(self.log_prob - prev_log_prob) > self._tol(n) and count < self.max_iters:
            prev_log_prob = self.log_prob
            self._e_step(n, sample_array, log_sample_weights)
            self._m_step(n, sample_array)
            count += 1
        for index in range(self.k):
            cov = self.covariances[index]
            cov = (cov + self.tempering_coeff * self.xpy.eye(self.d)) / (1 + self.tempering_coeff)
            self.covariances[index] = cov

    def print_params(self):
        '''
        Prints the model's parameters in an easily-readable format
        '''
        # Convert to numpy for printing
        means_np = [self.identity_convert(m) for m in self.means]
        covs_np = [self.identity_convert(c) for c in self.covariances]
        weights_np = self.identity_convert(self.weights)
        
        if self.d ==1:
            print("GMM:   component wt mean std ")
        for i in range(self.k):
            mean = means_np[i]
            cov = covs_np[i]
            weight = weights_np[i]
            if self.d >1:
                print('________________________________________\n')
                print('Component', i)
                print('Mean')
                print(mean)
                print('Covaraince')
                print(cov)
                print('Weight')
                print(weight, '\n')
            else:
                print(i, weight, mean[0], np.sqrt(cov[0,0]))


class gmm:
    '''
    More sophisticated implementation built on top of estimator class
    '''

    def __init__(self, k, bounds, max_iters=1000,epsilon=None,tempering_coeff=1e-8):
        self.k = k
        self.bounds = bounds
        self.max_iters = max_iters
        self.means = [None] * k
        self.covariances =[None] * k
        self.weights = [None] * k
        self.adapt = [True] * k
        self.d = None
        self.p_nk = None
        self.log_prob = None
        self.N = 0
        self.epsilon =epsilon
        if self.epsilon is None:
            self.epsilon = 1e-6
        else:
            self.epsilon=epsilon
        self.tempering_coeff = tempering_coeff
        self.xpy = xpy_default
        self.identity_convert = identity_convert
        self.identity_convert_togpu = identity_convert_togpu

    def _normalize(self, samples):
        n, d = samples.shape
        out = self.xpy.empty((n, d))
        for i in range(d):
            [llim, rlim] = self.bounds[i]
            out[:,i] = (2.0 * samples[:,i] - (rlim + llim)) / (rlim - llim)
        return out

    def _unnormalize(self, samples):
        n, d = samples.shape
        out = self.xpy.empty((n, d))
        for i in range(d):
            [llim, rlim] = self.bounds[i]
            out[:,i] = 0.5 * ((rlim - llim) * samples[:,i] + (llim + rlim))
        return out

    def fit(self, sample_array, log_sample_weights=None):
        '''
        Fit the model to data
        '''
        self.N, self.d = sample_array.shape
        if log_sample_weights is None:
            log_sample_weights = self.xpy.zeros(self.N)
        
        model = estimator(self.k, tempering_coeff=self.tempering_coeff,adapt=self.adapt)
        model.fit(self._normalize(sample_array), log_sample_weights)
        self.means = model.means
        self.covariances = model.covariances
        self.weights = model.weights
        self.p_nk = model.p_nk
        self.log_prob = model.log_prob

    def _match_components(self, new_model):
        '''
        Match components in new model to those in current model by minimizing the
        net Mahalanobis between all pairs of components
        '''
        orders = list(itertools.permutations(list(range(self.k)), self.k))
        distances = np.empty(len(orders))
        index = 0
        for order in orders:
            dist = 0
            i = 0
            for j in order:
                # These are likely small vectors, stay on CPU
                diff = self.identity_convert(new_model.means[j]) - self.identity_convert(self.means[i])
                cov_inv = np.linalg.inv(self.identity_convert(self.covariances[i]))
                temp_cov_inv = np.linalg.inv(self.identity_convert(new_model.covariances[j]))
                dist += np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                dist += np.sqrt(np.dot(np.dot(diff, temp_cov_inv), diff))
                i += 1
            distances[index] = dist
            index += 1
        return orders[np.argmin(distances)]

    def _merge(self, new_model, M):
        '''
        Merge corresponding components of new model and old model
        '''
        order = self._match_components(new_model)
        for i in range(self.k):
            j = order[i]
            old_mean = self.means[i]
            temp_mean = new_model.means[j]
            old_cov = self.covariances[i]
            temp_cov = new_model.covariances[j]
            old_weight = self.weights[i]
            temp_weight = new_model.weights[j]
            denominator = (self.N * old_weight) + (M * temp_weight)
            
            mean = (self.N * old_weight * old_mean) + (M * temp_weight * temp_mean)
            mean /= denominator
            
            cov1 = (self.N * old_weight * old_cov) + (M * temp_weight * temp_cov)
            cov1 /= denominator
            
            # outer product for means
            cov2 = (self.N * old_weight * self.xpy.outer(old_mean, old_mean)) + (M * temp_weight * self.xpy.outer(temp_mean, temp_mean))
            cov2 /= denominator
            
            cov = cov1 + cov2 - self.xpy.outer(mean, mean)
            cov = self._near_psd(cov)
            
            weight = denominator / (self.N + M)
            
            self.means[i] = mean
            self.covariances[i] = cov
            self.weights[i] = weight

    def _near_psd(self, x):
        '''
        Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix
        '''
        n = x.shape[0]
        var_list = self.xpy.array([self.xpy.sqrt(x[i,i]) for i in range(n)])
        y = x / (var_list[:, None] * var_list[None, :])
        while True:
            epsilon = self.epsilon
            if self.xpy.min(_xpy_eigvals(y)) > epsilon:
                return x

            var_list = self.xpy.array([self.xpy.sqrt(x[i,i]) for i in range(n)])
            y = x / (var_list[:, None] * var_list[None, :])

            eigval, eigvec = _xpy_eig(y)
            val_psd = self.xpy.maximum(eigval, epsilon)
            near_corr = eigvec @ self.xpy.diag(val_psd) @ eigvec.T
            near_cov = near_corr * (var_list[:, None] * var_list[None, :])
            if self.xpy.isreal(near_cov).all():
                break
            else:
                x = near_cov.real
        return near_cov

    def update(self, sample_array, log_sample_weights=None):
        '''
        Updates the model with new data without doing a full retraining.
        '''
        self.tempering_coeff /= 2
        new_model = estimator(self.k, self.max_iters, self.tempering_coeff)
        
        # Filter non-finite
        if log_sample_weights is not None:
            indx_ok = self.xpy.isfinite(log_sample_weights)
            s_filtered = sample_array[indx_ok]
            w_filtered = log_sample_weights[indx_ok]
        else:
            s_filtered = sample_array
            w_filtered = None
            
        new_model.fit(self._normalize(s_filtered), w_filtered)
        M, _ = sample_array.shape
        self._merge(new_model, M)
        self.N += M

    def score(self, sample_array,assume_normalized=True):
        '''
        Score samples under the current model.
        '''
        n, d = sample_array.shape
        scores = self.xpy.zeros(n)
        sample_array_norm = self._normalize(sample_array)
        
        # bounds_normalized
        bounds_norm = self._normalize(self.bounds.T).T
        normalization_constant = 0.
        
        for i in range(self.k):
            w = self.weights[i]
            mean = self.means[i]
            cov = self.covariances[i]
            
            if self.d > 1:
                if cupy_ok:
                    # Use gpu_logpdf and exponentiate
                    log_pdf = gpu_logpdf(sample_array_norm, mean, cov, self.xpy)
                    pdf = self.xpy.exp(log_pdf)
                else:
                    pdf = multivariate_normal.pdf(x=sample_array_norm, mean=mean, cov=cov, allow_singular=True)
                
                scores += pdf * w
                # mvnun is CPU only
                mean_cpu = self.identity_convert(mean)
                cov_cpu = self.identity_convert(cov)
                bounds_norm_cpu = self.identity_convert(bounds_norm)
                normalization_constant += w * mvnun(bounds_norm_cpu[:,0], bounds_norm_cpu[:,1], mean_cpu, cov_cpu)[0]
            else:
                sigma2 = cov[0,0]
                val = 1./self.xpy.sqrt(2*self.xpy.pi*sigma2) * self.xpy.exp( - 0.5*( sample_array_norm[:,0] - mean[0])**2/sigma2)
                scores += val * w
                
                mean_cpu = self.identity_convert(mean)[0]
                sigma_cpu = self.identity_convert(np.sqrt(sigma2))
                bounds_norm_cpu = self.identity_convert(bounds_norm[0])
                my_cdf = norm(loc=mean_cpu, scale=sigma_cpu).cdf
                normalization_constant += w * (my_cdf(bounds_norm_cpu[1]) - my_cdf(bounds_norm_cpu[0]))
        
        scores /= normalization_constant
        vol = self.xpy.prod(self.bounds[:,1] - self.bounds[:,0])
        scores *= (2.0**self.d) / vol
        return scores

    def sample(self, n, use_bounds=True):
        '''
        Draw samples from the current model.

        Note the model's means/covariances are stored in *normalized* coordinates
        (the [-1, 1] image of self.bounds under self._normalize). Samples are
        therefore drawn in normalized coordinates and then unnormalized back to
        the original coordinate frame before being returned, matching the
        pre-port behavior expected by MonteCarloEnsemble._sample().
        '''
        # Sampling is kept on CPU for stability (truncnorm is CPU-only)
        means_np = [self.identity_convert(m) for m in self.means]
        covs_np = [self.identity_convert(c) for c in self.covariances]
        weights_np = self.identity_convert(self.weights)

        # truncnorm bounds must match the coordinate frame of the model
        # parameters (mean/cov), which is normalized [-1, 1].
        bounds_normalized = np.empty((self.d, 2))
        bounds_normalized[:, 0] = -1.0
        bounds_normalized[:, 1] = 1.0

        sample_array_np = np.empty((n, self.d))
        start = 0
        for component in range(self.k):
            w = weights_np[component]
            mean = means_np[component]
            cov = covs_np[component]
            num_samples = int(n * w)
            if component == self.k - 1:
                end = n
            else:
                end = start + num_samples
            try:
                if not use_bounds:
                    sample_array_np[start:end] = np.random.multivariate_normal(mean, cov, end - start)
                else:
                    sample_array_np[start:end] = truncnorm.sample(mean, cov, bounds_normalized, end - start)
                start = end
            except Exception as e:
                print('Exiting due to non-positive-semidefinite', e)
                raise Exception("gmm covariance not positive-semidefinite")

        # Move to xpy and unnormalize back to original [llim, rlim] coordinates,
        # so callers receive samples in the same frame as self.bounds.
        sample_array_xpy = self.identity_convert_togpu(sample_array_np)
        return self._unnormalize(sample_array_xpy)

    def print_params(self):
        '''
        Prints the model's parameters in an easily-readable format
        '''
        means_np = [self.identity_convert(m) for m in self.means]
        covs_np = [self.identity_convert(c) for c in self.covariances]
        weights_np = self.identity_convert(self.weights)
        
        if self.d ==1:
            print("GMM:   component wt mean_correct mean_normed std_normed ")
        for i in range(self.k):
            mean = means_np[i]
            cov = covs_np[i]
            weight = weights_np[i]
            if self.d >1:
                print('________________________________________\n')
                print('Component', i)
                print('Mean (scaled and unscaled)')
                print(mean, self._unnormalize(np.array([mean])))
                print('Covariance')
                print(cov)
                print('Weight')
                print(weight, '\n')
            else:
                print(i, weight, self._unnormalize(np.array([mean]))[0,0], mean[0], np.sqrt(cov[0,0]))
