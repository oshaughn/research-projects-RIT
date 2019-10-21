# -*- coding: utf-8 -*-
'''
Gaussian Mixture Model
----------------------
Fit a Gaussian Mixture Model (GMM) to data and draw samples from it. Uses the
Expectation-Maximization algorithm.
'''


from six.moves import range

import numpy as np
from scipy.stats import multivariate_normal
#from scipy.misc import logsumexp
from scipy.special import logsumexp
from . import multivariate_truncnorm as truncnorm
import itertools


# Equation references are from Numerical Recipes for general GMM and
# https://www.cs.nmsu.edu/~joemsong/publications/Song-SPIE2005-updated.pdf for
# online updating features


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

    def __init__(self, k, max_iters=100, tempering_coeff=0.001):
        self.k = k # number of gaussian components
        self.max_iters = max_iters # maximum number of iterations to convergence
        self.means = [None] * k
        self.covariances =[None] * k
        self.weights = [None] * k
        self.d = None
        self.p_nk = None
        self.log_prob = None
        self.cov_avg_ratio = 0.05
        self.epsilon = 1e-4
        self.tempering_coeff = tempering_coeff

    def _initialize(self, n, sample_array, sample_weights=None):
        p_weights = (sample_weights / np.sum(sample_weights)).flatten()
        self.means = sample_array[np.random.choice(n, self.k, p=p_weights.astype(sample_array.dtype)), :]
        self.covariances = [np.identity(self.d)] * self.k
        self.weights = np.array([1.0 / self.k] * self.k)

    def _e_step(self, n, sample_array, sample_weights):
        '''
        Expectation step
        '''
        if sample_weights is None:
            log_sample_weights = np.zeros((n, 1))
        else:
            log_sample_weights = np.log(sample_weights)
        p_nk = np.empty((n, self.k))
        for index in range(self.k):
            mean = self.means[index]
            cov = self.covariances[index]
            log_p = np.log(self.weights[index])
            log_pdf = np.rot90([multivariate_normal.logpdf(x=sample_array, mean=mean, cov=cov, allow_singular=True)], -1) # (16.1.4)
            # note that allow_singular=True in the above line is probably really dumb and
            # terrible, but it seems to occasionally keep the whole thing from blowing up
            # so it stays for now
            p_nk[:,[index]] = log_pdf + log_p # (16.1.5)
        p_xn = logsumexp(p_nk, axis=1, keepdims=True) # (16.1.3)
        self.p_nk = p_nk - p_xn # (16.1.5)
        self.p_nk += log_sample_weights
        self.log_prob = np.sum(p_xn + log_sample_weights) # (16.1.2)

    def _m_step(self, n, sample_array):
        '''
        Maximization step
        '''
        p_nk = np.exp(self.p_nk)
        weights = np.sum(p_nk, axis=0)
        for index in range(self.k):
            # (16.1.6)
            w = weights[index]
            p_k = p_nk[:,[index]]
            mean = np.sum(np.multiply(sample_array, p_k), axis=0)
            mean /= w
            self.means[index] = mean
            # (16.1.6)
            diff = sample_array - mean
            cov = np.dot((p_k * diff).T, diff) / w
            # attempt to fix non-positive-semidefinite covariances
            self.covariances[index] = self._near_psd(cov)
            # (16.17)
        weights /= np.sum(p_nk)
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
        
        Code from here:
        https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
        '''
        n = x.shape[0]
        var_list = np.array([np.sqrt(x[i,i]) for i in range(n)])
        y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])
        while True:
            epsilon = self.epsilon
            if min(np.linalg.eigvals(y)) > epsilon:
                return x

            # Removing scaling factor of covariance matrix
            var_list = np.array([np.sqrt(x[i,i]) for i in range(n)])
            y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])

            # getting the nearest correlation matrix
            eigval, eigvec = np.linalg.eig(y)
            val = np.matrix(np.maximum(eigval, epsilon))
            vec = np.matrix(eigvec)
            T = 1/(np.multiply(vec, vec) * val.T)
            T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
            B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
            near_corr = B*B.T    

            # returning the scaling factors
            near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])
            if np.isreal(near_cov).all():
                break
            else:
                x = near_cov.real
        return near_cov
    
    def fit(self, sample_array, sample_weights):
        '''
        Fit the model to data

        Parameters
        ----------
        sample_array : np.ndarray
            Array of samples to fit
        sample_weights : np.ndarray
            Weights for samples
        '''
        n, self.d = sample_array.shape
        self._initialize(n, sample_array, sample_weights)
        prev_log_prob = 0
        self.log_prob = float('inf')
        count = 0
        while abs(self.log_prob - prev_log_prob) > self._tol(n) and count < self.max_iters:
            prev_log_prob = self.log_prob
            self._e_step(n, sample_array, sample_weights)
            self._m_step(n, sample_array)
            count += 1
        for index in range(self.k):
            cov = self.covariances[index]
            # temper
            cov = (cov + self.tempering_coeff * np.eye(self.d)) / (1 + self.tempering_coeff)
            self.covariances[index] = cov

    def print_params(self):
        '''
        Prints the model's parameters in an easily-readable format
        '''
        for i in range(self.k):
            mean = self.means[i]
            cov = self.covariances[i]
            weight = self.weights[i]
            print('________________________________________\n')
            print('Component', i)
            print('Mean')
            print(mean)
            print('Covaraince')
            print(cov)
            print('Weight')
            print(weight, '\n')


class gmm:
    '''
    More sophisticated implementation built on top of estimator class

    Includes functionality to update with new data rather than re-fit, as well
    as sampling and scoring of samples.

    Parameters
    ----------
    k : int
        Number of Gaussian components
    max_iters : int
        Maximum number of Expectation-Maximization iterations
    '''

    def __init__(self, k, max_iters=1000):
        self.k = k
        #self.tol = tol
        self.max_iters = max_iters
        self.means = [None] * k
        self.covariances =[None] * k
        self.weights = [None] * k
        self.d = None
        self.p_nk = None
        self.log_prob = None
        self.N = 0
        self.epsilon = 1e-4  # allow very strong correlations
        self.tempering_coeff = 0.01

    def fit(self, sample_array, sample_weights=None):
        '''
        Fit the model to data

        Parameters
        ----------
        sample_array : np.ndarray
            Array of samples to fit
        sample_weights : np.ndarray
            Weights for samples
        '''
        self.N, self.d = sample_array.shape
        if sample_weights is None:
            sample_weights = np.ones((self.N, 1))
        # just use base estimator
        model = estimator(self.k, tempering_coeff=self.tempering_coeff)
        model.fit(sample_array, sample_weights)
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
                # get Mahalanobis distance between current pair of components
                diff = new_model.means[j] - self.means[i]
                cov_inv = np.linalg.inv(self.covariances[i])
                temp_cov_inv = np.linalg.inv(new_model.covariances[j])
                dist += np.sqrt(np.dot(np.dot(diff, cov_inv), diff))
                dist += np.sqrt(np.dot(np.dot(diff, temp_cov_inv), diff))
                i += 1
            distances[index] = dist
            index += 1
        return orders[np.argmin(distances)] # returns order which gives minimum net Mahalanobis distance

    def _merge(self, new_model, M):
        '''
        Merge corresponding components of new model and old model

        Refer to paper linked at the top of this file

        M is the number of samples that the new model was fit using
        '''
        order = self._match_components(new_model)
        for i in range(self.k):
            j = order[i] # get corresponding component
            old_mean = self.means[i]
            temp_mean = new_model.means[j]
            old_cov = self.covariances[i]
            temp_cov = new_model.covariances[j]
            old_weight = self.weights[i]
            temp_weight = new_model.weights[j]
            denominator = (self.N * old_weight) + (M * temp_weight) # this shows up a lot so just compute it once
            # start equation (6)
            mean = (self.N * old_weight * old_mean) + (M * temp_weight * temp_mean)
            mean /= denominator
            # start equation (7)
            cov1 = (self.N * old_weight * old_cov) + (M * temp_weight * temp_cov)
            cov1 /= denominator
            cov2 = (self.N * old_weight * old_mean * old_mean.T) + (M * temp_weight * temp_mean * temp_mean.T)
            cov2 /= denominator
            cov = cov1 + cov2 - mean * mean.T
            # check for positive-semidefinite
            cov = self._near_psd(cov)
            # start equation (8)
            weight = denominator / (self.N + M)
            # update everything
            self.means[i] = mean
            self.covariances[i] = cov
            self.weights[i] = weight

    def _near_psd(self, x):
        '''
	Calculates the nearest postive semi-definite matrix for a correlation/covariance matrix

        Code from here:
        https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix
        '''
        n = x.shape[0]
        var_list = np.array([np.sqrt(x[i,i]) for i in range(n)])
        y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])
        while True:
            epsilon = self.epsilon
            if min(np.linalg.eigvals(y)) > epsilon:
                return x

            # Removing scaling factor of covariance matrix
            var_list = np.array([np.sqrt(x[i,i]) for i in range(n)])
            y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])

            # getting the nearest correlation matrix
            eigval, eigvec = np.linalg.eig(y)
            val = np.matrix(np.maximum(eigval, epsilon))
            vec = np.matrix(eigvec)
            T = 1/(np.multiply(vec, vec) * val.T)
            T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
            B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
            near_corr = B*B.T    

            # returning the scaling factors
            near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in range(n)] for j in range(n)])
            if np.isreal(near_cov).all():
                break
            else:
                x = near_cov.real
        return near_cov

    def update(self, sample_array, sample_weights=None):
        '''
        Updates the model with new data without doing a full retraining.

        Parameters
        ----------
        sample_array : np.ndarray
            Array of samples to fit
        sample_weights : np.ndarray
            Weights for samples
        '''
        self.tempering_coeff /= 2
        new_model = estimator(self.k, self.max_iters, self.tempering_coeff)
        new_model.fit(sample_array, sample_weights)
        M, _ = sample_array.shape
        self._merge(new_model, M)
        self.N += M

    def score(self, sample_array, bounds=None):
        '''
        Score samples (i.e. calculate likelihood of each sample) under the current
        model.

        Parameters
        ----------
        sample_array : np.ndarray
            Array of samples to fit
        bounds : np.ndarray
            Bounds for samples, used for renormalizing scores
        '''
        n, _ = sample_array.shape
        scores = np.zeros((len(sample_array), 1))
        for i in range(self.k):
            w = self.weights[i]
            mean = self.means[i]
            cov = self.covariances[i]
            scores += np.rot90([multivariate_normal.pdf(x=sample_array, mean=mean, cov=cov, allow_singular=True)], -1) * w
            # note that allow_singular=True in the above line is probably really dumb and
            # terrible, but it seems to occasionally keep the whole thing from blowing up
            # so it stays for now
        if bounds is not None:
            # we need to renormalize the PDF
            # to do this we sample from a full distribution (i.e. without truncation) and use the
            # fraction of samples that fall inside the bounds to renormalize
            full_sample_array = self.sample(n)
            llim = np.rot90(bounds[:,[0]])
            rlim = np.rot90(bounds[:,[1]])
            n1 = np.greater(full_sample_array, llim).all(axis=1)
            n2 = np.less(full_sample_array, rlim).all(axis=1)
            normalize = np.array(np.logical_and(n1, n2)).flatten()
            m = float(np.sum(normalize)) / n
            scores /= m
        return scores

    def sample(self, n, bounds=None):
        '''
        Draw samples from the current model, either with or without bounds

        Parameters
        ----------
        n : int
            Number of samples to draw
        bounds : np.ndarray
            Bounds for samples
        '''
        sample_array = np.empty((n, self.d))
        start = 0
        for component in range(self.k):
            w = self.weights[component]
            mean = self.means[component]
            cov = self.covariances[component]
            num_samples = int(n * w)
            if component == self.k -1:
                end = n
            else:
                end = start + num_samples
            try:
                if bounds is None:
                    sample_array[start:end] = np.random.multivariate_normal(mean, cov, end - start)
                else:
                    sample_array[start:end] = truncnorm.sample(mean, cov, bounds, end - start)
                start = end
            except:
                print('Exiting due to non-positive-semidefinite')
                exit()
        return sample_array

    def print_params(self):
        '''
        Prints the model's parameters in an easily-readable format
        '''
        for i in range(self.k):
            mean = self.means[i]
            cov = self.covariances[i]
            weight = self.weights[i]
            print('________________________________________\n')
            print('Component', i)
            print('Mean')
            print(mean)
            print('Covaraince')
            print(cov)
            print('Weight')
            print(weight, '\n')
