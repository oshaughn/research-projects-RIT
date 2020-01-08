import torch
import gpytorch
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, likelihood):
		super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
		self.mean_module = gpytorch.means.ConstantMean()
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Interpolator(object):

	def __init__(self, input, target, epochs=100, learning_rate=1e-1, betas=(0.9, 0.99), eps=1e-2, weight_decay=1e-6):

		for dim in range(input.shape[1]):
			input[:, dim], _, _ = self.preprocessing(input[:, dim])

		target, self.target_mu, self.target_sigma = self.preprocessing(target)

		self.input_train, self.target_train = torch.from_numpy(input).float(), torch.from_numpy(target).float()

		self.epochs = epochs
		self.gp_init()
		self.optim_init(learning_rate, betas, eps, weight_decay)


	def preprocessing(self, data, mu=0, sigma=0):

		if not mu and not sigma:
			data_mu, data_sigma = np.mean(data), np.std(data)
		else:
			data_mu = mu
			data_sigma = sigma
		if data_sigma == 0:
			data_sigma = 1
		data_copy = np.zeros(data.shape)
		data_copy = (data - data_mu)/data_sigma

		return data_copy, data_mu, data_sigma

	def optim_init(self, learning_rate, betas, eps, weight_decay):

		self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)	

	def gp_init(self):

		self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
		self.model = ExactGPModel(self.input_train, self.target_train, self.likelihood)

	def train(self):

		self.model.train()
		self.likelihood.train()

		mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

		for epoch in range(self.epochs):

			self.optim.zero_grad()

			output = self.model(self.input_train)

			loss = -mll(output, self.target_train)
			loss.backward()

			self.optim.step()

			print('Epoch = %d, loss = %f, noise = %f' % (epoch, loss.item(), self.model.likelihood.noise.item()))

	def evaluate(self, input):

		self.model.eval()
		self.likelihood.eval()

		for dim in range(input.shape[1]):
			input[:, dim], mu, sigma = self.preprocessing(input[:, dim])

		input = torch.from_numpy(input).float()

		output = self.model(input)
		output = output.sample()
		output *= self.target_sigma
		output += self.target_mu

		return output.numpy()

