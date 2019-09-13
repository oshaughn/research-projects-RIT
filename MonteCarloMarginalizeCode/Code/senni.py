'''
Single Event Neural Network Interpolator
'''

import torch
import os

class Net(torch.nn.Module):
            '''
            Network architecture definition
            '''
        
            def __init__(self, n_inputs, hlayer_size, n_outputs):
                super(Net, self).__init__()
                self.linear1 = torch.nn.Linear(n_inputs, hlayer_size)
                self.linear2 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.linear3 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.linear4 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.linear5 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.linear6 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.linear7 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.linear8 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.linear9 = torch.nn.Linear(hlayer_size, n_outputs)
      
            def forward(self, x):
                x = torch.selu(self.linear1(x))
                x = torch.selu(self.linear2(x))
                x = torch.selu(self.linear3(x))
                x = torch.selu(self.linear4(x))
                x = torch.selu(self.linear5(x))
                x = torch.selu(self.linear6(x))
                x = torch.selu(self.linear7(x))
                x = torch.selu(self.linear8(x))
                x = self.linear9(x)
                return x

class Interpolator(object): # interpolator


      def __init__(self, input, target, errors, frac=0.1, test_frac=0.1, hlayer_size=32,
                   epochs=100, learning_rate=1e-2, betas=(0.9, 0.99), eps=1e-2, epochs_per_lr=20, 
                   lr_divisions=5, lr_frac=1./3., batch_size=128, shuffle=True, working_dir='.', loss_func='mape'):

            '''
            :input: Input column vector with each column representing one input with size (n_samples, n_dim)
            :target: Target column vector of shape (n_samples, 1)
            :frac: The fraction of the input vector taken to be the size of the validation set
            :test_frac: The fraction of the input vector taken to be the size of the test set
            :hlayer_size: Number of hidden neurons in each hidden layer
            :epochs: Number of training epochs
            :learning_rate: Initial learning rate for the optimizer
            :betas: Decay rates for moving averages of gradient (see Adam optimizer documentation)
            :eps: Small number that prevents division by zero in the optimizer
            :epochs_per_lr: Number of epochs that must pass before the learning rate is reduced by lr_frac
            :lr_divisions: Number of times that the learning rate is reduced by lr_frac
            :lr_frac: Fraction by which the learning_rate is multiplied every epochs_per_lr epochs
            :batch_size: Size of the batches used during training
            :shuffle: Boolean deciding whether the data batches are shuffled prior to training
            '''

            import numpy as np
            import os

            try: os.mkdir(working_dir+'/models')
            except: pass

            self.select_device()

            self.working_dir = working_dir
            self.n_inputs = np.size(input, 1)
            self.n_outputs = np.size(target, 1)
            self.hlayer_size = hlayer_size

            self.epochs = epochs
            self.batch_size = batch_size
            self.shuffle = shuffle

            input_train, target_train, errors_train, input_valid, target_valid, errors_valid, input_test, target_test, errors_test \
            = self.set_separation(input, target, errors, frac, test_frac)

            target_train, self.target_mu, self.target_sigma = self.preprocessing(target_train)
            target_valid, _, _ = self.preprocessing(target_valid, self.target_mu, self.target_sigma)
            target_test, _, _ = self.preprocessing(target_test, self.target_mu, self.target_sigma)

            errors_train, err_mu, err_sigma = self.preprocessing(errors_train)
            errors_valid, _, _ = self.preprocessing(errors_valid, mu=err_mu, sigma=err_sigma)
            errors_test, _, _ = self.preprocessing(errors_test, mu=err_mu, sigma=err_sigma)

            for dim in xrange(self.n_inputs):
                  input_train[:, dim], mu, sigma = self.preprocessing(input_train[:, dim])
                  input_valid[:, dim], _, _ = self.preprocessing(input_valid[:, dim], mu=mu, sigma=sigma)
                  input_test[:, dim], _, _ = self.preprocessing(input_test[:, dim], mu=mu, sigma=sigma)

                  min, max = 2*np.min(input_train[:, dim]), 2*np.max(input_train[:, dim])

                  fakesamples = np.linspace(min, max, 0.5*input_train.shape[0])

                  bad_idxs = np.where((fakesamples > (min/2.)) & (fakesamples < (max/2.)))
                  fakesamples = np.delete(fakesamples, bad_idxs)

                  otherdims = np.random.normal(size=(fakesamples.shape[0], self.n_inputs-1))
                  fakesamples = np.insert(otherdims, dim, fakesamples.T, axis=1)

                  zerolnLs = np.zeros(fakesamples.shape[0])
                  zerolnLs = zerolnLs[:, np.newaxis]

                  unitysigmas = np.ones(fakesamples.shape[0])
                  unitysigmas = unitysigmas[:, np.newaxis]
                  
                  input_train = np.append(input_train, fakesamples, axis=0)
                  target_train = np.append(target_train, zerolnLs, axis=0)
                  errors_train = np.append(errors_train, unitysigmas, axis=0)

            self.input_train = torch.from_numpy(input_train).float().to(self.device)
            self.input_valid = torch.from_numpy(input_valid).float().to(self.device)
            self.input_test = torch.from_numpy(input_test).float().to(self.device)

            self.target_train = torch.from_numpy(target_train).float().to(self.device)
            self.target_valid = torch.from_numpy(target_valid).float().to(self.device)
            self.target_test = torch.from_numpy(target_test).float().to(self.device)

            self.errors_train = torch.from_numpy(errors_train).float().to(self.device)
            self.errors_valid = torch.from_numpy(errors_valid).float().to(self.device)
            self.errors_test = torch.from_numpy(errors_test).float().to(self.device)

            self.network_init()
            self.loss_func = loss_func
            self.optim_init(learning_rate=learning_rate, betas=betas, eps=eps)
            self.sched_init(epochs_per_lr=epochs_per_lr, lr_divisions=lr_divisions, lr_frac=lr_frac)

            return

      def set_separation(self, input, target, errors, frac, test_frac):
            '''
            Separates the input data array into training, validation, and test sets. Size of validation/test sets is based on fraction of whole input array.
            Both validation and test set have the same size fraction frac unless test_frac is specified.
            '''
            import numpy as np

            idxs = np.random.choice(target.shape[0], int(target.shape[0]*frac), replace=False) # validation set indices

            input_valid, target_valid, errors_valid = input[idxs, :], target[idxs, :], errors[idxs, :]
            input_train, target_train, errors_train = np.delete(input, idxs, 0), np.delete(target, idxs, 0), np.delete(errors, idxs, 0)

            idxs2 = np.random.choice(target_train.shape[0], int(target.shape[0]*test_frac), replace=False) # test set indices

            input_test, target_test, errors_test = input_train[idxs2, :], target_train[idxs2, :], errors_train[idxs2, :]
            input_train, target_train, errors_train = np.delete(input_train, idxs2, 0), np.delete(target_train, idxs2, 0), np.delete(errors_train, idxs2, 0)

            return input_train, target_train, errors_train, input_valid, target_valid, errors_valid, input_test, target_test, errors_test

      def preprocessing(self, data, mu=0, sigma=0):
            '''
            Processes the data such that it is in the proper format (mean-subtracted and sigma-divided) for input into the SELU NN
            If no mu/sigma provided, takes the mean and standard deviation of the data
            '''
            # NOTE: for the validation and test sets, the mean and std of the TRAINING SET are used, not of the respective sets
            import numpy as np

            if not mu and not sigma:
                  data_mu, data_sigma = np.mean(data), np.std(data)
            else:
                  data_mu = mu
                  data_sigma = sigma
            if data_sigma == 0:
                data_sigma = 1  # special case
            data = (data - data_mu)/data_sigma

            return data, data_mu, data_sigma

      def select_device(self):
            '''
            Runs on GPU if available, otherwise runs on CPU
            '''
            if torch.cuda.is_available():
                  device = torch.device('cuda:0')
            else:
                  device = torch.device('cpu')

            self.device = device

      def network_init(self):
            '''
            Network initialization given some number of inputs, outputs, and the size of the hidden layers
            '''
            def weights_init(layer):
                  '''
                  Initialization of weights such that they follow a normal distribution (mu=0, sigma=1)
                  '''
                  import numpy as np
  
                  if isinstance(layer, torch.nn.Linear):
                        torch.nn.init.normal_(layer.weight, std=1./np.sqrt(self.hlayer_size))
            
            if torch.cuda.is_available():
                  self.net = Net(self.n_inputs, self.hlayer_size, self.n_outputs).cuda().apply(weights_init)
            else:
                  self.net = Net(self.n_inputs, self.hlayer_size, self.n_outputs).apply(weights_init)

      def optim_init(self, learning_rate, betas, eps):
            '''
            Optimizer initialization, specifically Adam for now, using network parameters and specified optimizer parameters
            '''
            self.optim = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=betas, eps=eps)

      def sched_init(self, epochs_per_lr, lr_divisions, lr_frac):
            '''
            Scheduler initialization which selects the epoch intervals at which the learning rate is taken to be frac times the current learning rate.
            Epochs_per_lr indicates how many epochs transpire before the learning rate is divided. 
            Lr_divisions indicates how many times the learning rate is multiplied by frac.
            Frac is a fraction that multiplies the current learning rate to reduce it i.e. initial_lr*(frac^n) after n divisions
            '''
            import numpy as np

            self.epochs_per_lr = epochs_per_lr
            self.lr_divisions = lr_divisions
            self.sched = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=np.linspace(self.epochs_per_lr, self.lr_divisions*self.epochs_per_lr, self.lr_divisions), gamma=lr_frac)
      
      def MAPEloss(self, output, target, mu, sigma):
            '''
            Mean absolute percentage error loss function
            '''
            return 1./output.shape[0]*torch.sum(torch.abs((output-target)/(target+mu/sigma)))

      def reducedchisquareloss(self, output, target, error):

            return torch.sum((output-target)**2/(error)**2)/(output.shape[0]-self.n_inputs)

      def train(self,debug=True):
            '''
            Training on the inputs
            Inputs and target should be of the form [array1, array2, ...]
            '''
            import numpy as np
            from torch.autograd import Variable
            import torch.utils.data as Data

            try: os.system('rm ' + working_dir + '/models/bestmodel.pt')
            except: pass

            dataset = Data.TensorDataset(self.input_train, self.target_train, self.errors_train)

            loader = Data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)

            validation_threshold = 1e10

            for epoch in np.arange(1, self.epochs+1):

                  for step, (batch_input, batch_target, batch_error) in enumerate(loader):

                        batch_inputs = Variable(batch_input)
                        batch_targets = Variable(batch_target)
                        batch_errors = Variable(batch_error)

                        prediction = self.net(batch_inputs)

                        if self.loss_func == 'mape':
                            loss = self.MAPEloss(prediction, batch_targets, self.target_mu, self.target_sigma)
                        if self.loss_func == 'chi2':
                            loss = self.reducedchisquareloss(prediction, batch_targets, batch_errors)

                        self.optim.zero_grad()
                        loss.backward()      
                        self.optim.step()

                  if self.loss_func == 'mape':
                      validation_loss = self.MAPEloss(self.net(self.input_valid), self.target_valid, self.target_mu, self.target_sigma)
                  if self.loss_func == 'chi2':
                      validation_loss = self.reducedchisquareloss(self.net(self.input_valid), self.target_valid, self.errors_valid)

                  if validation_loss < validation_threshold:

                        validation_threshold = validation_loss

                        torch.save({
                        'model_state_dict': self.net.state_dict(),
                        }, self.working_dir+'/models/bestmodel.pt')

                  if epoch % self.epochs_per_lr == 0 and epoch <= self.lr_divisions*self.epochs_per_lr:
                      if os.path.exists(self.working_dir+'/models/bestmodel.pt'):
                        savedmodel = torch.load(self.working_dir+'/models/bestmodel.pt')
                        self.net.load_state_dict(savedmodel['model_state_dict'])

                  self.sched.step()
                  
                  if debug:
                      print "Epoch %d out of %d complete" % (epoch, self.epochs), '  loss ', validation_loss

            if self.loss_func == 'mape':
                self.train_loss = self.MAPEloss(self.net(self.input_train), self.target_train, self.target_mu, self.target_sigma)
                self.valid_loss = self.MAPEloss(self.net(self.input_valid), self.target_valid, self.target_mu, self.target_sigma)
                if debug:
                    print "   Loss (MAPE) ", self.train_loss, self.valid_loss
            if self.loss_func == 'chi2':
                self.train_loss = self.reducedchisquareloss(self.net(self.input_train), self.target_train, self.errors_train)
                self.valid_loss = self.reducedchisquareloss(self.net(self.input_valid), self.target_valid, self.errors_valid)
                if debug:
                    print "   Loss (chi2) ", self.train_loss, self.valid_loss

      def save(self, filename):
            '''
            Saves the model
            '''
            torch.save({
                  'model_state_dict': self.net.state_dict(),
                  'optimizer_state_dict': self.optim.state_dict(),
                  }, filename)
            print 'Model saved under %s with training loss %f and validation loss %f' % (filename, self.train_loss, self.valid_loss)

      def load(self, filename):
            '''
            Loads a saved model
            '''
            savedmodel = torch.load(filename)
            self.net.load_state_dict(savedmodel['model_state_dict'])
            self.optim.load_state_dict(savedmodel['optimizer_state_dict'])
            print 'Loaded model %s' % filename
            return

      def evaluate(self, input):
            '''
            Uses the currently loaded model to evaluate an input array
            '''
            import numpy as np

            for dim in xrange(np.size(input, 1)):
                  input[:, dim], _, _ = self.preprocessing(input[:, dim])

            self.net.eval()

            output = self.net(torch.from_numpy(input).float().to(self.device)).detach().numpy()[:,0] # convert back to 1d output
            output *= self.target_sigma
            output += self.target_mu

            return output
