'''
Single Event Neural Network Interpolator
'''

import torch
import os

class Net(torch.nn.Module):
            '''
            Network architecture definition
            '''
        
            def __init__(self, n_inputs, hlayer_size, n_outputs, p_drop):
                super(Net, self).__init__()
                self.linear1 = torch.nn.Linear(n_inputs, hlayer_size)
                self.linear2 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.dropout1 = torch.nn.Dropout(p=p_drop)
                self.linear3 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.dropout2 = torch.nn.Dropout(p=p_drop)
                self.linear4 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.dropout3 = torch.nn.Dropout(p=p_drop)
                self.linear5 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.dropout4 = torch.nn.Dropout(p=p_drop)
                self.linear6 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.dropout5 = torch.nn.Dropout(p=p_drop)
                self.linear7 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.dropout6 = torch.nn.Dropout(p=p_drop)
                self.linear8 = torch.nn.Linear(hlayer_size, hlayer_size)
                self.dropout7 = torch.nn.Dropout(p=p_drop)
                self.linear9 = torch.nn.Linear(hlayer_size, n_outputs)
      
            def forward(self, x):
                x = torch.selu(self.linear1(x))
                x = self.dropout1(torch.selu(self.linear2(x)))
                x = self.dropout2(torch.selu(self.linear3(x)))
                x = self.dropout3(torch.selu(self.linear4(x)))
                x = self.dropout4(torch.selu(self.linear5(x)))
                x = self.dropout5(torch.selu(self.linear6(x)))
                x = self.dropout6(torch.selu(self.linear7(x)))
                x = self.dropout7(torch.selu(self.linear8(x)))
                x = self.linear9(x)
                return x

class Interpolator(object): # interpolator


      def __init__(self, input, target, errors, frac=0.1, test_frac=0.1, hlayer_size=32, p_drop=0, regularize=False,
                   epochs=100, learning_rate=1e-2, betas=(0.9, 0.99), eps=1e-2, weight_decay=1e-6,
                   epochs_per_lr=20, lr_divisions=5, lr_frac=1./3., batch_size=128, shuffle=True, 
                   working_dir='.', loss_func='chi2', no_pad=False):

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
            self.p_drop = p_drop
            self.regularize=regularize

            self.epochs = epochs
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.store_mu_x =[]
            self.store_sigma_x=[]

            self.store_mu_x = np.zeros(len(input[0]))
            self.store_sigma_x = np.zeros(len(input[0]))

            for dim in xrange(self.n_inputs):
                  self.store_mu_x[dim] = np.mean(input[:,dim])
                  self.store_sigma_x[dim] = np.std(input[:,dim])
                  print( " Computing scaling factors on raw data ", dim, self.store_mu_x[dim], self.store_sigma_x[dim])

            input_train, target_train, errors_train, input_valid, target_valid, errors_valid, input_test, target_test, errors_test \
            = self.set_separation(input, target, errors, frac, test_frac)

            # set the scaling of the 'y' variable to the ORIGINAL problem, not allowing for any padding
            # also use the WHOLE sample to get the scaling factors
            target_copy = np.copy(target)
            _, self.target_mu, self.target_sigma = self.preprocessing(target_copy)
            print( " Computing scaling factors on scaled output ", self.target_mu, self.target_sigma)

            # Create copies to extend, based on input values
            #  - do *not* try to edit these on the fly, since we need to loop through the *original* sample
            input_train_revised = np.array(input_train)
            target_train_revised = np.array(target_train)
            errors_train_revised = np.array(errors_train)
            
            p_epsilon = 1e-3
            
            if not no_pad:
              for dim in xrange(self.n_inputs):
                  # take the max and min values of the array
                  #   - don't use actual min and max, in case of crazy outliers that will leave large gaps in NN coverage
                  true_min,true_max = np.percentile(input_train[:,dim],p_epsilon*100),np.percentile(input_train[:,dim],100*(1-p_epsilon))
                  # find target range
                  #   - choice depends on if signs are different!
                  #   - if same sign, use a factor 
                  test_min, test_max = 0.5*true_min,2*true_max
                  if test_min*test_max < 0: # two different signs for this parameter
                      test_min, test_max = 2*true_min, 2*true_max
                  # Creating fake samples. Note size scaled so TOTAL number of fake samples scales with total sample size
#                  print " Extending range to ", test_min, test_max, input_train[:,dim], "; cutoffs  in physical coordinates are ", self.store_mu_x[dim]+self.store_sigma_x[dim]*np.array([test_min,true_min, true_max, test_max])
                  n_samples_to_add = int(0.3*input_train.shape[0]/(1.0*self.n_inputs))
                  fakesamples = np.random.uniform(test_min, test_max, n_samples_to_add)
                  # remove all the values which occur in the dataset (keep np.where(newv<min) and np.where(newv>max))
                  bad_idxs = np.where((fakesamples > (true_min)) & (fakesamples < (true_max)))
                  fakesamples = np.delete(fakesamples, bad_idxs)
 #                 print " Adding fake samples to dimension ",dim, len(fakesamples), n_samples_to_add, "note mean, width of scaled variable are ",np.mean(input_train[:,dim]),np.std(input_train[:,dim])
                  # set random values (draw from normal) for other parameters and then set likelihood = 0
                  otherdims = np.random.normal(size=(fakesamples.shape[0], self.n_inputs-1))
                  # insert the non-random fakesamples into the relevant dimension index
                  fakesamples = np.insert(otherdims, dim, fakesamples.T, axis=1)
                  zerolnLs = np.ones(fakesamples.shape[0])*1e-5
                  zerolnLs = zerolnLs[:, np.newaxis]
                  unitysigmas = np.ones(fakesamples.shape[0])
                  unitysigmas = unitysigmas[:, np.newaxis]
                  input_train_revised = np.append(input_train_revised, fakesamples, axis=0)
                  target_train_revised = np.append(target_train_revised, zerolnLs, axis=0)
                  errors_train_revised = np.append(errors_train_revised, 0.001*unitysigmas, axis=0) # assume these zeros for lnL are well-known!

            input_train = input_train_revised
            target_train = target_train_revised
            errors_train = errors_train_revised

            # scale the output scale *including* the padded points
            target_train, self.target_mu, self.target_sigma = self.preprocessing(target_train)
            target_valid, _, _ = self.preprocessing(target_valid, self.target_mu, self.target_sigma)
            target_test, _, _ = self.preprocessing(target_test, self.target_mu, self.target_sigma)

            for dim in xrange(self.n_inputs):
                  mu, sigma = self.store_mu_x[dim], self.store_sigma_x[dim]
                  input_train[:, dim], _, _ = self.preprocessing(input_train[:, dim],mu=mu,sigma=sigma)
                  input_valid[:, dim], _, _ = self.preprocessing(input_valid[:, dim], mu=mu, sigma=sigma)
                  input_test[:, dim], _, _ = self.preprocessing(input_test[:, dim], mu=mu, sigma=sigma)

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
            self.optim_init(learning_rate=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
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
            data_copy =np.zeros(data.shape)
            data_copy = (data - data_mu)/data_sigma

            return data_copy, data_mu, data_sigma

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
                  self.net = Net(self.n_inputs, self.hlayer_size, self.n_outputs, self.p_drop).cuda().apply(weights_init)
            else:
                  self.net = Net(self.n_inputs, self.hlayer_size, self.n_outputs, self.p_drop).apply(weights_init)

      def optim_init(self, learning_rate, betas, eps, weight_decay):
            '''
            Optimizer initialization, specifically Adam for now, using network parameters and specified optimizer parameters
            '''
            self.optim = torch.optim.Adam(self.net.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

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
            t_max = torch.max(target)
            return 1./output.shape[0]*torch.sum(torch.abs((output-target)*torch.exp(-0.2*torch.abs(t_max-output))/(target+mu/sigma)))

      def reducedchisquareloss(self, output, target, error):

            weight_mag = 0
    
            #for param in self.net.parameters():
            #      param = torch.pow(param, 2)
            #      param = torch.sum(param)
            #      weight_mag += param

            for layer in self.net.children():
                  if not isinstance(layer, torch.nn.Linear): continue
                  weights = layer.weight.data
                  weights = torch.pow(weights, 2)
                  weights = torch.sum(weights)
                  weight_mag += weights
      
            t_max = torch.max(target)
            return torch.sum((output-target)**2*torch.exp(-0.2*torch.abs(t_max-output))/(error)**2)/(output.shape[0]-self.n_inputs) + 1e-6*weight_mag

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

                  # L2 regularization on *weights*. 
                  # ALREADY implemented in the optimizer with weight_decay: redundant to do both!
                  if self.regularize:
                      reg_loss = None
                      for param in self.net.named_parameters():
                          param_name =param[0]
                          param=param[1]
# Don't regularize last layer (as that sets overall scale of function, rather than structure).
#                      if 'linear9' in param_name:
#                          continue
#  do not regularize bias terms?
#                          if 'bias' in param_name:
#                              continue
                          if reg_loss is None:
                              reg_loss = 0.5 * torch.sum(param**2)
                          else:
                              reg_loss = reg_loss + 0.5 * param.norm(2)**2
                      validation_loss += 0.9*reg_loss
                  

                  if validation_loss < 1e-6:
                      print( "   ... should we stop? ")
                      break

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
                      print( "Epoch %d out of %d complete" % (epoch, self.epochs), '  loss ', validation_loss)

            if self.loss_func == 'mape':
                self.train_loss = self.MAPEloss(self.net(self.input_train), self.target_train, self.target_mu, self.target_sigma)
                self.valid_loss = self.MAPEloss(self.net(self.input_valid), self.target_valid, self.target_mu, self.target_sigma)
                if debug:
                    print( "   Loss (MAPE) ", self.train_loss, self.valid_loss)
            if self.loss_func == 'chi2':
                self.train_loss = self.reducedchisquareloss(self.net(self.input_train), self.target_train, self.errors_train)
                self.valid_loss = self.reducedchisquareloss(self.net(self.input_valid), self.target_valid, self.errors_valid)
                if debug:
                    print( "   Loss (chi2) ", self.train_loss, self.valid_loss)

      def save(self, filename):
            '''
            Saves the model
            '''
            torch.save({
                  'model_state_dict': self.net.state_dict(),
                  'optimizer_state_dict': self.optim.state_dict(),
                  }, filename)
            print( 'Model saved under %s with training loss %f and validation loss %f' % (filename, self.train_loss, self.valid_loss))

      def load(self, filename):
            '''
            Loads a saved model
            '''
            savedmodel = torch.load(filename)
            self.net.load_state_dict(savedmodel['model_state_dict'])
            self.optim.load_state_dict(savedmodel['optimizer_state_dict'])
            print( 'Loaded model %s' % filename)
            return

      def evaluate(self, input):
            '''
            Uses the currently loaded model to evaluate an input array
            '''
            import numpy as np

            input_copy = np.zeros(input.shape)
            for dim in xrange(np.size(input, 1)):
                  input_copy[:, dim], _, _ = self.preprocessing(input[:, dim],mu=self.store_mu_x[dim],sigma=self.store_sigma_x[dim])

            self.net.eval()

            output = self.net(torch.from_numpy(input_copy).float().to(self.device)).detach().cpu().numpy()[:,0] # convert back to 1d output
            output *= self.target_sigma
            output += self.target_mu

            return output

class AdaptiveInterpolator(Interpolator):

      def __init__(self, input, target, errors, frac=0.1, test_frac=0.1, hlayer_size=32, p_drop=0,
                   epochs=100, learning_rate=1e-2, betas=(0.9, 0.99), eps=1e-2, weight_decay=1e-6,
                   epochs_per_lr=20, lr_divisions=5, lr_frac=1./3., batch_size=128, shuffle=True, 
                   working_dir='.', loss_func='chi2', no_pad=False):

            self.input = input
            self.target = target
            self.errors = errors
            self.frac = frac
            self.test_frac = test_frac
            self.hlayer_size = hlayer_size
            self.p_drop=p_drop
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.betas = betas
            self.eps = eps
            self.weight_decay = weight_decay
            self.epochs_per_lr = epochs_per_lr
            self.lr_divisions = lr_divisions
            self.lr_frac = lr_frac
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.working_dir = working_dir
            self.loss_func = loss_func
            self.no_pad = no_pad

            Interpolator.__init__(self, self.input, self.target, self.errors, frac=self.frac, test_frac=self.test_frac, hlayer_size=self.hlayer_size, p_drop=self.p_drop,
                   epochs=self.epochs, learning_rate=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, epochs_per_lr=self.epochs_per_lr, 
                   lr_divisions=self.lr_divisions, lr_frac=self.lr_frac, batch_size=self.batch_size, shuffle=self.shuffle, working_dir=self.working_dir,
                   loss_func=self.loss_func, no_pad=self.no_pad)

      def train(self):

            import numpy as np

            print('Using adaptive training...')

            super(AdaptiveInterpolator, self).train()

            while self.valid_loss < 1e-3:

                  self.hlayer_size /= 2

                  print('Decreasing until reasonable layer size reached, currently trying %d' % self.hlayer_size)

                  Interpolator.__init__(self, self.input, self.target, self.errors, frac=self.frac, test_frac=self.test_frac, hlayer_size=self.hlayer_size, p_drop=self.p_drop,
                   epochs=self.epochs, learning_rate=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, epochs_per_lr=self.epochs_per_lr, 
                   lr_divisions=self.lr_divisions, lr_frac=self.lr_frac, batch_size=self.batch_size, shuffle=self.shuffle, working_dir=self.working_dir,
                   loss_func=self.loss_func, no_pad=self.no_pad)

                  super(AdaptiveInterpolator, self).train()

            if self.valid_loss < 1: self.hlayer_size /= 2

            ceiling = 2*self.hlayer_size

            floor = self.hlayer_size

            layer_sizes = np.linspace(floor, ceiling, 5)

            losses = np.zeros_like(layer_sizes)

            for size in layer_sizes:

                  self.hlayer_size = int(size)

                  print('Looping through layer sizes, currently at %d' % self.hlayer_size)

                  Interpolator.__init__(self, self.input, self.target, self.errors, frac=self.frac, test_frac=self.test_frac, hlayer_size=self.hlayer_size, p_drop=self.p_drop,
                   epochs=self.epochs, learning_rate=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, epochs_per_lr=self.epochs_per_lr, 
                   lr_divisions=self.lr_divisions, lr_frac=self.lr_frac, batch_size=self.batch_size, shuffle=self.shuffle, working_dir=self.working_dir,
                   loss_func=self.loss_func, no_pad=self.no_pad)

                  super(AdaptiveInterpolator, self).train()

                  losses[np.where(layer_sizes==size)] = np.abs(self.valid_loss.detach().cpu().numpy() - 1)

            self.hlayer_size = int(layer_sizes[np.argmin(losses)])

            self.losses = losses
            self.layer_sizes = layer_sizes

            Interpolator.__init__(self, self.input, self.target, self.errors, frac=self.frac, test_frac=self.test_frac, hlayer_size=self.hlayer_size, p_drop=self.p_drop,
                   epochs=self.epochs, learning_rate=self.learning_rate, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, epochs_per_lr=self.epochs_per_lr, 
                   lr_divisions=self.lr_divisions, lr_frac=self.lr_frac, batch_size=self.batch_size, shuffle=self.shuffle, working_dir=self.working_dir,
                   loss_func=self.loss_func, no_pad=self.no_pad)

            print( '-------OPTIMAL HIDDEN LAYERS FOUND TO BE %d, RE-TRAINING WITH THIS SETTING-------' % self.hlayer_size)

            super(AdaptiveInterpolator, self).train()
