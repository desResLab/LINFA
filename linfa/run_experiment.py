import os
import torch
import numpy as np
import scipy as sp
from linfa.maf import MAF, RealNVP

# from TrivialModels import circuitTrivial
# from circuitModels import rcModel, rcrModel
# from highdimModels import Highdim

torch.set_default_tensor_type(torch.DoubleTensor)

class experiment:
    def __init__(self):
        self.name = "Experiment"
        self.flow_type = 'maf'  # str: Type of flow                                 default 'realnvp'
        self.n_blocks = 15  # int: Number of layers                             default 5
        self.hidden_size = 100  # int: Hidden layer size for MADE in each layer     default 100
        self.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
        self.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
        self.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
        self.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True
        self.sampling_interval = 200  # int: How often to sample from normalizing flow
        self.store_nf_interval = 1000

        self.input_size = 3  # int: Dimensionality of input                      default 2
        self.batch_size = 500  # int: Number of samples generated                  default 100
        self.true_data_num = 2  # double: number of true model evaluated        default 2
        self.n_iter = 25001  # int: Number of iterations                         default 25001
        self.lr = 0.003  # float: Learning rate                              default 0.003
        self.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9999

        self.run_nofas = True  # boo: decide if nofas is used
        self.log_interval = 10  # int: How often to show loss stat                  default 10
        self.calibrate_interval = 300  # int: How often to update surrogate model          default 1000
        self.budget = 216  # int: Total number of true model evaluation

        self.optimizer = 'Adam'  # str: type of optimizer used
        self.lr_scheduler = 'StepLR'  # str: type of lr scheduler used
        self.lr_step = 1000  # int: number of steps for lr step scheduler
        self.tol = 0.001  # float: tolerance for AdaAnn scheduler
        self.t0 = 0.01  # float: initial inverse temperature value
        self.N = 100  # int: number of sample points during annealing
        self.N_1 = 1000  # int: number of sample points at t=1
        self.T_0 = 500  # int: number of parameter updates at initial t0
        self.T = 5  # int: number of parameter updates during annealing
        self.T_1 = 5001  # int: number of parameter updates at t=1
        self.M = 1000  # int: number of sample points used to update temperature
        self.annealing = True  # boo: decide if annealing is used
        self.scheduler = 'AdaAnn'  # str: type of annealing scheduler used
        self.linear_step = 0.0001  # float: step size for constant annealing scheduler

        self.output_dir = './results/' + self.name
        self.results_file = 'results.txt'
        self.log_file = 'log.txt'
        self.samples_file = 'samples.txt'
        self.seed = 35435  # int: Random seed used
        self.n_sample = 5000  # int: Total number of iterations

        self.no_cuda = True
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not self.no_cuda else 'cpu')

        # Local pointer to the main components for inference
        self.transform        = None
        self.model            = None
        self.model_logdensity = None
        self.surrogate        = None

    # @property
    # def model_logdensity(self):
    #     return self.__model_logdensity
    #
    # @model_logdensity.setter
    # def model_logdensity(self, model_logdensity):
    #     self.__model_logdensity = lambda z: model_logdensity(z) + self.transformation.compute_log_jacob_func(z)

    def run(self):

        # Check is surrogate exists
        if self.run_nofas:
            if not os.path.exists(self.name + ".sur") or not os.path.exists(self.name + ".npz"):
                print("Abort: NoFAS enabled, without surrogate files. \nPlease include the following surrogate files in root directory.\n{}.sur and {}.npz".format(self.name, self.name))
                exit(0)
        # setup file ops
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Save a copy of the data in the result folder so it is handy
        if hasattr(self.model,'data'):
          np.savetxt(self.output_dir + '/' + self.name + '_data', self.model.data, newline="\n")

        # setup device
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda': torch.cuda.manual_seed(self.seed)

        # model
        if self.flow_type == 'maf':
            nf = MAF(self.n_blocks, self.input_size, self.hidden_size, self.n_hidden, None,
                     self.activation_fn, self.input_order, batch_norm=self.batch_norm_order)
        elif self.flow_type == 'realnvp':  # Under construction
            nf = RealNVP(self.n_blocks, self.input_size, self.hidden_size, self.n_hidden, None,
                         batch_norm=self.batch_norm_order)
        else:
            raise ValueError('Unrecognized model.')

        nf = nf.to(self.device)
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(nf.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(nf.parameters(), lr=self.lr)
        else:
            raise ValueError('Unrecognized optimizer.')

        if self.lr_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_step)
        elif self.lr_scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)
        else:
            raise ValueError('Unrecognized learning rate scheduler.')

        if self.annealing:
            i = 1
            prev_i = 1
            tvals = []
            t = self.t0
            dt = 0
            loglist = []
            while t < 1:
                t = min(1, t + dt)
                tvals = np.concatenate([tvals, np.array([t])])
                self.n_iter = self.T
                self.batch_size = self.N
                
                if t == self.t0:
                    # print('')
                    # print('--- Initial iterations at t0')
                    # print('')
                    self.n_iter = self.T_0
                if t == 1:
                    # print('')
                    # print('--- Final iterations at t=1')
                    # print('')
                    self.batch_size = self.N_1
                    self.n_iter = self.T_1

                while i < prev_i + self.n_iter:
                    print(prev_i + self.n_iter)
                    # print('--- VI NF at temperature t=%.3f' % (t))
                    print(t)
                    self.train(nf, optimizer, i, loglist, sampling=True, update=self.run_nofas, t=t)
                    if t == 1:
                        scheduler.step()
                    i += 1
                prev_i = i

                if self.scheduler == 'AdaAnn':
                    print('')
                    print('--- Updating temperature')
                    print('')
                    z0 = nf.base_dist.sample([self.M])
                    zk, _ = nf(z0)
                    log_qk = self.model_logdensity(zk)
                    dt = self.tol / torch.sqrt(log_qk.var())
                    dt = dt.detach().numpy()

                if self.scheduler == 'Linear':
                    dt = self.linear_step
        else:
            loglist = []
            for i in range(1, self.n_iter+1):                
                self.train(nf, optimizer, i, loglist, sampling=True, update=True)
                scheduler.step()

        print('')
        print('--- Simulation completed!')
        # rt.surrogate.surrogate_save() # Used for saving the resulting surrogate model

    def train(self, nf, optimizer, iteration, log, sampling=True, update=True, t=1):

        # Train the normalizing flow
        nf.train()

        x0 = nf.base_dist.sample([self.batch_size])
        xk, sum_log_abs_det_jacobians = nf(x0)

        # generate and save samples evaluation
        if sampling and iteration % self.sampling_interval == 0:
            print('--- Saving results at iteration '+str(iteration))
            x00 = nf.base_dist.sample([self.n_sample])
            xkk, _ = nf(x00)
            # Save surrogate grid
            if not(self.surrogate is None):
              np.savetxt(self.output_dir + '/' + self.name + '_grid_' + str(iteration), self.surrogate.grid_record.detach().numpy(), newline="\n")
            # Save log profile
            np.savetxt(self.output_dir + '/' + self.log_file, np.array(log), newline="\n")
            # Save transformed samples          
            np.savetxt(self.output_dir + '/' + self.name + '_samples_' + str(iteration), xkk.data.numpy(), newline="\n")
            # Save samples in the original space
            if not(self.transform is None):
              np.savetxt(self.output_dir + '/' + self.name + '_params_' + str(iteration), self.transform.forward(xkk).data.numpy(), newline="\n")
            else:
              np.savetxt(self.output_dir + '/' + self.name + '_params_' + str(iteration), xkk.data.numpy(), newline="\n")
            # Save log density at the same samples
            np.savetxt(self.output_dir + '/' + self.name + '_logdensity_' + str(iteration), self.model_logdensity(xkk).data.numpy(), newline="\n")
            # Save model outputs at the samples - If a model is defined
            if not(self.transform is None):
              stds = torch.abs(self.model.solve_t(self.model.defParam)) * self.model.stdRatio
              o00 = torch.randn(x00.size(0), self.model.data.shape[0])
              noise = o00*stds.repeat(o00.size(0),1)
              if self.surrogate:
                np.savetxt(self.output_dir + '/' + self.name + '_outputs_' + str(iteration), (self.surrogate.forward(xkk) + noise).data.numpy(), newline="\n")
              else:
                np.savetxt(self.output_dir + '/' + self.name + '_outputs_' + str(iteration), (self.model.solve_t(self.transform.forward(xkk)) + noise).data.numpy(), newline="\n")

        if torch.any(torch.isnan(xk)):
            print("Error: samples xk are nan at iteration " + str(iteration))
            print(xk)
            np.savetxt(self.output_dir + '/' + self.log_file, np.array(log), newline="\n")
            exit(-1)

        # updating surrogate model
        if iteration % self.calibrate_interval == 0 and update and self.surrogate.grid_record.size(0) < self.budget:
            xk0 = xk[:self.true_data_num, :].data.clone()            
            # print("\n")
            # print(list(self.surrogate.grid_record.size())[0])
            # print(xk0)
            self.surrogate.update(xk0, max_iters=6000)

        # Free energy bound
        loss = (- torch.sum(sum_log_abs_det_jacobians, 1) - t * self.model_logdensity(xk)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('VI NF: it: %7d | loss: %8.3e' % (iteration, loss.item()), end='\r')
        if iteration % self.log_interval == 0:
            print('VI NF (t=%5.3f): it: %7d | loss: %8.3e' % (t,iteration, loss.item()))
            # log.append([iteration, loss.item()] + list(torch.std(xk, dim=0).detach().numpy()))
            log.append([t, iteration, loss.item()])
        
        # Save state of normalizing flow layers
        if self.store_nf_interval > 0 and iteration % self.store_nf_interval == 0:
            torch.save(nf.state_dict(), self.output_dir + '/' + self.name + "_" + str(iteration) + ".nf")
