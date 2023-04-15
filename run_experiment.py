import os
import torch
import numpy as np
import scipy as sp
from maf import MAF, RealNVP
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

        self.input_size = 3  # int: Dimensionality of input                      default 2
        self.batch_size = 500  # int: Number of samples generated                  default 100
        self.true_data_num = 2  # double: number of true model evaluated        default 2
        self.n_iter = 25001  # int: Number of iterations                         default 25001
        self.lr = 0.003  # float: Learning rate                              default 0.003
        self.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9999
        self.log_interval = 10  # int: How often to show loss stat                  default 10
        self.calibrate_interval = 300  # int: How often to update surrogate model          default 1000
        self.budget = 216  # int: Total number of true model evaluation

        self.output_dir = './results/' + self.name
        self.results_file = 'results.txt'
        self.log_file = 'log.txt'
        self.samples_file = 'samples.txt'
        self.seed = 35435  # int: Random seed used
        self.n_sample = 5000  # int: Total number of iterations
        self.no_cuda = True

        self.device = torch.device('cuda:0' if torch.cuda.is_available() and not self.no_cuda else 'cpu')
        # self.model_solve = None
        self.model_logdensity = None
        self.surrogate = None

    def run(self):
        # setup file ops
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

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
        optimizer = torch.optim.RMSprop(nf.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)

        loglist = []
        for i in range(self.n_iter):
            scheduler.step()
            self.train_NoFAS(nf, optimizer, i, loglist, sampling=True, update=True)

        # rt.surrogate.surrogate_save() # Used for saving the resulting surrogate model
        np.savetxt(self.output_dir + '/grid_trace.txt', self.surrogate.grid_record.detach().numpy())
        np.savetxt(self.output_dir + '/' + self.log_file, np.array(loglist), newline="\n")

    def train_NoFAS(self, nf, optimizer, iteration, log, sampling=True, update=True):
        nf.train()
        x0 = nf.base_dist.sample([self.batch_size])
        xk, sum_log_abs_det_jacobians = nf(x0)

        # generate samples on the way
        if sampling and iteration % self.sampling_interval == 0:
            x00 = nf.base_dist.sample([self.n_sample])
            xkk, _ = nf(x00)
            np.savetxt(self.output_dir + '/samples' + str(iteration), xkk.data.numpy(), newline="\n")

        if torch.any(torch.isnan(xk)):
            print("Error: " + str(iteration))
            print(xk)
            np.savetxt(self.output_dir + '/' + self.log_file, np.array(log), newline="\n")
            exit(-1)

        # updating surrogate model
        if iteration % self.calibrate_interval == 0 and update and self.surrogate.grid_record.size(0) < self.budget:
            xk0 = xk[:self.true_data_num, :].data.clone()
            print("\n")
            print(list(self.surrogate.grid_record.size())[0])
            print(xk0)
            self.surrogate.update(xk0, max_iters=6000)

        # Free energy bound
        loss = (- torch.sum(sum_log_abs_det_jacobians, 1) - self.model_logdensity(xk)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("{}\t{}".format(iteration, loss.item()), end='\r')
        if iteration % self.log_interval == 0:
            print("{}\t{}".format(iteration, loss.item()))
            log.append([iteration, loss.item()] + list(torch.std(xk, dim=0).detach().numpy()))


