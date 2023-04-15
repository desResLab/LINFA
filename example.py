from run_experiment import experiment
from NoFAS import Surrogate
import torch
import random
import numpy as np
from models.circuitModels import rcrModel

# Experiment Setting
exp = experiment()
exp.name = "Experiment"
exp.flow_type = 'maf'  # str: Type of flow                                 default 'realnvp'
exp.n_blocks = 15  # int: Number of layers                             default 5
exp.hidden_size = 100  # int: Hidden layer size for MADE in each layer     default 100
exp.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
exp.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
exp.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
exp.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True
exp.sampling_interval = 200  # int: How often to sample from normalizing flow

exp.input_size = 3  # int: Dimensionality of input                      default 2
exp.batch_size = 500  # int: Number of samples generated                  default 100
exp.true_data_num = 2  # double: number of true model evaluated        default 2
exp.n_iter = 25001  # int: Number of iterations                         default 25001
exp.lr = 0.003  # float: Learning rate                              default 0.003
exp.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9999
exp.log_interval = 10  # int: How often to show loss stat                  default 10
exp.calibrate_interval = 300  # int: How often to update surrogate model          default 1000
exp.budget = 216  # int: Total number of true model evaluation

exp.output_dir = './results/' + exp.name
exp.results_file = 'results.txt'
exp.log_file = 'log.txt'
exp.samples_file = 'samples.txt'
exp.seed = random.randint(0, 10 ** 9)  # int: Random seed used
exp.n_sample = 5000  # int: Total number of iterations
exp.no_cuda = True

exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

# Model Setting
cycleTime = 1.07
totalCycles = 10
forcing = np.loadtxt('source/data/inlet.flow')
model = rcrModel(cycleTime, totalCycles, forcing)  # RCR Model Defined
model.data = np.loadtxt('source/data/data_rcr.txt')
exp.surrogate = Surrogate("RCR", lambda x: model.solve_t(model.transform(x)), exp.input_size, 3,
                          torch.Tensor([[-7, 7], [-7, 7], [-7, 7]]), 20)
exp.surrogate.surrogate_load()


# Define log density
def log_density(x, model, surrogate):
    batch_size = x.size(0)
    x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
    adjust = torch.log(1.0 - torch.tanh(x1 / 7.0 * 3.0) ** 2) \
             + torch.log(1.0 - torch.tanh(x2 / 7.0 * 3.0) ** 2) \
             + x3 / 7 * 3

    modelOut = surrogate.forward(x)
    return - model.evalNegLL_t(modelOut).reshape(batch_size, 1) + adjust


exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate)
exp.run()
