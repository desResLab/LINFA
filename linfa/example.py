from run_experiment import experiment
from nofas import Surrogate
import torch
import random
import numpy as np
import os

def trivial_example():
    from models.TrivialModels import Trivial
    exp = experiment()
    exp.name = "Trivial"
    exp.flow_type = 'realnvp'  # str: Type of flow                                 default 'realnvp'
    exp.n_blocks = 5  # int: Number of layers                             default 5
    exp.hidden_size = 100  # int: Hidden layer size for MADE in each layer     default 100
    exp.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
    exp.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
    exp.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
    exp.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True
    exp.sampling_interval = 200  # int: How often to sample from normalizing flow

    exp.input_size = 2  # int: Dimensionality of input                      default 2
    exp.batch_size = 200  # int: Number of samples generated                  default 100
    exp.true_data_num = 2  # double: number of true model evaluated        default 2
    exp.n_iter = 25001  # int: Number of iterations                         default 25001
    exp.lr = 0.002  # float: Learning rate                              default 0.003
    exp.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9999
    exp.log_interval = 10  # int: How often to show loss stat                  default 10

    exp.run_nofas = True
    exp.annealing = False
    exp.calibrate_interval = 1000  # int: How often to update surrogate model          default 1000
    exp.budget = 64  # int: Total number of true model evaluation

    exp.output_dir = './results/' + exp.name
    exp.results_file = 'results.txt'
    exp.log_file = 'log.txt'
    exp.samples_file = 'samples.txt'
    exp.seed = random.randint(0, 10 ** 9)  # int: Random seed used
    exp.n_sample = 5000  # int: Total number of iterations
    exp.no_cuda = True

    exp.optimizer = 'RMSprop'
    exp.lr_scheduler = 'ExponentialLR'

    exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

    # Model Setting
    model = Trivial()
    model.data = np.loadtxt('source/data/data_trivial.txt')
    exp.surrogate = Surrogate("Trivial", model.solve_t, 2, 2, [[0, 6], [0, 6]], 20)
    if exp.run_nofas:
        if not os.path.exists(exp.name + ".sur") or not os.path.exists(exp.name + ".npz"):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz are not detected. ".format(exp.name))
            print("Training Surrogate ...")
            exp.surrogate.gen_grid(gridnum=4)
            exp.surrogate.pre_train(120000, 0.03, 0.9999, 500, store=True)
    exp.surrogate.surrogate_load()

    # Define log density
    def log_density(x, model, surrogate):
        stds = torch.abs(model.solve_t(model.defParam)) * model.stdRatio
        Data = torch.tensor(model.data)
        modelOut = surrogate.forward(x)
        # Eval LL
        ll1 = -0.5 * np.prod(model.data.shape) * np.log(2.0 * np.pi)
        ll2 = (-0.5 * model.data.shape[1] * torch.log(torch.prod(stds))).item()
        ll3 = - 0.5 * torch.sum(torch.sum((modelOut.unsqueeze(0) - Data.t().unsqueeze(1)) ** 2, dim=0) / stds[0] ** 2,
                                dim=1, keepdim=True)
        negLL = -(ll1 + ll2 + ll3)
        return -negLL

    exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate)
    exp.run()


def highdim_example():
    from models.highdimModels import Highdim
    exp = experiment()
    exp.name = "Highdim"
    exp.flow_type = 'realnvp'  # str: Type of flow                                 default 'realnvp'
    exp.n_blocks = 15  # int: Number of layers                             default 5
    exp.hidden_size = 100  # int: Hidden layer size for MADE in each layer     default 100
    exp.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
    exp.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
    exp.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
    exp.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True
    exp.sampling_interval = 200  # int: How often to sample from normalizing flow

    exp.input_size = 5  # int: Dimensionality of input                      default 2
    exp.batch_size = 150  # int: Number of samples generated                  default 100
    exp.true_data_num = 12  # double: number of true model evaluated        default 2
    exp.n_iter = 25000  # int: Number of iterations                         default 25001
    exp.lr = 0.0005  # float: Learning rate                              default 0.003
    exp.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9999
    exp.log_interval = 10  # int: How often to show loss stat                  default 10

    exp.run_nofas = True
    exp.annealing = False
    exp.calibrate_interval = 250  # int: How often to update surrogate model          default 1000
    exp.budget = 1023  # int: Total number of true model evaluation

    exp.output_dir = './results/' + exp.name
    exp.results_file = 'results.txt'
    exp.log_file = 'log.txt'
    exp.samples_file = 'samples.txt'
    exp.seed = random.randint(0, 10 ** 9)  # int: Random seed used
    exp.n_sample = 5000  # int: Total number of iterations
    exp.no_cuda = True

    exp.optimizer = 'RMSprop'
    exp.lr_scheduler = 'ExponentialLR'

    exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

    # Model Setting
    model = Highdim()
    model.data = np.loadtxt('source/data/data_highdim.txt')
    exp.surrogate = Surrogate("highdim", lambda x: model.solve_t(model.transform(x)), model.input_num, model.output_num,
                              torch.Tensor([[-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3]]), 20)
    if exp.run_nofas:
        if not os.path.exists(exp.name + ".sur") or not os.path.exists(exp.name + ".npz"):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz are not detected. ".format(exp.name))
            print("Training Surrogate ...")
            exp.surrogate.gen_grid(gridnum=4)
            exp.surrogate.pre_train(120000, 0.03, 0.9999, 500, store=True)
    exp.surrogate.surrogate_load()

    # Define log density
    def log_density(x, model, surrogate):
        batch_size = x.size(0)
        adjust = torch.sum(x, dim=1, keepdim=True)
        modelOut = surrogate.forward(x)
        return - model.evalNegLL_t(modelOut).reshape(batch_size, 1) + adjust

    exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate)
    exp.run()


def RC_example():
    from models.circuitModels import rcModel
    exp = experiment()
    exp.name = "RC"
    exp.flow_type = 'maf'  # str: Type of flow                                 default 'realnvp'
    exp.n_blocks = 5  # int: Number of layers                             default 5
    exp.hidden_size = 100  # int: Hidden layer size for MADE in each layer     default 100
    exp.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
    exp.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
    exp.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
    exp.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True
    exp.sampling_interval = 200  # int: How often to sample from normalizing flow

    exp.input_size = 2  # int: Dimensionality of input                      default 2
    exp.batch_size = 250  # int: Number of samples generated                  default 100
    exp.true_data_num = 2  # double: number of true model evaluated        default 2
    exp.n_iter = 25000  # int: Number of iterations                         default 25001
    exp.lr = 0.003  # float: Learning rate                              default 0.003
    exp.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9999
    exp.log_interval = 10  # int: How often to show loss stat                  default 10

    exp.run_nofas = True
    exp.annealing = False
    exp.calibrate_interval = 1000  # int: How often to update surrogate model          default 1000
    exp.budget = 64  # int: Total number of true model evaluation

    exp.output_dir = './results/' + exp.name
    exp.results_file = 'results.txt'
    exp.log_file = 'log.txt'
    exp.samples_file = 'samples.txt'
    exp.seed = random.randint(0, 10 ** 9)  # int: Random seed used
    exp.n_sample = 5000  # int: Total number of iterations
    exp.no_cuda = True

    exp.optimizer = 'RMSprop'
    exp.lr_scheduler = 'ExponentialLR'

    exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

    # Model Setting
    cycleTime = 1.07
    totalCycles = 10
    forcing = np.loadtxt('source/data/inlet.flow')
    model = rcModel(cycleTime, totalCycles, forcing)  # RCR Model Defined
    model.data = np.loadtxt('source/data/data_rc.txt')
    exp.surrogate = Surrogate("RC", lambda x: model.solve_t(model.transform(x)), exp.input_size, 3,
                              torch.Tensor([[-7, 7], [-7, 7]]), 20)
    if exp.run_nofas:
        if not os.path.exists(exp.name + ".sur") or not os.path.exists(exp.name + ".npz"):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz are not detected. ".format(exp.name))
            print("Training Surrogate ...")
            exp.surrogate.gen_grid(gridnum=4)
            exp.surrogate.pre_train(120000, 0.03, 0.9999, 500, store=True)
    exp.surrogate.surrogate_load()

    # Define log density
    def log_density(x, model, surrogate):
        batch_size = x.size(0)
        x1, x2 = torch.chunk(x, chunks=3, dim=1)
        adjust = torch.log(1.0 - torch.tanh(x1 / 7.0 * 3.0) ** 2) + x2 / 7 * 3

        modelOut = surrogate.forward(x)
        return - model.evalNegLL_t(modelOut).reshape(batch_size, 1) + adjust

    exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate)
    exp.run()


def RCR_example():
    from models.circuitModels import rcrModel
    exp = experiment()
    exp.name = "RCR"
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
    exp.n_iter = 25000  # int: Number of iterations                         default 25001
    exp.lr = 0.003  # float: Learning rate                              default 0.003
    exp.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9999
    exp.log_interval = 10  # int: How often to show loss stat                  default 10

    exp.run_nofas = True
    exp.annealing = False
    exp.calibrate_interval = 300  # int: How often to update surrogate model          default 1000
    exp.budget = 216  # int: Total number of true model evaluation

    exp.output_dir = './results/' + exp.name
    exp.results_file = 'results.txt'
    exp.log_file = 'log.txt'
    exp.samples_file = 'samples.txt'
    exp.seed = random.randint(0, 10 ** 9)  # int: Random seed used
    exp.n_sample = 5000  # int: Total number of iterations
    exp.no_cuda = True

    exp.optimizer = 'RMSprop'
    exp.lr_scheduler = 'ExponentialLR'

    exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

    # Model Setting
    cycleTime = 1.07
    totalCycles = 10
    forcing = np.loadtxt('source/data/inlet.flow')
    model = rcrModel(cycleTime, totalCycles, forcing)  # RCR Model Defined
    model.data = np.loadtxt('source/data/data_rcr.txt')
    exp.surrogate = Surrogate("RCR", lambda x: model.solve_t(model.transform(x)), exp.input_size, 3,
                              torch.Tensor([[-7, 7], [-7, 7], [-7, 7]]), 20)
    if exp.run_nofas:
        if not os.path.exists(exp.name + ".sur") or not os.path.exists(exp.name + ".npz"):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz are not detected. ".format(exp.name))
            print("Training Surrogate ...")
            exp.surrogate.gen_grid(gridnum=4)
            exp.surrogate.pre_train(120000, 0.03, 0.9999, 500, store=True)
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


def AdaANN_example():
    from run_experiment import experiment
    import torch
    import random
    import numpy as np
    import pandas as pd

    # Experiment Setting
    exp = experiment()
    exp.name = "AdaANN"
    exp.flow_type = 'realnvp'  # str: Type of flow                                 default 'realnvp'
    exp.n_blocks = 10  # int: Number of layers                             default 5
    exp.hidden_size = 20  # int: Hidden layer size for MADE in each layer     default 100
    exp.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
    exp.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
    exp.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
    exp.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True
    exp.sampling_interval = 200  # int: How often to sample from normalizing flow

    exp.input_size = 10  # int: Dimensionality of input                      default 2
    exp.batch_size = 100  # int: Number of samples generated                  default 100
    exp.n_iter = 1000  # int: Number of iterations                         default 25001
    exp.lr = 0.005  # float: Learning rate                              default 0.003
    exp.log_interval = 10  # int: How often to show loss stat                  default 10
    exp.run_nofas = False
    exp.annealing = True

    exp.optimizer = 'Adam'  # str: type of optimizer used
    exp.lr_scheduler = 'StepLR'  # str: type of lr scheduler used
    exp.lr_step = 500  # int: number of steps for lr step scheduler
    exp.tol = 0.4  # float: tolerance for AdaAnn scheduler
    exp.t0 = 0.001  # float: initial inverse temperature value
    exp.N = 100  # int: number of sample points during annealing
    exp.N_1 = 400  # int: number of sample points at t=1
    exp.T_0 = 500  # int: number of parameter updates at initial t0
    exp.T = 3  # int: number of parameter updates during annealing
    exp.T_1 = 2000  # int: number of parameter updates at t=1
    exp.M = 1000  # int: number of sample points used to update temperature
    exp.annealing = True  # boo: decide if annealing is used
    exp.scheduler = 'AdaAnn'  # str: type of annealing scheduler used

    exp.output_dir = './results/' + exp.name
    exp.results_file = 'results.txt'
    exp.log_file = 'log.txt'
    exp.samples_file = 'samples.txt'
    exp.seed = 35435  # int: Random seed used
    exp.n_sample = 5000  # int: Total number of iterations
    exp.no_cuda = True

    exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

    # Model Setting
    data_set = pd.read_csv('source/data/D1000.csv')
    data = torch.tensor(data_set.values)

    def log_density(params, d):
        def targetPosterior(b, x):
            return b[0] * torch.sin(np.pi * x[:, 0] * x[:, 1]) + b[1] ** 2 * (x[:, 2] - b[2]) ** 2 + x[:, 3] * b[3] + x[
                                                                                                                      :,
                                                                                                                      4] * \
                   b[4] + x[:, 5] * b[5] + x[:, 6] * b[6] + x[:, 7] * b[7] + x[:, 8] * b[8] + x[:, 9] * b[9]

        f = torch.zeros(len(params))

        for i in range(len(params)):
            y_out = targetPosterior(params[i], d)
            val = torch.linalg.norm(y_out - d[:, 10])
            f[i] = -val ** 2 / 2

        return f

    exp.model_logdensity = lambda x: log_density(x, data)
    exp.run()


if __name__ == '__main__':
    # trivial_example() # Checked
    # highdim_example()
    RC_example()
    # RCR_example()
    # AdaANN_example()
