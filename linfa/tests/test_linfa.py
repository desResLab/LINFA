import unittest
import os
import linfa
from linfa.run_experiment import experiment
from linfa.transform import Transformation
from linfa.nofas import Surrogate
from linfa.discrepancy import Discrepancy
import torch
import random
import numpy as np

class linfa_test_suite(unittest.TestCase):

    def trivial_example(self):

        if "it" in os.environ:
          max_it  = int(os.environ["it"])
          max_pre = 1000
        else:
          max_it  = 25001
          max_pre = 40000

        print('')
        print('--- TEST 1: TRIVIAL FUNCTION - NOFAS')
        print('')

        # Import trivial model
        from linfa.models.trivial_models import Trivial

        exp = experiment()
        exp.name              = "trivial"
        exp.flow_type         = 'realnvp'     # str: Type of flow (default 'realnvp')
        exp.n_blocks          = 5             # int: Number of layers (default 5)
        exp.hidden_size       = 100           # int: Hidden layer size for MADE in each layer (default 100)
        exp.n_hidden          = 1             # int: Number of hidden layers in each MADE (default 1)
        exp.activation_fn     = 'relu'        # str: Actication function used (default 'relu')
        exp.input_order       = 'sequential'  # str: Input order for create_mask (default 'sequential')
        exp.batch_norm_order  = True          # bool: Order to decide if batch_norm is used
        exp.save_interval = 5000          # int: How often to sample from normalizing flow

        exp.input_size    = 2       # int: Dimensionality of input (default 2)
        exp.batch_size    = 200     # int: Number of samples generated (default 100)
        exp.true_data_num = 2       # double: number of true model evaluated (default 2)
        # exp.n_iter      = 25001
        exp.n_iter        = max_it  # int: Number of iterations (default 25001)
        exp.lr            = 0.002   # float: Learning rate (default 0.003)
        exp.lr_decay      = 0.9999  # float: Learning rate decay (default 0.9999)
        exp.log_interval  = 10      # int: How often to show loss stat (default 10)

        exp.run_nofas          = True
        exp.surr_pre_it        = max_pre #:int: Number of pre-training iterations for surrogate model
        exp.surr_upd_it        = 6000  #:int: Number of iterations for the surrogate model update
        exp.annealing          = False
        exp.calibrate_interval = 1000  # int: How often to update surrogate model (default 1000)
        exp.budget             = 64    # int: Total number of true model evaluation
        exp.surr_folder        = "./"
        exp.use_new_surr       = True

        exp.output_dir   = './results/' + exp.name
        exp.log_file     = 'log.txt'
        exp.seed         = random.randint(0, 10 ** 9)  # int: Random seed used
        exp.n_sample     = 5000                        # int: Batch size to generate final results/plots
        exp.no_cuda      = True                        # Running on CPU by default but tested on CUDA

        exp.optimizer    = 'RMSprop'
        exp.lr_scheduler = 'ExponentialLR'

        exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

        # Define transformation
        # One list for each variable
        trsf_info = [['identity',0,0,0,0],
                     ['identity',0,0,0,0]]
        trsf = Transformation(trsf_info)        
        exp.transform = trsf

        # Define model
        model = Trivial(device=exp.device)
        exp.model = model

        # Get data
        res_path = os.path.abspath(os.path.join(os.path.dirname(linfa.__file__), os.pardir))
        model.data = np.loadtxt(res_path + '/resource/data/data_trivial.txt')

        # Define surrogate
        exp.surrogate = Surrogate(exp.name, lambda x: model.solve_t(trsf.forward(x)), 2, 2, 
                                  model_folder=exp.surr_folder, limits=[[0, 6], [0, 6]], 
                                  memory_len=20, device=exp.device)
        surr_filename = exp.surr_folder + exp.name
        if exp.use_new_surr or (not os.path.isfile(surr_filename + ".sur")) or (not os.path.isfile(surr_filename + ".npz")):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz could not be found. ".format(surr_filename))
            # 4 samples for each dimension: pre-grid size = 16
            exp.surrogate.gen_grid(gridnum=4)
            exp.surrogate.pre_train(exp.surr_pre_it, 0.03, 0.9999, 500, store=True)
        # Load the surrogate
        exp.surrogate.surrogate_load()

        # Define log density
        def log_density(x, model, surrogate, transform):
            # x contains the original, untransformed inputs

            # Compute transformation log Jacobian
            adjust = transform.compute_log_jacob_func(x)

            # Eval model output
            stds = torch.abs(model.solve_t(model.defParam)) * model.stdRatio
            Data = torch.tensor(model.data).to(exp.device)
            if surrogate:
              modelOut = exp.surrogate.forward(x)
            else:
              modelOut = model.solve_t(transform.forward(x))
              
            # Eval LL
            ll1 = -0.5 * np.prod(model.data.shape) * np.log(2.0 * np.pi)
            ll2 = (-0.5 * model.data.shape[1] * torch.log(torch.prod(stds))).item()
            ll3 = 0.0
            for i in range(2):
              ll3 += - 0.5 * torch.sum(((modelOut[:, i].unsqueeze(1) - Data[i, :].unsqueeze(0)) / stds[0, i]) ** 2, dim=1)
            negLL = -(ll1 + ll2 + ll3)

            # Return LL
            return -negLL.reshape(x.size(0), 1) + adjust

        # Assign log-density model
        exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate, trsf)

        # Run VI
        exp.run()


    def highdim_example(self):

        if "it" in os.environ:
          max_it  = int(os.environ["it"])
          max_pre = 1000
        else:
          max_it  = 25001
          max_pre = 100000

        print('')
        print('--- TEST 2: HIGH DIMENSIONAL SOBOL FUNCTION - NOFAS')
        print('')

        from linfa.models.highdim_models import Highdim
        
        exp = experiment()
        exp.name = "highdim"
        exp.flow_type         = 'realnvp'     # str: Type of flow (default 'realnvp')
        exp.n_blocks          = 15            # int: Number of layers (default 5)
        exp.hidden_size       = 100           # int: Hidden layer size for MADE in each layer (default 100)
        exp.n_hidden          = 1             # int: Number of hidden layers in each MADE (default 1)
        exp.activation_fn     = 'relu'        # str: Actication function used (default 'relu')
        exp.input_order       = 'sequential'  # str: Input order for create_mask (default 'sequential')
        exp.batch_norm_order  = True          # bool: Order to decide if batch_norm is used (default True)
        exp.save_interval     = 5000          # int: How often to sample from normalizing flow

        exp.input_size    = 5       # int: Dimensionality of input (default 2)
        exp.batch_size    = 250     # int: Number of samples generated (default 100)
        exp.true_data_num = 12      # double: number of true model evaluated (default 2)
        # exp.n_iter      = 25001
        exp.n_iter        = max_it  # int: Number of iterations (default 25001)
        exp.lr            = 0.003   # float: Learning rate (default 0.003)
        exp.lr_decay      = 0.9999  # float: Learning rate decay (default 0.9999)
        exp.log_interval  = 10      # int: How often to show loss stat (default 10)

        exp.run_nofas          = True
        exp.surr_pre_it       = max_pre #:int: Number of pre-training iterations for surrogate model
        exp.surr_upd_it       = 6000   #:int: Number of iterations for the surrogate model update

        exp.annealing          = False
        exp.calibrate_interval = 200   # int: How often to update surrogate model (default 1000)
        exp.budget             = 1023  # int: Total number of true model evaluation
        exp.surr_folder        = "./"
        exp.use_new_surr       = True

        exp.output_dir   = './results/' + exp.name
        exp.log_file     = 'log.txt'
        exp.seed         = random.randint(0, 10 ** 9)  # int: Random seed used
        exp.n_sample     = 5000                        # int: Batch size to generate final results/plots
        exp.no_cuda      = True                        # Running on CPU by default but tested on CUDA

        exp.optimizer    = 'RMSprop'
        exp.lr_scheduler = 'ExponentialLR'

        exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

        # Define transformation
        # One list for each variable
        trsf_info = [['identity',0,0,0,0],
                     ['identity',0,0,0,0],
                     ['identity',0,0,0,0],
                     ['identity',0,0,0,0],
                     ['identity',0,0,0,0]]
        trsf = Transformation(trsf_info)
        exp.transform = trsf

        # Define the model
        model = Highdim(exp.device)
        exp.model = model

        # Read data
        res_path = os.path.abspath(os.path.join(os.path.dirname(linfa.__file__), os.pardir))
        model.data = np.loadtxt(res_path + '/resource/data/data_highdim.txt')

        # Define the surrogate
        exp.surrogate = Surrogate(exp.name, lambda x: model.solve_t(trsf.forward(x)), model.input_num, model.output_num,
                                  model_folder=exp.surr_folder, limits=torch.Tensor([[-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0], [-3.0, 3.0]]), 
                                  memory_len=20, device=exp.device)
        surr_filename = exp.surr_folder + exp.name
        if exp.use_new_surr or (not os.path.isfile(surr_filename + ".sur")) or (not os.path.isfile(surr_filename + ".npz")):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz could not be found. ".format(surr_filename))
            exp.surrogate.gen_grid(gridnum=3)
            exp.surrogate.pre_train(exp.surr_pre_it, 0.03, 0.9999, 500, store=True)
        # Load the surrogate
        exp.surrogate.surrogate_load()

        # Define the log density        
        def log_density(x, model, surrogate, transform):

            # Compute transformation log Jacobian
            adjust = transform.compute_log_jacob_func(x)

            # Eval model or surrogate
            if surrogate:
              modelOut = surrogate.forward(x)
            else:
              modelOut = model.solve_t(transform(x))
            stds = model.defOut * model.stdRatio
            Data = torch.tensor(model.data).to(exp.device)

            # Compute LL
            ll1 = -0.5 * np.prod(model.data.shape) * np.log(2.0 * np.pi)  # a number
            ll2 = (-0.5 * model.data.shape[1] * torch.log(torch.prod(stds))).item()  # a number
            ll3 = 0.0
            for i in range(4):
              ll3 += - 0.5 * torch.sum(((modelOut[:, i].unsqueeze(1) - Data[i, :].unsqueeze(0)) / stds[0, i]) ** 2, dim=1)
            negLL = -(ll1 + ll2 + ll3) 
            res = -negLL.reshape(x.size(0), 1) + adjust
            
            # Return LL
            return res
        
        # Assign log-density model
        exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate,trsf)

        # Run VI
        exp.run()


    def rc_example(self):

        if "it" in os.environ:
          max_it  = int(os.environ["it"])
          max_pre = 1000
        else:
          max_it  = 25001
          max_pre = 40000

        print('')
        print('--- TEST 3: RC MODEL - NOFAS')
        print('')

        # Import model
        from linfa.models.circuit_models import rcModel
        
        exp = experiment()
        exp.name = "rc"
        exp.flow_type         = 'maf'         # str: Type of flow (default 'realnvp')
        exp.n_blocks          = 5             # int: Number of layers (default 5)
        exp.hidden_size       = 100           # int: Hidden layer size for MADE in each layer (default 100)
        exp.n_hidden          = 1             # int: Number of hidden layers in each MADE (default 1)
        exp.activation_fn     = 'relu'        # str: Actication function used (default 'relu')
        exp.input_order       = 'sequential'  # str: Input order for create_mask (default 'sequential')
        exp.batch_norm_order  = True          # bool: Order to decide if batch_norm is used (default True)
        exp.save_interval = 5000          # int: How often to sample from normalizing flow

        exp.input_size    = 2       # int: Dimensionality of input (default 2)
        exp.batch_size    = 250     # int: Number of samples generated (default 100)
        exp.true_data_num = 2       # int: number of true model evaluated (default 2)
        # exp.n_iter        = 25001   # int: Number of iterations (default 25001)
        exp.n_iter        = max_it
        exp.lr            = 0.005   # float: Learning rate (default 0.003)
        exp.lr_decay      = 0.9999  # float: Learning rate decay (default 0.9999)
        exp.log_interval  = 10      # int: How often to show loss stat (default 10)

        exp.run_nofas          = True
        exp.surr_pre_it       = max_pre #:int: Number of pre-training iterations for surrogate model
        exp.surr_upd_it       = 6000  #:int: Number of iterations for the surrogate model update

        exp.annealing          = False
        exp.calibrate_interval = 1000  # int: How often to update surrogate model (default 1000)
        exp.budget             = 64    # int: Total number of true model evaluation
        exp.surr_folder        = "./"
        exp.use_new_surr       = True

        exp.output_dir   = './results/' + exp.name
        exp.log_file     = 'log.txt'
        exp.seed         = random.randint(0, 10 ** 9)  # int: Random seed used
        exp.n_sample     = 5000                        # int: Batch size to generate final results/plots
        exp.no_cuda      = True                        # Running on CPU by default but tested on CUDA

        exp.optimizer = 'RMSprop'
        exp.lr_scheduler = 'ExponentialLR'

        exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

        # Define transformation
        # One list for each variable
        trsf_info = [['tanh',-7.0,7.0,100.0,1500.0],
                     ['exp',-7.0,7.0,1.0e-5,1.0e-2]]
        trsf = Transformation(trsf_info)
        exp.transform = trsf

        # Define model
        cycleTime = 1.07
        totalCycles = 10
        res_path = os.path.abspath(os.path.join(os.path.dirname(linfa.__file__), os.pardir))
        forcing = np.loadtxt(res_path + '/resource/data/inlet.flow')
        model = rcModel(cycleTime, totalCycles, forcing, device=exp.device)  # RCR Model Defined
        exp.model = model

        # Read Data
        model.data = np.loadtxt(res_path + '/resource/data/data_rc.txt')

        # Define surrogate model
        exp.surrogate = Surrogate(exp.name, lambda x: model.solve_t(trsf.forward(x)), exp.input_size, 3,
                                  model_folder=exp.surr_folder, limits=torch.Tensor([[-7, 7], [-7, 7]]), 
                                  memory_len=20, device=exp.device)
        surr_filename = exp.surr_folder + exp.name
        if exp.use_new_surr or (not os.path.isfile(surr_filename + ".sur")) or (not os.path.isfile(surr_filename + ".npz")):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz could not be found. ".format(surr_filename))
            exp.surrogate.gen_grid(gridnum=4)
            exp.surrogate.pre_train(exp.surr_pre_it, 0.03, 0.9999, 500, store=True)
        # Load the surrogate
        exp.surrogate.surrogate_load()

        # Define log density
        def log_density(x, model, surrogate, transform):
            
            # Compute transformation log Jacobian
            adjust = transform.compute_log_jacob_func(x)

            if surrogate:
                modelOut = surrogate.forward(x)
            else:
                modelOut = model.solve_t(transform.forward(x))

            # Get the absolute values of the standard deviations
            stds = model.defOut * model.stdRatio
            Data = torch.tensor(model.data).to(exp.device)
            
            # Eval Gaussian LL
            ll1 = -0.5 * np.prod(model.data.shape) * np.log(2.0 * np.pi)  # a number
            ll2 = (-0.5 * model.data.shape[1] * torch.log(torch.prod(stds))).item()  # a number
            ll3 = 0.0
            for i in range(3):
                ll3 += - 0.5 * torch.sum(((modelOut[:, i].unsqueeze(1) - Data[i, :].unsqueeze(0)) / stds[0, i]) ** 2, dim=1)
            negLL = -(ll1 + ll2 + ll3)
            res = -negLL.reshape(x.size(0), 1) + adjust
            
            # Return LL
            return res

        # Assign log-density
        exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate, trsf)

        # Run VI
        exp.run()

    def rcr_example(self):

        if "it" in os.environ:
          max_it  = int(os.environ["it"])
          max_pre = 1000
        else:
          max_it  = 25001
          max_pre = 50000

        print('')
        print('--- TEST 4: RCR MODEL - NOFAS')
        print('')

        # Import rcr model
        from linfa.models.circuit_models import rcrModel

        exp = experiment()
        exp.name = "rcr"
        exp.flow_type         = 'maf'         # str: Type of flow (default 'realnvp')
        exp.n_blocks          = 15            # int: Number of layers (default 5)
        exp.hidden_size       = 100           # int: Hidden layer size for MADE in each layer (default 100)
        exp.n_hidden          = 1             # int: Number of hidden layers in each MADE (default 1)
        exp.activation_fn     = 'relu'        # str: Actication function used (default 'relu')
        exp.input_order       = 'sequential'  # str: Input order for create_mask (default 'sequential')
        exp.batch_norm_order  = True          # bool: Order to decide if batch_norm is used (default True)
        exp.save_interval = 5000          # int: How often to sample from normalizing flow

        exp.input_size    = 3       # int: Dimensionality of input (default 2)
        exp.batch_size    = 500     # int: Number of samples generated (default 100)
        exp.true_data_num = 2       # double: number of true model evaluated (default 2)
        # exp.n_iter        = 25001   # int: Number of iterations (default 25001)
        exp.n_iter        = max_it
        exp.lr            = 0.003   # float: Learning rate (default 0.003)
        exp.lr_decay      = 0.9999  # float: Learning rate decay (default 0.9999)
        exp.log_interval  = 10      # int: How often to show loss stat (default 10)

        exp.run_nofas          = True
        exp.surr_pre_it       = max_pre #:int: Number of pre-training iterations for surrogate model
        exp.surr_upd_it       = 6000  #:int: Number of iterations for the surrogate model update

        exp.annealing          = False
        exp.calibrate_interval = 300  # int: How often to update surrogate model (default 1000)
        exp.budget             = 216  # int: Total number of true model evaluation
        exp.surr_folder        = "./"
        exp.use_new_surr       = True

        exp.output_dir = './results/' + exp.name
        exp.log_file = 'log.txt'
        exp.seed = random.randint(0, 10 ** 9)  # int: Random seed used
        exp.n_sample = 5000                    # int: Batch size to generate final results/plots
        exp.no_cuda = True                     # Running on CPU by default but tested on CUDA

        exp.optimizer = 'RMSprop'
        exp.lr_scheduler = 'ExponentialLR'

        exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

        # Define transformation
        # One list for each variable
        trsf_info = [['tanh',-7.0,7.0,100.0,1500.0],
                     ['tanh',-7.0,7.0,100.0,1500.0],
                     ['exp',-7.0,7.0,1.0e-5,1.0e-2]]
        trsf = Transformation(trsf_info)
        exp.transform = trsf

        # Define model
        cycleTime = 1.07
        totalCycles = 10
        res_path = os.path.abspath(os.path.join(os.path.dirname(linfa.__file__), os.pardir))
        forcing = np.loadtxt(res_path + '/resource/data/inlet.flow')
        model = rcrModel(cycleTime, totalCycles, forcing, device=exp.device)  # RCR Model Defined
        exp.model = model

        # Read data
        model.data = np.loadtxt(res_path + '/resource/data/data_rcr.txt')

        # Define surrogate
        exp.surrogate = Surrogate(exp.name, lambda x: model.solve_t(trsf.forward(x)), exp.input_size, 3,
                                  model_folder=exp.surr_folder, limits=torch.Tensor([[-7, 7], [-7, 7], [-7, 7]]), 
                                  memory_len=20, device=exp.device)
        surr_filename = exp.surr_folder + exp.name
        if exp.use_new_surr or (not os.path.isfile(surr_filename + ".sur")) or (not os.path.isfile(surr_filename + ".npz")):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz could not be found. ".format(surr_filename))
            exp.surrogate.gen_grid(gridnum=4)
            exp.surrogate.pre_train(exp.surr_pre_it, 0.03, 0.9999, 500, store=True)
        # Load the surrogate
        exp.surrogate.surrogate_load()

        # Define log density
        def log_density(x, model, surrogate, transform):

            # Compute transformation log Jacobian
            adjust = transform.compute_log_jacob_func(x)

            if surrogate:
                modelOut = surrogate.forward(x)
            else:
                modelOut = model.solve_t(transform.forward(x))

            # Get the absolute values of the standard deviations
            stds = model.defOut * model.stdRatio
            Data = torch.tensor(model.data).to(exp.device)
            
            # Eval LL
            ll1 = -0.5 * np.prod(model.data.shape) * np.log(2.0 * np.pi)  # a number
            ll2 = (-0.5 * model.data.shape[1] * torch.log(torch.prod(stds))).item()  # a number
            ll3 = 0.0
            for i in range(3):
                ll3 += - 0.5 * torch.sum(((modelOut[:, i].unsqueeze(1) - Data[i, :].unsqueeze(0)) / stds[0, i]) ** 2, dim=1)
            negLL = -(ll1 + ll2 + ll3)
            res = -negLL.reshape(x.size(0), 1) + adjust

            # Return LL
            return res

        # Assign logdensity model
        exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate, trsf)

        # Run VI
        exp.run()


    def adaann_example(self):

        # Quick run when repository is pushed
        if "it" in os.environ:
          run_T_1  = int(os.environ["it"])
          run_T    = 1
          run_T_0  = 1
          run_t0   = 0.99          
        else:
          run_T_1  = 10000
          run_T    = 3
          run_T_0  = 500
          run_t0   = 0.001          

        print('')
        print('--- TEST 5: FRIEDMAN 1 MODEL - ADAANN')
        print('')

        from linfa.run_experiment import experiment

        # Experiment Setting
        exp = experiment()
        exp.name              = "adaann"
        exp.flow_type         = 'realnvp'     # str: Type of flow (default 'realnvp')
        exp.n_blocks          = 10            # int: Number of layers (default 5)
        exp.hidden_size       = 20            # int: Hidden layer size for MADE in each layer (default 100)
        exp.n_hidden          = 1             # int: Number of hidden layers in each MADE (default 1)
        exp.activation_fn     = 'relu'        # str: Actication function used (default 'relu')
        exp.input_order       = 'sequential'  # str: Input order for create_mask (default 'sequential')
        exp.batch_norm_order  = True          # bool: Order to decide if batch_norm is used (default True)
        exp.save_interval     = 1000          # int: How often to sample from normalizing flow

        exp.input_size   = 10     # int: Dimensionality of input (default 2)
        exp.batch_size   = 100    # int: Number of samples generated (default 100)
        exp.lr           = 0.005  # float: Learning rate (default 0.003)
        # exp.lr_decay   = 0.75
        exp.lr_decay     = 0.9
        exp.log_interval = 10     # int: How often to show loss stat (default 10)
        exp.run_nofas    = False
        exp.annealing    = True

        exp.optimizer    = 'Adam'    # str: type of optimizer used
        exp.lr_scheduler = 'StepLR'  # str: type of lr scheduler used
        exp.lr_step      = 500       # int: number of steps for lr step scheduler
        exp.tol          = 0.4       # float: tolerance for AdaAnn scheduler
        # exp.t0           = 0.001     # float: initial inverse temperature value
        exp.t0           = run_t0
        exp.N            = 100       # int: number of sample points during annealing
        exp.N_1          = 400       # int: number of sample points at t=1
        # exp.T_0          = 500       # int: number of parameter updates at initial t0
        # exp.T            = 3         # int: number of parameter updates during annealing
        # exp.T_1          = 10000     # int: number of parameter updates at t=1
        exp.T_0          = run_T_0
        exp.T            = run_T
        exp.T_1          = run_T_1
        exp.M            = 1000      # int: number of sample points used to update temperature
        exp.scheduler    = 'AdaAnn'  # str: type of annealing scheduler used

        exp.output_dir   = './results/' + exp.name
        exp.log_file     = 'log.txt'
        # exp.seed = 35435  # int: Random seed used
        exp.seed         = random.randint(0, 10 ** 9)  # int: Random seed used
        exp.n_sample     = 5000                        # int: Batch size to generate final results/plots
        exp.no_cuda      = True                        # Running on CPU by default but tested on CUDA

        exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

        # Model Setting
        res_path = os.path.abspath(os.path.join(os.path.dirname(linfa.__file__), os.pardir))
        data_set = np.loadtxt(res_path + '/resource/data/D1000.csv',delimiter=',',skiprows=1)
        data = torch.tensor(data_set).to(exp.device)

        def log_density(params, d):

            # Compute Model
            # Modified bimodal version of the Friedman model
            def targetPosterior(b, x):
                return b[0] * torch.sin(np.pi * x[:, 0] * x[:, 1]) + \
                       (b[1] ** 2) * (x[:, 2] - b[2]) ** 2 + \
                       x[:, 3] * b[3] + \
                       x[:, 4] * b[4] + \
                       x[:, 5] * b[5] + \
                       x[:, 6] * b[6] + \
                       x[:, 7] * b[7] + \
                       x[:, 8] * b[8] + \
                       x[:, 9] * b[9]

            f = torch.zeros(len(params)).to(exp.device)

            for i in range(len(params)):
                y_out = targetPosterior(params[i], d)
                val = torch.linalg.norm(y_out - d[:, 10])
                f[i] = -val ** 2 / 2

            return f

        exp.model_logdensity = lambda x: log_density(x, data)
        exp.run()
    
    def rcr_nofas_adaann_example(self):
        
        # Quick run when repository is pushed
        if "it" in os.environ:
          max_it  = int(os.environ["it"])
          max_pre = 1000
        else:
          max_it  = 25001
          max_pre = 50000

        print('')
        print('--- TEST 6: RCR MODEL - NOFAS and ADAANN')
        print('')

        # Import rcr model
        from linfa.models.circuit_models import rcrModel

        exp = experiment()
        exp.name = "rcr_nofas_adaann"
        exp.flow_type         = 'maf'         # str: Type of flow (default 'realnvp')
        exp.n_blocks          = 15            # int: Number of layers (default 5)
        exp.hidden_size       = 100           # int: Hidden layer size for MADE in each layer (default 100)
        exp.n_hidden          = 1             # int: Number of hidden layers in each MADE (default 1)
        exp.activation_fn     = 'relu'        # str: Actication function used (default 'relu')
        exp.input_order       = 'sequential'  # str: Input order for create_mask (default 'sequential')
        exp.batch_norm_order  = True          # bool: Order to decide if batch_norm is used (default True)
        exp.save_interval     = 200           # int: How often to sample from normalizing flow

        exp.input_size    = 3       # int: Dimensionality of input (default 2)
        exp.batch_size    = 500     # int: Number of samples generated (default 100)
        exp.true_data_num = 2       # double: number of true model evaluated (default 2)
        # exp.n_iter        = 25001   # int: Number of iterations (default 25001)
        exp.n_iter        = max_it
        exp.lr            = 0.003   # float: Learning rate (default 0.003)
        exp.lr_decay      = 0.9999  # float: Learning rate decay (default 0.9999)
        exp.log_interval  = 10      # int: How often to show loss stat (default 10)

        exp.run_nofas          = True
        exp.calibrate_interval = 300  # int: How often to update surrogate model (default 1000)
        exp.budget             = 216  # int: Total number of true model evaluation
        exp.surr_pre_it       = max_pre #:int: Number of pre-training iterations for surrogate model
        exp.surr_upd_it       = 6000  #:int: Number of iterations for the surrogate model update
        exp.surr_folder        = "./"
        exp.use_new_surr       = True

        exp.annealing = True
        exp.tol       = 0.01     # float: tolerance for AdaAnn scheduler
        exp.t0        = 0.05     # float: initial inverse temperature value
        exp.N         = 100      # int: number of sample points during annealing
        exp.N_1       = 100      # int: number of sample points at t=1
        exp.T_0       = 500      # int: number of parameter updates at initial t0
        exp.T         = 5        # int: number of parameter updates during annealing
        exp.T_1       = 5000     # int: number of parameter updates at t=1
        exp.M         = 1000     # int: number of sample points used to update temperature
        exp.scheduler = 'AdaAnn' # str: type of annealing scheduler used
    
        exp.output_dir = './results/' + exp.name
        exp.log_file = 'log.txt'
        exp.seed = random.randint(0, 10 ** 9)  # int: Random seed used
        exp.n_sample = 5000                    # int: Batch size to generate final results/plots
        exp.no_cuda = False                     # Running on CPU by default but tested on CUDA

        exp.optimizer = 'RMSprop'
        exp.lr_scheduler = 'ExponentialLR'

        exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

        # Define transformation
        # One list for each variable
        trsf_info = [['tanh',-7.0,7.0,100.0,1500.0],
                     ['tanh',-7.0,7.0,100.0,1500.0],
                     ['exp',-7.0,7.0,1.0e-5,1.0e-2]]
        trsf = Transformation(trsf_info)
        exp.transform = trsf

        # Define model
        cycleTime = 1.07
        totalCycles = 10
        res_path = os.path.abspath(os.path.join(os.path.dirname(linfa.__file__), os.pardir))
        forcing = np.loadtxt(res_path + '/resource/data/inlet.flow')
        model = rcrModel(cycleTime, totalCycles, forcing, device=exp.device)  # RCR Model Defined
        exp.model = model

        # Read data
        model.data = np.loadtxt(res_path + '/resource/data/data_rcr.txt')

        # Define surrogate
        exp.surrogate = Surrogate(exp.name, lambda x: model.solve_t(trsf.forward(x)), exp.input_size, 3,
                                  model_folder=exp.surr_folder, limits=torch.Tensor([[-7, 7], [-7, 7], [-7, 7]]), 
                                  memory_len=20, device=exp.device)
        surr_filename = exp.surr_folder + exp.name
        if exp.use_new_surr or (not os.path.isfile(surr_filename + ".sur")) or (not os.path.isfile(surr_filename + ".npz")):
            print("Warning: Surrogate model files: {0}.npz and {0}.npz could not be found. ".format(surr_filename))
            exp.surrogate.gen_grid(gridnum=4)
            exp.surrogate.pre_train(exp.surr_pre_it, 0.03, 0.9999, 500, store=True)
        # Load the surrogate
        exp.surrogate.surrogate_load()

        # Define log density
        def log_density(x, model, surrogate, transform):

            # Compute transformation log Jacobian
            adjust = transform.compute_log_jacob_func(x)

            if surrogate:
                modelOut = surrogate.forward(x)
            else:
                modelOut = model.solve_t(transform.forward(x))

            # Get the absolute values of the standard deviations
            stds = model.defOut * model.stdRatio
            Data = torch.tensor(model.data).to(exp.device)
            
            # Eval LL
            ll1 = -0.5 * np.prod(model.data.shape) * np.log(2.0 * np.pi)  # a number
            ll2 = (-0.5 * model.data.shape[1] * torch.log(torch.prod(stds))).item()  # a number
            ll3 = 0.0
            for i in range(3):
                ll3 += - 0.5 * torch.sum(((modelOut[:, i].unsqueeze(1) - Data[i, :].unsqueeze(0)) / stds[0, i]) ** 2, dim=1)
            negLL = -(ll1 + ll2 + ll3)
            res = -negLL.reshape(x.size(0), 1) + adjust

            # Return LL
            return res

        # Assign logdensity model
        exp.model_logdensity = lambda x: log_density(x, model, exp.surrogate, trsf)

        # Run VI
        exp.run()

    def discrepancy_example(self):

        if "it" in os.environ:
          max_it  = int(os.environ["it"])
          max_pre = 1000
        else:
          max_it  = 25001
          max_pre = 50000

        print('')
        print('--- TEST 5: DISCREPANCY MODEL')
        print('')

        # Import rcr model
        from linfa.models.discrepancy_models import PhysChem

        exp = experiment()
        exp.name = "discrepancy_example"
        exp.flow_type           = 'maf'         # str: Type of flow (default 'realnvp')
        exp.n_blocks            = 15            # int: Number of hidden layers   
        exp.hidden_size         = 100           # int: Hidden layer size for MADE in each layer (default 100)
        exp.n_hidden            = 1             # int: Number of hidden layers in each MADE
        exp.activation_fn       = 'relu'        # str: Activation function used (default 'relu')
        exp.input_order         = 'sequential'  # str: Input oder for create_mask (default 'sequential')
        exp.batch_norm_order    = True          # bool: Order to decide if batch_norm is used (default True)
        exp.save_interval       = 50            # int: How often to sample from normalizing flow
        
        exp.input_size          = 2             # int: Dimensionalty of input (default 2)
        exp.batch_size          = 200           # int: Number of samples generated (default 100)
        exp.true_data_num       = 2             # double: Number of true model evaluted (default 2)
        exp.n_iter              = 25001         # int: Number of iterations (default 25001)
        exp.lr                  = 0.001         # float: Learning rate (default 0.003)
        exp.lr_decay            = 0.9999        # float:  Learning rate decay (default 0.9999)
        exp.log_interal         = 10            # int: How often to show loss stat (default 10)

        exp.run_nofas           = True
        exp.surrogate_type     = 'discrepancy'
        exp.surr_pre_it         = 1000          # int: Number of pre-training iterations for surrogate model
        exp.surr_upd_it         = 1000          # int: Number of iterations for the surrogate model update

        exp.annealing           = False         
        exp.calibrate_interval  = 300           # int: How often to update the surrogate model (default 1000)
        exp.budget              = 216           # int: Total number of true model evaulations
        exp.surr_folder         = "./" 
        exp.use_new_surr        = True

        exp.output_dir = './results/' + exp.name
        exp.log_file = 'log.txt'
        exp.seed = random.randint(0, 10 ** 9)   # int: Random seed used
        exp.n_sample = 5000                     # int: Batch size to generate final results/plots
        exp.no_cuda = True                      # Running on CPU by default but teste on CUDA

        exp.optimizer = 'RMSprop'
        exp.lr_scheduler = 'ExponentialLR'

        exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

        # Define transformation
        # One list for each variable
        trsf_info = [['linear', -1.0, 1.0, 500.0, 1500.0],
                     ['linear', -1.0, 1.0, -30000.0, -15000.0]]
        trsf = Transformation(trsf_info)
        exp.trasform = trsf

        # Add temperatures and pressures for each evaluation
        variable_inputs = [[350.0, 400.0, 450.0],
                           [1.0, 2.0, 3.0, 4.0, 5.0]]

        # Define model
        langmuir_model = PhysChem(variable_inputs)
        # Assign as experiment model
        exp.model = langmuir_model
        # Read data
        exp.model.data = np.loadtxt('observations.csv', delimiter = ',', skiprows = 1)

        # Form tensors for variables and results in observations
        var_grid_in = torch.tensor(exp.model.data[:,:2])
        var_grid_out = torch.tensor(exp.model.data[:,2:])

        # TEMPORARY
        use_surrogate = True

        # Define surrogate
        if(use_surrogate):

            # Create new discrepancy
            exp.surrogate = Discrepancy(model_name=exp.name, 
                                        lf_model=exp.model.solve_lf,
                                        input_size=exp.model.var_in.size(1),
                                        output_size=1,
                                        var_grid_in=var_grid_in,
                                        var_grid_out=var_grid_out)
            # Initially tune on the default values of the calibration variables
            exp.surrogate.update(langmuir_model.defParams, exp.surr_pre_it, 0.03, 0.9999, 100, store=True)
            # Load the surrogate
            exp.surrogate.surrogate_load()
        else:
            exp.surrogate = None

        # Define log density
        def log_density(calib_inputs, model, surrogate, transform):
            
            # Compute transformation by log Jacobian
            adjust = transform.compute_log_jacob_func(calib_inputs)

            # Initialize negative log likelihood
            total_nll = torch.zeros((calib_inputs.size(0), 1))
            
            # Initialize total number of variable inputs
            total_var_inputs = len(model.var_in)
                        
            # Evaluate discrepancy
            if (surrogate is None):
                discrepancy = torch.zeros()
            else:
                # (num_var)
                discrepancy = surrogate.forward(model.var_in)
            
            # Evaluate model response - (num_var x num_batch)
            modelOut = langmuir_model.solve_lf(transform.forward(calib_inputs)).t()

            # Get the absolute values of the standard deviation (num_var)
            stds = langmuir_model.defOut * langmuir_model.stdRatio
            # Get data - (num_var x num_obs)
            Data = torch.tensor(langmuir_model.data[:,2:]).to(exp.device)

            num_obs = Data.size(1)
            
            # Evaluate log-likelihood:
            # Loop on the available observations
            for loopA in range(num_obs):
                l1 = -0.5 * np.prod(langmuir_model.data.shape) * np.log(2.0 * np.pi)
                l2 = (-0.5 * langmuir_model.data.shape[1] * torch.log(torch.prod(stds))).item()
                # print('model out ',modelOut.size())
                # print('discrepancy ',discrepancy.t().size())
                # print('data ',Data[:,loopA].unsqueeze(0).size())
                # print('stds ',stds.t().size())
                # exit()
                l3 = -0.5 * torch.sum(((modelOut + discrepancy.t() - Data[:,loopA].unsqueeze(0)) / stds.t())**2, dim = 1) 
                # Compute negative ll (num_batch x 1)
                negLL = -(l1 + l2 + l3) # sum contributions
                res = -negLL.reshape(calib_inputs.size(0), 1) # reshape
            
            # Accumulate
            total_nll += res
                    
            # Return log-likelihood
            return total_nll/num_obs + adjust

        # Assign log density model
        exp.model_logdensity = lambda x: log_density(x, langmuir_model, exp.surrogate, trsf)

        # Run VI
        exp.run()

# MAIN         
if __name__ == '__main__':
    
    # Execute tests
    unittest.main()
