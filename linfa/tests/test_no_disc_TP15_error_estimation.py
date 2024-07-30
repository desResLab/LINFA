from linfa.run_experiment import experiment
from linfa.transform import Transformation
from linfa.discrepancy import Discrepancy
import torch
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Import rcr model
from linfa.models.discrepancy_models import PhysChem_error

def run_test():

    exp = experiment()
    exp.name = "TP15_no_disc_error_estimation"
    exp.flow_type           = 'realnvp'     # str: Type of flow (default 'realnvp') # TODO: generalize to work for TP1
    exp.n_blocks            = 15            # int: Number of hidden layers   
    exp.hidden_size         = 100           # int: Hidden layer size for MADE in each layer (default 100)
    exp.n_hidden            = 1             # int: Number of hidden layers in each MADE
    exp.activation_fn       = 'relu'        # str: Activation function used (default 'relu')
    exp.input_order         = 'sequential'  # str: Input oder for create_mask (default 'sequential')
    exp.batch_norm_order    = True          # bool: Order to decide if batch_norm is used (default True)
    exp.save_interval       = 1000          # int: How often to sample from normalizing flow
    
    # p0,e,sigma_e (measurement noise also estimated)
    exp.input_size          = 3             # int: Dimensionalty of input (default 2)
    exp.batch_size          = 200           # int: Number of samples generated (default 100)
    exp.true_data_num       = 2             # double: Number of true model evaluted (default 2)
    exp.n_iter              = 10000         # int: Number of iterations (default 25001)
    exp.lr                  = 0.001 #00075       # float: Learning rate (default 0.003)
    exp.lr_decay            = 0.9999        # float:  Learning rate decay (default 0.9999)
    exp.log_interval        = 1             # int: How often to show loss stat (default 10)

    exp.run_nofas           = False         # normalizing flow with adaptive surrogate
    exp.surrogate_type      = 'discrepancy' # type of surrogate we are using
    exp.surr_pre_it         = 1000          # int: Number of pre-training iterations for surrogate model
    exp.surr_upd_it         = 2000          # int: Number of iterations for the surrogate model update
    exp.calibrate_interval  = 1000          #:int:    How often the surrogate model is updated

    exp.annealing           = False         
    exp.budget              = 216           # int: Total number of true model evaulations
    exp.surr_folder         = "./" 
    exp.use_new_surr        = True

    exp.output_dir = './results/' + exp.name
    exp.log_file = 'log.txt'
    
    exp.seed = 35435                        # int: Random seed used
    exp.n_sample = 5000                     # int: Batch size to generate final results/plots
    exp.no_cuda = True                      # Running on CPU by default but teste on CUDA

    exp.optimizer = 'RMSprop'
    exp.lr_scheduler = 'ExponentialLR'

    exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

    # Define transformation
    trsf_info = [['tanh', -40.0, 30.0, 500.0, 1500.0],
                 ['tanh', -30.0, 30.0, -30000.0, -15000.0],
                 ['tanh', -20.0, 20.0, 0.00001, 0.2]]
    trsf = Transformation(trsf_info)
    
    # Apply the transformation
    exp.transform = trsf

    # Add temperatures and pressures for each evaluation
    variable_inputs = [[350.0, 400.0, 450.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Assign as experiment model
    exp.model = PhysChem_error(variable_inputs)

    # Read data
    exp.model.data = np.loadtxt('observations.csv', delimiter = ',', skiprows = 1)

    if(len(exp.model.data.shape) < 2):
        exp.model.data = np.expand_dims(exp.model.data, axis=0)
    
    # Form tensors for variables and results in observations
    var_grid_in = torch.tensor(exp.model.data[:,:2])
    var_grid_out = torch.tensor(exp.model.data[:,2:])

    # The discrepancy is included in the model
    exp.surrogate = None

    # Define log density
    def log_density(calib_inputs, model, surrogate, transform):
        
        # Compute transformation by log Jacobian
        adjust = transform.compute_log_jacob_func(calib_inputs)
        phys_inputs = transform.forward(calib_inputs)

        # Initialize negative log likelihood
        total_nll = torch.zeros((calib_inputs.size(0), 1))
        
        # Initialize total number of variable inputs
        total_var_inputs = len(model.var_in)
           
        # Evaluate model response - (num_var x num_batch)
        modelOut = model.solve_t(transform.forward(calib_inputs)).t()

        # Evaluate discrepancy
        if (surrogate is None):
            discrepancy = torch.zeros(modelOut.shape).t()
        else:
            # (num_var)
            discrepancy = surrogate.forward(model.var_in)
        
        # Get the calibration parameters
        p0Const, eConst, std_dev_ratio = torch.chunk(phys_inputs, chunks = 3, dim = 1)
        
        # Get data - (num_var x num_obs)
        Data = torch.tensor(model.data[:,2:]).to(exp.device)
        num_reapeat_obs = Data.size(1)

        # Convert standard deviation ratio to standard deviation
        std_dev = std_dev_ratio.flatten() * torch.mean(Data)
        
        # Evaluate log-likelihood:
        # Loop on the available observations
        for loopA in range(num_reapeat_obs):
            
            # -n / 2 * log ( 2 pi ) 
            l1 = -0.5 * np.prod(model.data.shape[1]) * np.log(2.0 * np.pi)

            # - n / 2 * log (sigma)
            l2 = -0.5 * model.data.shape[1] * torch.log(std_dev)
            
            # - 1 / (2 * sigma^2) sum_{i = 1} ^ N (eta_i + disc_i - y_i)^2 
            l3 = -0.5 / (std_dev ** 2) * torch.sum((modelOut + discrepancy.t() - Data[:,loopA].unsqueeze(0))**2, dim = 1)

            negLL = -(l1 + l2 + l3) # sum contributions
            res = -negLL.reshape(calib_inputs.size(0), 1) # reshape

            # Accumulate
            total_nll += res

            fin_res = total_nll/num_reapeat_obs + adjust
                
        # Return log-likelihood
        return fin_res

    # Assign log density model
    exp.model_logdensity = lambda x: log_density(x, exp.model, exp.surrogate, exp.transform)
   
    # Define log prior
    def log_prior(calib_inputs, transform):
        # TODO: figure out why calib_inputs are coming in on physical basus

        # Compute transformation log Jacobian
        adjust = transform.compute_log_jacob_func(calib_inputs)
        
        # Compute the calibration inputs in the physical domain
        phys_inputs = transform.forward(calib_inputs)
        
        # Gaussian prior on physically meaningful inputs
        mean = np.array([1.0E3,-21.0E3])
        cov = np.array([[100.0**2, 0.0],
                        [0.0,   500.0**2]])
        gauss_prior_res = torch.from_numpy(stats.multivariate_normal.logpdf(phys_inputs[:,:2].detach().numpy(), mean = mean, cov = cov))
        
        # Beta prior on noise
        std_dev_ratio = torch.distributions.beta.Beta(torch.tensor([2.0]), torch.tensor([19.0]))
        
        # Wrong prior being applied to phys_inputs
        prior_res = std_dev_ratio.log_prob(phys_inputs[:,2])

        # xvalues = torch.linspace(0, 1, 100)
        # plt.plot(xvalues.numpy(), torch.exp(std_dev_ratio.log_prob(xvalues)).numpy())
        # plt.hist(phys_inputs[:,2].detach().numpy(), density = True)
        # plt.xlim(0,0.6)
        # plt.show()
        # exit()

        res = gauss_prior_res + prior_res +  adjust
        return res

    exp.model_logprior = lambda x: log_prior(x, exp.transform)
    # exp.model_logprior = None

    # Run VI
    exp.run()

def generate_data(use_true_model=False,num_observations=50):

    # Set variable grid
    var_grid = [[350.0, 400.0, 450.0],
                [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Create model
    model = PhysChem_error(var_grid)
    
    # Generate data
    model.genDataFile(use_true_model = use_true_model, num_observations = num_observations)

# Main code
if __name__ == "__main__":

    generate_data(use_true_model = False, num_observations = 1)

    run_test()