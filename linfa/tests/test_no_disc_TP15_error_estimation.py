from linfa.run_experiment import experiment
from linfa.transform import Transformation
from linfa.discrepancy import Discrepancy
import torch
import random
import numpy as np

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
    exp.lr                  = 0.0005        # float: Learning rate (default 0.003)
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
        
        # Evaluate default model output to compute standard deviation
        def_out = model.solve_t(model.defParams)

        # Evaluate discrepancy
        if (surrogate is None):
            discrepancy = torch.zeros(modelOut.shape).t()
        else:
            # (num_var)
            discrepancy = surrogate.forward(model.var_in)
        
        # Get the absolute values of the standard deviation (num_var)
        p0Const, eConst, sigma_e = torch.chunk(phys_inputs, chunks = 3, dim = 1)
        std_dev = sigma_e.flatten() * torch.mean(torch.abs(def_out))

        # Get data - (num_var x num_obs)
        Data = torch.tensor(model.data[:,2:]).to(exp.device)
        num_obs = Data.size(1)
        
        # Evaluate log-likelihood:
        # Loop on the available observations
        for loopA in range(num_obs):
            
            # -n / 2 * log ( 2 pi ) 
            l1 = -0.5 * np.prod(model.data.shape[1]) * np.log(2.0 * np.pi)

            # Ask about this likelihood
            # - n / 2 * log (sigma^2)
            l2 = (-0.5 * model.data.shape[1] * torch.log(std_dev ** 2))
            
            # - 1 / (2 * sigma^2) sum_{i = 1} ^ N (eta_i + disc_i - y_i)^2 
            l3 = -0.5 / (std_dev ** 2) * torch.sum((modelOut + discrepancy.t() - Data[:,loopA].unsqueeze(0))**2, dim = 1)

            # Compute negative ll (num_batch x 1)
            # print('l1', l1)
            # print('l2', l2)
            # print('l3', l3)
            # exit()
            negLL = -(l1 + l2 + l3) # sum contributions
            res = -negLL.reshape(calib_inputs.size(0), 1) # reshape
        
            # Accumulate
            total_nll += res

            fin_res = total_nll/num_obs + adjust

            # print('total nll', (total_nll/num_obs).mean().item())
            # print('adjust', adjust.mean().item())
            # print('total', fin_res.mean().item())
            # exit()
                
        # Return log-likelihood
        return fin_res

    # Assign log density model
    exp.model_logdensity = lambda x: log_density(x, exp.model, exp.surrogate, exp.transform)

    # Define log prior
    def log_prior(calib_inputs, transform):
        
        # Compute transformation log Jacobian
        adjust = transform.compute_log_jacob_func(calib_inputs)
        
        # Compute the calibration inputs in the physical domain
        phys_inputs = transform.forward(calib_inputs)

        # Define prior moments for first two variables
        pr_avg = torch.tensor([[1E3, -21E3]])
        pr_std = torch.tensor([[1E2, 500]])

        # Eval log prior
        # -n / 2 * log ( 2 pi ) 
        l1 = -0.5 * calib_inputs.size(1) * np.log(2.0 * np.pi)     

        # - n / 2 * log (sigma^2)       
        l2 = (-0.5 * torch.log(torch.prod(pr_std))).item()
        
        # - 1 / (2 * sigma^2) sum_{i = 1} ^ N (eta_i + disc_i - y_i)^2 
        l3 = -0.5 * torch.sum(((phys_inputs[:,:1] - pr_avg)/pr_std)**2, dim = 1).unsqueeze(1)
        # Add gaussian log prior for first two parameters
        gauss_prior_res = l1 + l2 + l3
        
        if False:
             # Add beta prior for third parameter
            sigma_prior = torch.distributions.beta.Beta(torch.tensor([1.0]), torch.tensor([0.5]))
            prior_res = sigma_prior.log_prob(phys_inputs[:,2])
        else:
            # Add uniform prior
            a = torch.tensor([0.0])
            b = torch.tensor([0.15])
            sigma_prior = torch.distributions.uniform.Uniform(low = a, high = b)
            prior_res = torch.zeros(phys_inputs[:,2].size(0))
            for loopA, siggy in enumerate(phys_inputs[:,2]):
                if a <= siggy <= b:
                    prior_res[loopA] = sigma_prior.log_prob(siggy)
                else:
                    # Large negative number
                    prior_res[loopA] = 500
        
        res = gauss_prior_res + prior_res +  adjust

        # print('Prior on std. dev. ratio:', prior_res.mean().item())
        # print('Gaussian prior on physical params:', gauss_prior_res.mean().item())
        # print('Adjustment:', adjust.mean().item())
        # print('Total:', res.mean().item())
        # exit()


        # Log prior is constributing the most to the loss function
        # Loss function is insensitive to values of sigma
        # Add transformation
        
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

    # Add beta prior for third parameter
    # test = torch.distributions.beta.Beta(torch.tensor([1.0]), torch.tensor([3.0]))
    # samples = test.sample_n(50)
    # print(samples)
    # exit()

    generate_data(use_true_model = False, num_observations = 1)

    run_test()



