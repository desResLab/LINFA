from linfa.run_experiment import experiment
from linfa.transform import Transformation
import torch
import random
import numpy as np

# Import rcr model
from linfa.models.discrepancy_models import PhysChem

def run_test():

    exp = experiment()
    exp.name = "lf_no_disc_hf_data_tp15"
    exp.flow_type           = 'maf'         # str: Type of flow (default 'realnvp')
    exp.n_blocks            = 15            # int: Number of hidden layers   
    exp.hidden_size         = 100           # int: Hidden layer size for MADE in each layer (default 100)
    exp.n_hidden            = 1             # int: Number of hidden layers in each MADE
    exp.activation_fn       = 'relu'        # str: Activation function used (default 'relu')
    exp.input_order         = 'sequential'  # str: Input oder for create_mask (default 'sequential')
    exp.batch_norm_order    = True          # bool: Order to decide if batch_norm is used (default True)
    exp.save_interval       = 1000          # int: How often to sample from normalizing flow
    
    exp.input_size          = 2             # int: Dimensionalty of input (default 2)
    exp.batch_size          = 200           # int: Number of samples generated (default 100)
    exp.true_data_num       = 2             # double: Number of true model evaluted (default 2)
    exp.n_iter              = 25001         # int: Number of iterations (default 25001)
    exp.lr                  = 0.001         # float: Learning rate (default 0.003)
    exp.lr_decay            = 0.9999        # float:  Learning rate decay (default 0.9999)
    exp.log_interal         = 10            # int: How often to show loss stat (default 10)

    exp.run_nofas           = False         # normalizing flow with adaptive surrogate
    exp.surrogate_type      = 'discrepancy' # type of surrogate we are using
    exp.surr_pre_it         = 1000          # int: Number of pre-training iterations for surrogate model
    exp.surr_upd_it         = 1000          # int: Number of iterations for the surrogate model update

    exp.annealing           = False         # TODO : turn this on eventually
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
    trsf_info = [['tanh', -7.0, 7.0, 500.0, 1500.0],
                 ['tanh', -7.0, 7.0, -30000.0, -15000.0]]
    trsf = Transformation(trsf_info)
    
    # Apply the transformation
    exp.transform = trsf

    # Add temperatures and pressures for each evaluation
    variable_inputs = [[350.0, 400.0, 450.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Define model
    langmuir_model = PhysChem(variable_inputs)
    # Assign as experiment model
    exp.model = langmuir_model

    # Read data
    exp.model.data = np.loadtxt('observations.csv', delimiter = ',', skiprows = 1)

    if(len(exp.model.data.shape) < 2):
        exp.model.data = np.expand_dims(exp.model.data, axis=0)
    
    # No surrogate
    exp.surrogate = None

    # Define log density
    def log_density(calib_inputs, model, surrogate, transform):
        
        # Compute transformation by log Jacobian
        adjust = transform.compute_log_jacob_func(calib_inputs)

        # Initialize negative log likelihood
        total_nll = torch.zeros((calib_inputs.size(0), 1))
        
        # HAD TO MOVE THIS UP BEFORE EVALUATE DISCREPANCY            
        # Evaluate model response - (num_var x num_batch)
        modelOut = langmuir_model.solve_t(transform.forward(calib_inputs)).t()

        # Evaluate discrepancy
        if (surrogate is None):
            discrepancy = torch.zeros(modelOut.shape).t()
        else:
            # (num_var)
            discrepancy = surrogate.forward(model.var_in)
        
        # Get the absolute values of the standard deviation (num_var)
        stds = langmuir_model.defOut * langmuir_model.stdRatio
        
        # Get data - (num_var x num_obs)
        Data = torch.tensor(langmuir_model.data[:,2:]).to(exp.device)
        num_obs = Data.size(1)
        
        # Evaluate log-likelihood:
        # Loop on the available observations
        for loopA in range(num_obs):
            l1 = -0.5 * np.prod(langmuir_model.data.shape) * np.log(2.0 * np.pi)
            
            # TODO: generalize to multiple inputs
            l2 = (-0.5 * langmuir_model.data.shape[1] * torch.log(torch.prod(stds))).item()
            l3 = -0.5 * torch.sum(((modelOut + discrepancy.t() - Data[:,loopA].unsqueeze(0)) / stds.t())**2, dim = 1)
            
            # Compute negative ll (num_batch x 1)
            negLL = -(l1 + l2 + l3) # sum contributions
            res = -negLL.reshape(calib_inputs.size(0), 1) # reshape
        
            # Accumulate
            total_nll += res
                
        # Return log-likelihood
        return total_nll/num_obs + adjust

    # Assign log density model
    exp.model_logdensity = lambda x: log_density(x, exp.model, exp.surrogate, exp.transform)

    # Run VI
    exp.run()

def generate_data():

    # Set variable grid
    var_grid = [[350.0, 400.0, 450.0],
                [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Create model
    model = PhysChem(var_grid)
    
    # Generate data
    model.genDataFile(use_true_model=True,num_observations=1)

# Main code
if __name__ == "__main__":
    
    generate_data()
    
    run_test()







