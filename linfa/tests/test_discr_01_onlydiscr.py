from linfa.run_experiment import experiment
from linfa.transform import Transformation
from linfa.discrepancy import Discrepancy
import torch
import random
import numpy as np

# Import rcr model
from linfa.models.discrepancy_models import PhysChem

def run_test():

    exp = experiment()
    exp.name = "only_discr_ex"
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
    exp.n_iter              = 2501          # int: Number of iterations (default 25001)
    exp.lr                  = 0.001         # float: Learning rate (default 0.003)
    exp.lr_decay            = 0.9999        # float:  Learning rate decay (default 0.9999)
    exp.log_interal         = 10            # int: How often to show loss stat (default 10)

    exp.run_nofas           = True          # normalizing flow with adaptive surrogate
    exp.surrogate_type      = 'discrepancy' # type of surrogate we are using
    exp.surr_pre_it         = 10000         # int: Number of pre-training iterations for surrogate model
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
    
    # Form tensors for variables and results in observations
    var_grid_in = torch.tensor(exp.model.data[:,:2])
    var_grid_out = torch.tensor(exp.model.data[:,2:])

    # Define surrogate
    if(exp.run_nofas):

        # Create new discrepancy
        exp.surrogate = Discrepancy(model_name=exp.name, 
                                    lf_model=exp.model.solve_t,
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


    # Verify discrepancy training
    discr = exp.surrogate.forward(var_grid_in)
    lf_model = exp.model.solve_t(exp.model.defParams)
    print('%15s %15s' % ('Model+Discr','Observations'))
    for loopA in range(var_grid_in.size(0)):
        print('%15.6f %15.6f' % (lf_model[loopA][0].item()+discr[loopA][0].item(),var_grid_out[loopA][0].item()))

def generate_data(use_true_model=False, num_observations=50):

    # Set variable grid
    var_grid = [[350.0, 400.0, 450.0],
                [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Create model
    model = PhysChem(var_grid)
    
    # Generate data
    model.genDataFile(use_true_model=use_true_model,num_observations=num_observations)

# Main code
if __name__ == "__main__":
    
    generate_data(use_true_model = True, num_observations = 1)
    run_test()







