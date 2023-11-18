#import unittest
import os
import linfa
from linfa.run_experiment_mf import experiment_mf
from linfa.transform import Transformation
from linfa.nofas import Surrogate
from linfa.discrepancy import Discrepancy
import torch
import random
import numpy as np

def linear_example():

    # Experiment Setting
    exp = experiment_mf()
    exp.name              = "linear_mf"
    exp.input_size        = 3             
    exp.flow_type         = 'realnvp'         
    exp.n_blocks_lf       = 10     
    exp.n_blocks_hf       = 12     
    exp.hidden_size       = 20           
    exp.n_hidden          = 1             
    exp.activation_fn     = 'relu'        
    exp.input_order       = 'sequential'  
    exp.batch_norm_order  = True          

    # NOFAS parameters
    exp.run_nofas          = False  
    exp.log_interval       = 100     

    # OPTIMIZER parameters
    exp.optimizer       = 'Adam'   
    exp.lr_lf           = 0.008   
    exp.lr_hf           = 0.005   
    exp.lr_decay_lf     = 0.75    
    exp.lr_decay_hf     = 0.75    
    exp.lr_scheduler    = 'StepLR' 
    exp.lr_step_lf      = 2000    
    exp.lr_step_hf      = 2000      
    exp.batch_size_lf   = 100     
    exp.batch_size_hf   = 10             
    exp.n_iter_lf       = 5000  
    exp.n_iter_hf       = 5000    

    # ANNEALING parameters
    exp.annealing     = False     

    # OUTPUT parameters
    exp.output_dir          = './results/' + exp.name 
    exp.log_file            = 'log.txt'                
    exp.seed                = 2                 
    exp.n_sample            = 5000                                                 

    # DEVICE parameters
    exp.no_cuda = True #:bool: Flag to use CPU

    # Set device
    exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

    # Compute data

    low_params = torch.tensor([0.5, 10, -5])
    high_params = torch.tensor([1, 8, -6])
    
    # Functions for both models
    def model(x, p):
        return p[0]*(6*x - 2)**2*torch.sin(12*x - 4) + p[1]*(x - 0.5) + p[2]

    x_low = torch.tensor([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
    x_high = torch.tensor([0, 0.4, 0.6, 1])

    y_low = model(x_low, low_params)
    y_high = model(x_high, high_params)

    # Add noise to the data
    low_data = y_low + np.random.normal(0,0.5,21)
    high_data = y_high + np.random.normal(0,0.5,4)

    # Concatenate the x and y data values together
    low_data = np.concatenate([np.reshape(x_low, [21,1]), np.reshape(low_data, [21,1])], 1)
    high_data = np.concatenate([np.reshape(x_high, [4,1]),np.reshape(high_data, [4,1])], 1)

    # Create into tensors for Pytorch
    low_data = torch.from_numpy(low_data)
    high_data = torch.from_numpy(high_data)


    # Model Setting
    def log_density(params, d):

        # Compute Model
        def targetPosterior(p, x):
            return p[0]*(6*x - 2)**2*torch.sin(12*x - 4) + p[1]*(x - 0.5) + p[2]

        f = torch.zeros(len(params)).to(exp.device)

        for i in range(len(params)):
            y_out = targetPosterior(params[i], d[:,0])
            val = torch.linalg.norm(y_out - d[:, 1])
            f[i] = -val ** 2 / (2*0.5**2)

        return f

    # Define low and high fidelity models
    exp.model_logdensity_lf = lambda x: log_density(x, low_data)
    exp.model_logdensity_hf = lambda x: log_density(x, high_data)

    exp.run()
    
def nonlinear_example_2d():
    # Experiment Setting
    exp = experiment_mf()
    exp.name              = "nonlinear_2d"
    exp.input_size        = 2             
    exp.flow_type         = 'realnvp'         
    exp.n_blocks_lf       = 10     
    exp.n_blocks_hf       = 14     
    exp.hidden_size       = 20           
    exp.n_hidden          = 1             
    exp.activation_fn     = 'relu'        
    exp.input_order       = 'sequential'  
    exp.batch_norm_order  = True          

    # NOFAS parameters
    exp.run_nofas         = False  
    exp.log_interval      = 500     

    # OPTIMIZER parameters
    exp.optimizer       = 'Adam'   
    exp.lr_lf           = 0.005   
    exp.lr_hf           = 0.005   
    exp.lr_decay_lf     = 0.75    
    exp.lr_decay_hf     = 0.75    
    exp.lr_scheduler    = 'StepLR' 
    exp.lr_step_lf      = 2000    
    exp.lr_step_hf      = 2000      
    exp.batch_size_lf   = 100     
    exp.batch_size_hf   = 10             
    exp.n_iter_lf       = 5000    
    exp.n_iter_hf       = 5000    

    # ANNEALING parameters
    exp.annealing     = False     

    # OUTPUT parameters
    exp.output_dir          = './results/' + exp.name 
    exp.log_file            = 'log.txt'                
    exp.seed                = 2                 
    exp.n_sample            = 5000    

    # DEVICE parameters
    exp.no_cuda = True #:bool: Flag to use CPU

    # Set device
    exp.device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')

    # Define transformation
    # One list for each variable
    trsf_info_low = [['tanh',-7,7,-0.2,0.2],
                     ['tanh',-7,7,0.5,1.5]]
    trsf_low = Transformation(trsf_info_low)        
    exp.transform_lf = trsf_low

    trsf_info_high = [['tanh',-7,7,0.25,1.25],
                      ['tanh',-7,7,-0.5,0.5]]
    trsf_high = Transformation(trsf_info_high)        
    exp.transform_hf = trsf_high

    # Compute data

    low_params = torch.tensor([0.01, 0.99, 0, 1])
    high_params = torch.tensor([0.7, 0.3, 1, 0])

    #functions for both models
    def model(x, p):
        return torch.exp(p[0]*x[:,0] + p[1]*x[:,1]) + 0.15*torch.sin(2*np.pi*(p[2]*x[:,0] + p[3]*x[:,1]))

    x_low = y_low = torch.linspace(0,1,10)
    X_low, Y_low = torch.meshgrid(x_low, y_low)
    X_low_flatten, Y_low_flatten = np.reshape(X_low, (-1, 1)), np.reshape(Y_low, (-1, 1))
    XY_low = torch.from_numpy(np.concatenate([X_low_flatten, Y_low_flatten], 1))
    Z_low_data = model(XY_low, low_params)

    x_high = y_high = torch.linspace(0,1,5)
    X_high, Y_high = torch.meshgrid(x_high, y_high)
    X_high_flatten, Y_high_flatten = np.reshape(X_high, (-1, 1)), np.reshape(Y_high, (-1, 1))
    XY_high = torch.from_numpy(np.concatenate([X_high_flatten, Y_high_flatten], 1))
    Z_high_data = model(XY_high, high_params)

    #add noise to the data
    low_data = Z_low_data + np.random.normal(0,0.1,100)
    high_data = Z_high_data + np.random.normal(0,0.1,25)

    #concatenate the x and y data values together
    low_data = np.concatenate([np.reshape(XY_low, [100,2]), np.reshape(low_data, [100,1])], 1)
    high_data = np.concatenate([np.reshape(XY_high, [25,2]),np.reshape(high_data, [25,1])], 1)

    #create into tensors for Pytorch
    low_data = torch.from_numpy(low_data)
    high_data = torch.from_numpy(high_data)

    # Model Setting
    def log_density_low(x, d, transform):

        # Compute transformation log Jacobian
        adjust = transform.compute_log_jacob_func(x)

        params = transform.forward(x)

        # Compute Model
        def targetPosterior(p, Z):
            return torch.exp(p[0]*Z[:,0] + 0.99*Z[:,1]) + 0.15*torch.sin(2*np.pi*(0*Z[:,0] + p[1]*Z[:,1]))

        f = torch.zeros(len(params)).to(exp.device)

        for i in range(len(params)):
            y_out = targetPosterior(params[i], d[:,0:2])
            val = torch.linalg.norm(y_out - d[:, 2])
            f[i] = -val ** 2 / (2*0.1**2)

        return f.reshape(x.size(0), 1) + adjust


    def log_density_high(x, d, transform):

        # Compute transformation log Jacobian
        adjust = transform.compute_log_jacob_func(x)

        params = transform.forward(x)

        # Compute Model
        def targetPosterior(p, Z):
            return torch.exp(p[0]*Z[:,0] + 0.3*Z[:,1]) + 0.15*torch.sin(2*np.pi*(1*Z[:,0] + p[1]*Z[:,1]))

        f = torch.zeros(len(params)).to(exp.device)

        for i in range(len(params)):
            y_out = targetPosterior(params[i], d[:,0:2])
            val = torch.linalg.norm(y_out - d[:, 2])
            f[i] = -val ** 2 / (2*0.1**2)

        return f.reshape(x.size(0), 1) + adjust

    # Define low and high fidelity models
    exp.model_logdensity_lf = lambda x: log_density_low(x, low_data, trsf_low)
    exp.model_logdensity_hf = lambda x: log_density_high(x, high_data, trsf_high)
    
    exp.run()
    
if __name__ == '__main__':
    
    # Execute tests
    linear_example()
    #nonlinear_example_2d()