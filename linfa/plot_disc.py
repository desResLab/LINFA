import torch
import matplotlib.pyplot as plt
from linfa.discrepancy import Discrepancy
import numpy as np
import os

# directory = 'results'
# iteration = '25000'
# experiment = 'test_lf_with_disc_hf_data_prior_TP1'
# filename1 = 'outputs_lf'
# filename2 = 'outputs_lf+discr'pi
# filename3 = 'outputs_lf+discr+noise'

# path1 = os.path.join(directory, experiment, experiment + '_' + filename1 + '_' + iteration)
# path2 = os.path.join(directory, experiment, experiment + '_' + filename2 + '_' + iteration)
# path3 = os.path.join(directory, experiment, experiment + '_' + filename3 + '_' + iteration)

# lf_model = np.loadtxt(path1)
# lf_model_plus_disc = np.loadtxt(path2)
# lf_model_plus_disc_plus_noise = np.loadtxt(path3)

# plt.hist(lf_model[:,0], label = 'LF')
# plt.hist(lf_model_plus_disc[:,0],  label = 'LF + disc')
# plt.hist(lf_model_plus_disc_plus_noise[:,0],  label = 'LF + disc + noise')
# plt.xlabel('Coverage')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()

# shape: number of ___?___ (15) x Batch size (5000)
# print(np.shape(lf_model))

def plot_discrepancy(file_path,train_grid_in,train_grid_out,test_grid):

    exp_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path) 

    print(exp_name)
    print(dir_name)

    # Create new discrepancy
    dicr = Discrepancy(model_name=exp_name, 
                       model_folder=dir_name,
                       lf_model=None,
                       input_size=train_grid_in.size(1),
                       output_size=1,
                       var_grid_in=train_grid_in,
                       var_grid_out=train_grid_out)
    # Load the surrogate
    dicr.surrogate_load()

    # Evaluate surrogate
    return dicr.forward(test_grid)

# =========
# MAIN CODE
# =========
if __name__ == '__main__':

    file_path = './tests/results/test_lf_with_disc_hf_data_TP1/test_lf_with_disc_hf_data_TP1'
    obs_file = './tests/results/test_lf_with_disc_hf_data_TP1/test_lf_with_disc_hf_data_TP1_data'
    train_grid_in = torch.from_numpy(np.loadtxt(obs_file).reshape(1,-1)[:,:2])
    train_grid_out = torch.from_numpy(np.loadtxt(obs_file).reshape(1,-1)[:,2:])

    # Create a testing grid    
    t_test = torch.linspace(350.0,450.0,20)
    p_test = torch.linspace(1.0,5.0,20)
    grid_t,grid_p = torch.meshgrid(t_test, p_test, indexing='ij')
    test_grid = torch.cat((grid_t.reshape(-1,1),grid_p.reshape(-1,1)),1)
    
    res = plot_discrepancy(file_path,train_grid_in,train_grid_out,test_grid)

    print(res)


    # Draw histograms of lf+discr
    # Draw histograms of lf
    # Draw histograms of lf+discr+noise
