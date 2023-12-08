import torch
import matplotlib.pyplot as plt
from linfa.discrepancy import Discrepancy
import numpy as np
import os
import argparse

def scale_limits(min,max,factor):
    if(min>max):
        temp = max
        max = min
        min = temp
    center = 0.5*(min+max)
    range = max - min
    return center - factor*0.5*range, center + factor*0.5*range

def plot_disr_histograms(lf_file,lf_dicr_file,lf_discr_noise_file):

    # Read result files
    lf_model = np.loadtxt(lf_file)
    lf_model_plus_disc = np.loadtxt(lf_dicr_file)
    lf_model_plus_disc_plus_noise = np.loadtxt(lf_discr_noise_file)

    # Plot histograms
    plt.hist(lf_model, label = 'LF', alpha =0.5)
    plt.hist(lf_model_plus_disc,  label = 'LF + disc', alpha =0.5)
    plt.hist(lf_model_plus_disc_plus_noise,  label = 'LF + disc + noise', alpha = 0.5)
    plt.xlabel('Coverage')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_discr_surface_2d(file_path,data_file,num_1d_grid_points,data_limit_factor):

    exp_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path) 

    print(exp_name)
    print(dir_name)

    # Create new discrepancy
    dicr = Discrepancy(model_name=exp_name, 
                       model_folder=dir_name,
                       lf_model=None,
                       input_size=None,
                       output_size=None,
                       var_grid_in=None,
                       var_grid_out=None)
    dicr.surrogate_load()

    # Get the number of dimensions for the aux variable
    num_dim = dicr.var_grid_in.size(1)
    if(num_dim == 2):
        min_dim_1 = torch.min(dicr.var_grid_in[:,0])
        max_dim_1 = torch.max(dicr.var_grid_in[:,0])
        min_dim_2 = torch.min(dicr.var_grid_in[:,1])
        max_dim_2 = torch.max(dicr.var_grid_in[:,1])
        min_dim_1,max_dim_1 = scale_limits(min_dim_1,max_dim_1,data_limit_factor)
        min_dim_2,max_dim_2 = scale_limits(min_dim_2,max_dim_2,data_limit_factor)

        test_grid_1 = torch.linspace(min_dim_1,max_dim_1,num_1d_grid_points)
        test_grid_2 = torch.linspace(min_dim_2,max_dim_2,num_1d_grid_points)
        grid_t,grid_p = torch.meshgrid(test_grid_1, test_grid_2, indexing='ij')
        test_grid = torch.cat((grid_t.reshape(-1,1),grid_p.reshape(-1,1)),1)
    
        res = dicr.forward(test_grid)

        x = test_grid[:,0].cpu().detach().numpy()
        y = test_grid[:,1].cpu().detach().numpy()
        z = res.cpu().detach().numpy().flatten()

        print(x.shape,y.shape,z.shape)

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot_trisurf(x,y,z,linewidth=0.2, antialiased=True)
        plt.show()

    else:
        print('ERROR. Invalid number of dimensions. Should be 2. Instead is ',num_dim)
        exit(-1)

def eval_discrepancy_custom_grid(file_path,train_grid_in,train_grid_out,test_grid):

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

    # Init parser
    parser = argparse.ArgumentParser(description='.')

    # folder name
    parser.add_argument('-f', '--folder',
                        action=None,
                        # nargs='+',
                        const=None,
                        default='./',
                        type=str,
                        required=False,
                        help='Folder with experiment results',
                        metavar='',
                        dest='folder_name')


    # folder name
    parser.add_argument('-n', '--name',
                        action=None,
                        # nargs='+',
                        const=None,
                        default='./',
                        type=str,
                        required=True,
                        help='Name of numerical experiment',
                        metavar='',
                        dest='exp_name')

    # iteration number = 1
    parser.add_argument('-i', '--iter',
                        action=None,
                        # nargs='+',
                        const=None,
                        default=1,
                        type=int,
                        choices=None,
                        required=True,
                        help='Iteration number',
                        metavar='',
                        dest='step_num')

    # plot format
    parser.add_argument('-p', '--picformat',
                        action=None,
                        const=None,
                        default='png',
                        type=str,
                        choices=['png','pdf','jpg'],
                        required=False,
                        help='Output format for picture',
                        metavar='',
                        dest='img_format')

    # Enable dark mode for pictures
    parser.add_argument('-d', '--dark',
                        action='store_true',
                        default=False,
                        required=False,
                        help='Generate pictures for dark background',
                        dest='use_dark_mode')

    # Enable dark mode for pictures
    parser.add_argument('-m', '--mode',
                        action=None,
                        const=None,
                        default='histograms',
                        type=str,
                        choices=['histograms','discr_surface'],
                        required=False,
                        help='Type of plot/result to generate',
                        metavar='',
                        dest='result_mode')
    
    # folder name
    parser.add_argument('-z', '--num_points',
                        action=None,
                        # nargs='+',
                        const=None,
                        default=10,
                        type=int,
                        required=False,
                        help='Number of on-dimensional test grid points (same in every dimension)',
                        metavar='',
                        dest='num_1d_grid_points')

    # folder name
    parser.add_argument('-y', '--limfactor',
                        action=None,
                        # nargs='+',
                        const=None,
                        default=1.0,
                        type=float,
                        required=False,
                        help='Factor for test grid limits from data file',
                        metavar='',
                        dest='data_limit_factor')

    # Parse Commandline Arguments
    args = parser.parse_args()

    # Set file name/path for lf and discr results
    out_dir     = args.folder_name + args.exp_name + '/'
    lf_file = out_dir + args.exp_name + '_outputs_lf_' + str(args.step_num)
    lf_dicr_file = out_dir + args.exp_name + '_outputs_lf+discr_' + str(args.step_num) 
    lf_discr_noise_file = out_dir + args.exp_name + '_outputs_lf+discr+noise_' + str(args.step_num)
    discr_sur_file = out_dir + args.exp_name
    data_file = out_dir + args.exp_name + '_data'

    # out_info    = args.exp_name + '_' + str(args.step_num)

    if(args.result_mode == 'histograms'):
        plot_disr_histograms(lf_file,lf_dicr_file,lf_discr_noise_file)
    elif(args.result_mode == 'discr_surface'):
        plot_discr_surface_2d(discr_sur_file,data_file,args.num_1d_grid_points,args.data_limit_factor)
    else:
        print('ERROR. Invalid execution mode')
        exit(-1)


