import torch
import matplotlib.pyplot as plt
from linfa.discrepancy import Discrepancy
import numpy as np
import os
import argparse
from numpy import random
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, ScalarFormatter

def scale_limits(min, max, factor):
    if(min > max):
        temp = max
        max = min
        min = temp
    range = max - min
    center = 0.5 * (max + min)
    return center - factor * 0.5 * range, center + factor * 0.5 * range

def plot_disr_histograms(lf_file, lf_dicr_file, lf_discr_noise_file, data_file, step_num, out_dir, img_format = 'png', sample_size = 250):

    # Read result files
    data                          = np.loadtxt(data_file)           # Experimental observations
    lf_model                      = np.loadtxt(lf_file)             # Samples from low-fidelity posterior
    lf_model_plus_disc            = np.loadtxt(lf_dicr_file)        # Samples from low-fidelity posterior + discrepancy
    lf_model_plus_disc_plus_noise = np.loadtxt(lf_discr_noise_file) # Samples from low-fidelity posterior + discrepancy + noise
    
    # Check for repated observations
    ## shape : no. var inputs pairs x no. batches
    num_dim = len(np.shape(lf_model))

    if(num_dim == 1):

        # DES: NEED TO TEST FOR TP1 I THINK !!!

        # Plot histograms
        plt.figure(figsize = (6,4))
        ax = plt.gca()
        plt.hist(lf_model, label = r'$\eta \vert \mathbf{\theta}$', alpha = 0.5, density = True, hatch = '/')
        plt.hist(lf_model_plus_disc, label = r'$\zeta \vert \mathbf{\theta}, \delta$', alpha = 0.5, density = True)
        plt.hist(lf_model_plus_disc_plus_noise, label = r'$y \vert \mathbf{\theta}, \delta, \epsilon$', alpha = 0.5, density = True, hatch = '.')

        for loopA in range(len(data[2:])):
            if loopA == 0:
                # add label to legend
                plt.axvline(data[2 + loopA], label = r'$y$', color = 'k', linewidth = 3)
            else:
                plt.axvline(data[2 + loopA], color = 'k', linewidth = 3)

        plt.legend(fontsize = 14)
        ax.set_xlabel('Coverage [ ]', fontweight = 'bold', fontsize = 16)
        ax.set_ylabel('Density',      fontweight = 'bold', fontsize = 16)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True, labelsize = 15)
        plt.tight_layout()
        plt.savefig(out_dir + 'hist_' + str(step_num) +'.%s' % img_format, format = img_format, bbox_inches = 'tight', dpi = 300)
        plt.close()
   
    else:
        
        batch_size = len(lf_model_plus_disc[0])
        temps = np.unique(data[:, 0])
        pressures = np.unique(data[:, 1])
        no_reps = len(data[0,2:])
        observed = data[:,2:].reshape(len(temps), len(pressures), no_reps)

         ## Prepare for random sampling of batches
        lf_model_plus_disc = lf_model_plus_disc.reshape(len(temps), len(pressures), batch_size)
        random_array = np.random.randint(low = 0, high = batch_size, size = sample_size) # Randomly sample batch numbers without replacement
        sample = np.zeros([len(temps), len(pressures), sample_size])                     # Initialize

         # For plotting
        plt.figure(figsize = (6,4))
        ax = plt.gca()
        clrs = ['b', 'm', 'r'] # Line colors for each temperature
        mkrs = ['v', 's', 'o'] # Line colors for each temperature
        lines = []             # List to store Line2D objects for legend lines
            
        # Loop over temperatures
        for loopA, temp in enumerate(temps):
            
            # Loop over pressures
            for loopB, pressure in enumerate(pressures):

                # Evalute random sample of true process posterior
                sample[loopA, loopB] = lf_model_plus_disc[loopA, loopB, random_array]

            # Plot function & save line psroperties for legend
            plt.plot(np.tile(pressures, (sample_size, 1)).transpose(), sample[loopA], linewidth=0.1, color=clrs[loopA])
            
            for loopC in range(no_reps):
                line = plt.plot(np.tile(pressures, (sample_size, 1)).transpose(), 
                                observed[loopA, :, loopC], 
                                color = clrs[loopA], 
                                marker = mkrs[loopA],
                                markeredgecolor = 'k',
                                markersize = 8,
                                linestyle = '')[0]
            lines.append(line)
        
        # Manually create the legend with custom linewidth
        legend = plt.legend(lines, ['{} K'.format(round(temp)) for temp in temps], fontsize = 14)

        # Set the linewidth for the legend lines
        for line in legend.get_lines():
            line.set_linewidth(2.0)  # Adjust the linewidth as needed
            
        ax.set_xlabel('Pressure [Pa]', fontsize = 16, fontweight = 'bold')
        ax.set_ylabel('Coverage [ ]',  fontsize = 16, fontweight = 'bold')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True, labelsize = 15)
        plt.tight_layout()
        plt.savefig(out_dir + 'hist_' + str(step_num) +'.%s' % img_format, format = img_format, bbox_inches = 'tight', dpi = 300)
        plt.close()

def prep_test_grid(vars, limit_factor, no_grid_pts):
    
    ## Get min and max values of the variable inputs
    # Variable input 1
    min_dim_1 = torch.min(vars[:,0])
    max_dim_1 = torch.max(vars[:,0])
    min_dim_1, max_dim_1 = scale_limits(min_dim_1, max_dim_1, limit_factor)
    
    # Variable input 2
    min_dim_2 = torch.min(vars[:,1])
    max_dim_2 = torch.max(vars[:,1])
    min_dim_2, max_dim_2 = scale_limits(min_dim_2, max_dim_2, limit_factor)

    # Create test grid of variable inputs to evaluate discrepancy surrogate
    test_grid_1 = torch.linspace(min_dim_1, max_dim_1, no_grid_pts)
    test_grid_2 = torch.linspace(min_dim_2, max_dim_2, no_grid_pts)
    grid_t, grid_p = torch.meshgrid(test_grid_1, test_grid_2, indexing='ij')
    test_grid = torch.cat((grid_t.reshape(-1,1), grid_p.reshape(-1,1)),1)

    return test_grid

def print_disc_stats(avg_disc, disc_bounds):
    print('\n')
    print('Avg. Disc.\t LB \t\t UB')
    print('----------------------------------------')
    for loopA in range(len(disc_bounds[0])):
        print('{:<10}\t{:<10}\t{:<10}'.format(round(avg_disc[loopA], 4), round(disc_bounds[0][loopA], 4), round(disc_bounds[1][loopA], 4)))
        # Check for violation of bounds
        if avg_disc[loopA] < disc_bounds[0][loopA] or avg_disc[loopA] > disc_bounds[1][loopA]:
            print('Error in percentile calculation. Exiting...')
            exit()

def compare_disc(train, test):
    print('\n')
    print('Train Disc. \t Test Disc.')
    print('----------------------------')
    for loopA in range(len(train)):
        avg_disc_str = '{:<10}'.format(round(train.tolist()[loopA], 4))
        pred_disc_str = '{:<10}'.format(round(test[loopA].item(), 4))  # Access the element directly
        print(f'{avg_disc_str}\t{pred_disc_str}')

def plot_discr_surface_2d(file_path, lf_file, data_file, num_1d_grid_points, data_limit_factor, step_num, out_dir, img_format = 'png', nom_coverage = 95.0):

    # Load training data
    exp_name = os.path.basename(file_path)       # Name of experiment
    dir_name = os.path.dirname(file_path)        # Name of results directory
    exp_data = np.loadtxt(data_file)             # Experimental data
    lf_train = np.loadtxt(lf_file)               # Low-fidelity model posterior samples

    # Create an instance of the discrepancy class
    dicr = Discrepancy(model_name = exp_name, 
                       model_folder = dir_name,
                       lf_model = None,
                       input_size = None,
                       output_size = None,
                       var_grid_in = None,
                       var_grid_out = None)
    
    # Load trained FNN from experiment
    dicr.surrogate_load()

    # Print discrepancy surrogate
    dicr.pretty_print(print_data = True)

    # Get the number of dimensions for the aux variable
    num_var_pairs = dicr.var_grid_in.size(0)    # No. of variable input-pairs
    num_var_ins   = dicr.var_grid_in.size(1)    # No. of variable inputs

    # Check for invalid number of variable inputs
    if num_var_ins != 2:
        print('ERROR. Invalid number of variable inputs. Should be 2. Instead is ', num_var_ins)
        exit(-1)
    
    # Check for invalid number of variable input pairs
    elif num_var_pairs <= 1:
        print('ERROR. Invalid number of variable input pairs. Should be > 1. Instead is ', num_var_pairs)
        exit(-1)

    else:
        # Define test grid
        test_grid = prep_test_grid(dicr.var_grid_in, data_limit_factor, num_1d_grid_points)

        ## Variable input 1
        min_dim_1 = torch.min(dicr.var_grid_in[:,0])
        max_dim_1 = torch.max(dicr.var_grid_in[:,0])
        min_dim_1, max_dim_1 = scale_limits(min_dim_1, max_dim_1, data_limit_factor)
        
        ## Variable input 2
        min_dim_2 = torch.min(dicr.var_grid_in[:,1])
        max_dim_2 = torch.max(dicr.var_grid_in[:,1])
        min_dim_2, max_dim_2 = scale_limits(min_dim_2, max_dim_2, data_limit_factor)

        # Create test grid of variable inputs to evaluate discrepancy surrogate
        test_grid_1 = torch.linspace(min_dim_1, max_dim_1, num_1d_grid_points)
        test_grid_2 = torch.linspace(min_dim_2, max_dim_2, num_1d_grid_points)
        grid_t, grid_p = torch.meshgrid(test_grid_1, test_grid_2, indexing='ij')
        test_grid = torch.cat((grid_t.reshape(-1,1), grid_p.reshape(-1,1)),1)

        # Evaluate discrepancy over test grid
        res = dicr.forward(test_grid)

         # Assign obsersations from data
        observations = exp_data[:,num_var_ins:]

        # Assign variable inputs   
        var_train = [exp_data[:, 0], exp_data[:, 1]]

        # Assign discrepancy target, i.e., obs - lf model predictions for each batch
        disc = observations - lf_train

        # Compute average discrepancy across batches used for training
        train_disc = disc.mean(axis=1)
        
        # Compute error bars for averaged discrepancy
        discBnds = [np.percentile(disc, 100 - nom_coverage, axis = 1), # 5 percentile
                    np.percentile(disc, nom_coverage, axis = 1)] # 95 percentile
        
        errBnds = [train_disc - np.percentile(disc, 100 - nom_coverage, axis = 1), # 5 percentile
                   np.percentile(disc, nom_coverage, axis = 1) -  train_disc] # 95 percentile
                
        # For debugging
        if True:
            print_disc_stats(train_disc, discBnds)
            compare_disc(train_disc, res)
                    
        # Prepare test grid and discpreancy for plotting
        x = test_grid[:,0].cpu().detach().numpy() # Variable input 1
        y = test_grid[:,1].cpu().detach().numpy() # Variable input 2
        z = res.cpu().detach().numpy().flatten()  # Discrepancy

        # Plot discrepancy surface as a function of variable inputs 1 & 2
        ax = plt.figure(figsize = (4,4)).add_subplot(projection='3d')
        ax.plot_trisurf(x, y, z, cmap = plt.cm.Spectral, linewidth = 0.2, antialiased = True)
        ax.scatter(var_train[0], var_train[1], train_disc, color = 'k', s = 8)
        ax.errorbar(var_train[0], var_train[1], train_disc, zerr = errBnds, fmt = 'o', color = 'k', ecolor = 'k', capsize = 3)
        ax.set_xlabel('Temperature [K]', fontsize = 16, fontweight = 'bold', labelpad = 15)
        ax.set_ylabel('Pressure [Pa]',   fontsize = 16, fontweight = 'bold', labelpad = 15)
        ax.set_zlabel('Discrepancy [ ]', fontsize = 16, fontweight = 'bold', labelpad = 15)
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True, labelsize = 15)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        plt.tight_layout()
        print('Generating plot...: ',out_dir+'disc_surf_'+ str(step_num) +'.%s' % img_format)
        plt.savefig(out_dir+'disc_surf_'+ str(step_num) +'.%s' % img_format, format = img_format, bbox_inches = 'tight', dpi = 300)

def plot_marginal_stats(marg_stats_file, step_num, saveinterval, img_format, out_dir):
    
    # Get array of iterations where marginal statistics were saved
    iterations = np.arange(start = saveinterval, 
                           stop = step_num + saveinterval, 
                           step = saveinterval, 
                           dtype = int)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes = axes.flatten()

    # Loop over save intervals
    for loopA in range(len(iterations)):
        
        # Read file
        stats = np.loadtxt(marg_stats_file + str(iterations[loopA])) 
        
        # Unpack sample statistics
        mean = stats[:, 0]  # Sample mean
        sd = stats[:, 1]    # Sample std. dev.

        # Plot mean and sd of calibration parameters with adjusted data point label font size
        axes[0].plot(iterations[loopA], mean[0], 'o', color = 'r', markersize = 8)
        axes[1].plot(iterations[loopA], mean[1], 'o', color = 'r', markersize = 8)
        axes[2].plot(iterations[loopA], sd[0], 'v', color = 'b', markersize = 8)
        axes[3].plot(iterations[loopA], sd[1], 'v', color = 'b', markersize = 8)

    # Set common labels for the y-axis
    for ax, ylabel in zip(axes, [r'$\mathbb{E}(\theta_1 \vert \mathcal{D})$',
                                 r'$\mathbb{E}(\theta_2 \vert \mathcal{D})$', 
                                 r'$\sqrt{\mathbb{V}(\theta_1 \vert \mathcal{D})}$', 
                                 r'$\sqrt{\mathbb{V}(\theta_2 \vert \mathcal{D})}$']):
        ax.set_ylabel(ylabel, fontsize = 16, fontweight = 'bold')

    # Add ground truth parameters to the sample average plots
    gtCalParams = np.array([1.0E3, -21.0E3])
    axes[0].axhline(gtCalParams[0], color = 'k', linewidth = 3, label = r'$\theta_1$')
    axes[1].axhline(gtCalParams[1], color = 'k', linewidth = 3, label = r'$\theta_2$')
    axes[0].legend(fontsize = 14)
    axes[1].legend(fontsize = 14)
        
    # Set x-axis label only for the bottom two subplots
    for ax in axes[-2:]:
        ax.set_xlabel('Iterations', fontsize = 16, fontweight = 'bold')

    # Set tick label font size
    for ax in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis = 'both', 
                       which = 'both', 
                       direction = 'in', 
                       top = True, 
                       right = True, 
                       labelsize = 15)
    
    # Remove redundant tick labels
    axes[0].tick_params(axis = 'x', labelbottom = False)
    axes[1].tick_params(axis = 'x', labelbottom = False)

    # Set specific formatting for parameter 2
    axes[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(out_dir + 'marginal_stats_' + str(step_num) +'.%s' % img_format, format = img_format, bbox_inches = 'tight', dpi = 300)

def plot_marginal_posterior(params_file, step_num, out_dir, img_format = 'png'):
    

    gtCalParams = np.array([1.0E3, -21.0E3])
    params = np.loadtxt(params_file)
    calInput1 = params[:, 0]
    calInput2 = params[:, 1]

    fig, axes = plt.subplots(1, 2, figsize = (8, 4))
    axes = axes.flatten()
    axes[0].hist(calInput1)
    axes[1].hist(calInput2)
    
    # Add groundtruth parameter
    axes[0].axvline(gtCalParams[0], color = 'k', linewidth = 3)
    axes[1].axvline(gtCalParams[1], color = 'k', linewidth = 3)

    # Set common labels
    for ax, xlabel in zip(axes, [r'$\theta_1$', r'$\theta_2$']):
        ax.set_xlabel(xlabel, fontsize = 16, fontweight = 'bold')
    
    axes[0].set_ylabel('Frequency', fontsize = 16, fontweight = 'bold')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(out_dir + 'marginal_posterior_' + str(step_num) +'.%s' % img_format, format = img_format, bbox_inches = 'tight', dpi = 300)
    


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
                        action = None,
                        const = None,
                        default = 'histograms',
                        type = str,
                        choices = ['histograms','discr_surface', 'marginal_stats', 'marginal_posterior'],
                        required=False,
                        help = 'Type of plot/result to generate',
                        metavar = '',
                        dest = 'result_mode')
    
    
    # folder name
    parser.add_argument('-z', '--num_points',
                        action = None,
                        const = None,
                        default = 10,
                        type = int,
                        required = False,
                        help = 'Number of on-dimensional test grid points (same in every dimension)',
                        metavar = '',
                        dest = 'num_1d_grid_points')

    # folder name
    parser.add_argument('-y', '--limfactor',
                        action = None,
                        const = None,
                        default = 1.0,
                        type = float,
                        required = False,
                        help = 'Factor for test grid limits from data file',
                        metavar = '',
                        dest = 'data_limit_factor')
    
    # save interval
    parser.add_argument('-si', '--saveinterval',
                        action = None,
                        const = None,
                        default = 1.0,
                        type = float,
                        required = False,
                        help = 'Save interval to read for each iteration',
                        metavar = '',)

    # Parse Commandline Arguments
    args = parser.parse_args()

    # Set file name/path for lf and discr results
    out_dir     = args.folder_name + args.exp_name + '/'
    lf_file      = out_dir + args.exp_name + '_outputs_lf_' + str(args.step_num)
    lf_dicr_file        = out_dir + args.exp_name + '_outputs_lf+discr_' + str(args.step_num) 
    lf_discr_noise_file = out_dir + args.exp_name + '_outputs_lf+discr+noise_' + str(args.step_num)
    discr_sur_file      = out_dir + args.exp_name
    data_file           = out_dir + args.exp_name + '_data'
    marg_stats_file     = out_dir + args.exp_name + '_marginal_stats_'
    params_file         = out_dir + args.exp_name + '_params_' + str(args.step_num)

    if(args.result_mode == 'histograms'):
        plot_disr_histograms(lf_file, lf_dicr_file, lf_discr_noise_file, data_file, args.step_num, out_dir, args.img_format)
    elif(args.result_mode == 'discr_surface'):
        plot_discr_surface_2d(discr_sur_file, lf_file, data_file, args.num_1d_grid_points, args.data_limit_factor, args.step_num, out_dir, args.img_format)
    elif(args.result_mode == 'marginal_stats'):
        plot_marginal_stats(marg_stats_file, args.step_num, args.saveinterval, args.img_format, out_dir)
    elif(args.result_mode == 'marginal_posterior'):
        plot_marginal_posterior(params_file, args.step_num, out_dir, args.img_format)
    else:
        print('ERROR. Invalid execution mode')
        exit(-1)


