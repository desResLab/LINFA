import torch
import matplotlib.pyplot as plt
from linfa.discrepancy import Discrepancy
import numpy as np
import os
import argparse
from numpy import random
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

def scale_limits(min,max,factor):
    if(min > max):
        temp = max
        max = min
        min = temp
    center = 0.5 * (min + max)
    range = max - min
    return center - factor * 0.5 * range, center + factor * 0.5 * range

def plot_disr_histograms(lf_file, lf_dicr_file, lf_discr_noise_file, data_file, out_dir, sample_size = 250):

    # Read result files
    lf_model = np.loadtxt(lf_file)
    lf_model_plus_disc = np.loadtxt(lf_dicr_file)
    lf_model_plus_disc_plus_noise = np.loadtxt(lf_discr_noise_file)
    data = np.loadtxt(data_file)

    # Check for multiple TP-pairs
    ## shape : no. var inputs pairs x no. batches
    num_dim = len(np.shape(lf_model))
    
    if num_dim == 1:
        # Plot histograms
        plt.figure(figsize = (6,4))
        ax = plt.gca()
        plt.hist(lf_model,  label = r'$\eta \vert \mathbf{\theta}$', alpha = 0.5, density = True, hatch = '/')
        plt.hist(lf_model_plus_disc,  label = r'$\zeta \vert \mathbf{\theta}, \delta$', alpha = 0.5, density = True)
        plt.hist(lf_model_plus_disc_plus_noise,  label = r'$y \vert \mathbf{\theta}, \delta, \epsilon$', alpha = 0.5, density = True, hatch = '.')
        plt.axvline(data[2], label = r'$y$', color = 'k', linewidth = 3)
        plt.legend(fontsize = 14)
        ax.set_xlabel('Coverage [ ]', fontweight = 'bold', fontsize = 16)
        ax.set_ylabel('Density', fontweight = 'bold', fontsize = 16)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True, labelsize = 15)
        plt.tight_layout()
        plt.savefig(out_dir+'hist.png', bbox_inches = 'tight', dpi = 300)
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
        sample = np.zeros([len(temps), len(pressures), sample_size]) # Initialize

         # For plotting
        plt.figure(figsize = (6,4))
        ax = plt.gca()
        clrs = ['b', 'm', 'r'] # Line colors for each temperature
        mkrs = ['v', 's', 'o'] # Line colors for each temperature
        lines = []  # List to store Line2D objects for legend lines
            
        # Loop over temperatures
        for loopA, temp in enumerate(temps):
            
            # Loop over pressures
            for loopB, pressure in enumerate(pressures):

                # Evalute random sample of true process posterior
                sample[loopA, loopB] = lf_model_plus_disc[loopA, loopB, random_array]

            # Plot function & save line psroperties for legend
            plt.plot(np.tile(pressures, (sample_size, 1)).transpose(),
                    sample[loopA],
                    linewidth=0.01,
                    color=clrs[loopA])
            
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
        ax.set_ylabel('Coverage [ ]', fontsize = 16, fontweight = 'bold')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True, labelsize = 15)
        plt.tight_layout()
        plt.savefig(out_dir+'hist.png', bbox_inches='tight', dpi = 300)
        plt.show()
        plt.close()

def plot_discr_surface_2d(file_path, data_file, num_1d_grid_points, data_limit_factor, out_dir):

    # TODO: need to rewrite conditional statements to raise error message when there is only one measurement
    exp_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)

    # Create new discrepancy
    dicr = Discrepancy(model_name = exp_name, 
                       model_folder = dir_name,
                       lf_model = None,
                       input_size = None,
                       output_size = None,
                       var_grid_in = None,
                       var_grid_out = None)
    dicr.surrogate_load()

    # Get the number of dimensions for the aux variable
    num_dim = dicr.var_grid_in.size(1)
   
    if(num_dim == 2):
        min_dim_1 = torch.min(dicr.var_grid_in[:,0])
        max_dim_1 = torch.max(dicr.var_grid_in[:,0])
        min_dim_2 = torch.min(dicr.var_grid_in[:,1])
        max_dim_2 = torch.max(dicr.var_grid_in[:,1])
        min_dim_1,max_dim_1 = scale_limits(min_dim_1, max_dim_1, data_limit_factor)
        min_dim_2,max_dim_2 = scale_limits(min_dim_2, max_dim_2, data_limit_factor)

        test_grid_1 = torch.linspace(min_dim_1, max_dim_1, num_1d_grid_points)
        test_grid_2 = torch.linspace(min_dim_2, max_dim_2, num_1d_grid_points)
        grid_t, grid_p = torch.meshgrid(test_grid_1, test_grid_2, indexing='ij')
        test_grid = torch.cat((grid_t.reshape(-1,1), grid_p.reshape(-1,1)),1)
    
        res = dicr.forward(test_grid)

        x = test_grid[:,0].cpu().detach().numpy()
        y = test_grid[:,1].cpu().detach().numpy()
        z = res.cpu().detach().numpy().flatten()

        ax = plt.figure(figsize = (4,4)).add_subplot(projection='3d')
        ax.plot_trisurf(x, y, z, cmap = plt.cm.Spectral, linewidth = 0.2, antialiased = True)
        ax.set_xlabel('Temperature [K]', fontsize = 16, fontweight = 'bold', labelpad = 10)
        ax.set_ylabel('Pressure [Pa]', fontsize = 16, fontweight = 'bold', labelpad = 10)
        ax.set_zlabel('Discrepancy [ ]', fontsize = 16, fontweight = 'bold', labelpad = 10)
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True, labelsize = 15)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        plt.tight_layout()
        plt.savefig(out_dir+'disc_surf.png', bbox_inches = 'tight', dpi = 300)
        plt.close()

    else:
        print('ERROR. Invalid number of dimensions. Should be 2. Instead is ',num_dim)
        exit(-1)

def eval_discrepancy_custom_grid(file_path,train_grid_in,train_grid_out,test_grid):

    exp_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path) 

    print(exp_name)
    print(dir_name)

    # Create new discrepancy
    dicr = Discrepancy(model_name = exp_name, 
                       model_folder = dir_name,
                       lf_model = None,
                       input_size = train_grid_in.size(1),
                       output_size = 1,
                       var_grid_in = train_grid_in,
                       var_grid_out = train_grid_out)
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
        plot_disr_histograms(lf_file, lf_dicr_file, lf_discr_noise_file, data_file, out_dir)
    elif(args.result_mode == 'discr_surface'):
        plot_discr_surface_2d(discr_sur_file,data_file,args.num_1d_grid_points,args.data_limit_factor, out_dir)
    else:
        print('ERROR. Invalid execution mode')
        exit(-1)


