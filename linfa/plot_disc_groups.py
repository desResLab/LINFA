import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, ScalarFormatter

def get_data_labels(grouping):
    if grouping == 'TP1' or 'TP15':
        labels = ['none', 'prior', 'rep_meas', 'prior + rep_meas']
    elif grouping == 'prior':
        labels = ['TP1', 'TP15', 'TP1 + rep_meas', 'TP15 + rep_meas']
    elif grouping == 'repeated measurements':
        labels = ['TP1', 'TP15', 'TP1 + prior', 'TP15 + prior']
    return labels

def plot_marginal_stats(marg_stats_files, step_num, saveinterval, grouping, out_dirs):
    
    labels = get_data_labels(grouping)

    # Get array of iterations where marginal statistics were saved
    iterations = np.arange(start = saveinterval, 
                           stop = step_num + saveinterval, 
                           step = saveinterval, 
                           dtype = int)
    
    # Initialize figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    axes = axes.flatten()

    color1 = '#1f78b4'  # Blue
    color2 = '#33a02c'  # Teal
    color3 = '#e31a1c'  # Red
    color4 = '#ff7f00'  # Orange
    colors = [color1, color2, color3, color4]
    markers = ['v', 's', 'h','o']

    # Loop over experiments
    for loopA in range(len(marg_stats_files)):
        
        # Loop over save intervals
        for loopB in range(len(iterations)):

            # Read file
            stats = np.loadtxt(marg_stats_files[loopA] + str(iterations[loopB])) 
            
            # Unpack sample statistics
            mean = stats[:, 0]  # Sample mean
            sd = stats[:, 1]    # Sample std. dev.

            # Plot mean and sd of calibration parameters with adjusted data point label font size
            axes[0].plot(iterations[loopB], mean[0], markers[loopA], color = colors[loopA], markersize = 8)
            axes[1].plot(iterations[loopB], mean[1], markers[loopA], color = colors[loopA], markersize = 8)
            axes[2].plot(iterations[loopB], sd[0], markers[loopA], color = colors[loopA], markersize = 8)
            axes[3].plot(iterations[loopB], sd[1], markers[loopA], color = colors[loopA], markersize = 8)

    # Set common labels for the y-axis
    for ax, ylabel in zip(axes, [r'$\mathbb{E}(\theta_1 \vert \mathcal{D})$',
                                 r'$\mathbb{E}(\theta_2 \vert \mathcal{D})$', 
                                 r'$\mathbb{V}(\theta_1 \vert \mathcal{D})$', 
                                 r'$\mathbb{V}(\theta_2 \vert \mathcal{D})$']):
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
        
    # # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.savefig(out_dirs + 'marginal_stats.png', bbox_inches = 'tight', dpi = 300)
    plt.show()


# =========
# MAIN CODE
# =========
if __name__ == '__main__':

    # Init parser
    parser = argparse.ArgumentParser(description='.')

    # folder name
    parser.add_argument('-f', '--folder',
                        action = None,
                        # nargs='+',
                        const=None,
                        default='./',
                        type=str,
                        required=False,
                        help='Folder with experiment results',
                        metavar='',
                        dest='folder_name')

    # experiment names
    parser.add_argument('-n', '--names',
                        action=None,
                        nargs='+',
                        const=None,
                        default='./',
                        type = str,
                        required=True,
                        help='Names of numerical experiments',
                        dest='exp_names')

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
    parser.add_argument('-m', '--mode',
                        action=None,
                        const=None,
                        default='marginal_stats_group',
                        type=str,
                        choices=['marginal_stats_group'],
                        required=False,
                        help='Type of plot/result to generate',
                        metavar='',
                        dest='result_mode')
    
    # Enable dark mode for pictures
    parser.add_argument('-g', '--grouping',
                        action=None,
                        const=None,
                        default='repeated',
                        type=str,
                        choices=['TP1', 'TP15', 'prior', 'repeated measurements'],
                        required=False,
                        help='Type of group',
                        metavar='',
                        dest='grouping')
    
    # save interval
    parser.add_argument('-si', '--saveinterval',
                        action=None,
                        # nargs='+',
                        const=None,
                        default=1.0,
                        type=float,
                        required=False,
                        help='Save interval to read for each iteration',
                        metavar='',)
    
    # Parse Commandline Arguments
    args = parser.parse_args()
    
    # Set file name/path for lf and discr results
    out_dirs = [args.folder_name + x + '/' for x in args.exp_names]
    
    lf_files = []
    lf_dicr_files = []
    lf_discr_noise_files = []
    data_files = []
    marg_stats_files = []
    params_files = []
    for loopA in range(len(out_dirs)):
        lf_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_outputs_lf_' + str(args.step_num))
        lf_dicr_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_outputs_lf+discr_' + str(args.step_num) )
        lf_discr_noise_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_outputs_lf+discr+noise_' + str(args.step_num))
        data_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_data')
        marg_stats_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_marginal_stats_')
        params_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_params_' + str(args.step_num))

    if(args.result_mode == 'marginal_stats_group'):
        plot_marginal_stats(marg_stats_files, args.step_num, args.saveinterval, args.grouping, out_dirs)
    else:
        print('ERROR. Invalid execution mode')
        exit(-1)