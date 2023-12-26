import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, ScalarFormatter
from matplotlib.lines import Line2D

# Set parameters common to all plots
gtCalParams             = np.array([1.0E3, -21.0E3])
colors                  = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00'] # [Blue, Teal, Red, Orange]
markers                 = ['v', 's', 'h', 'o']
axis_label_font_size    = 16
data_label_font_size    = 15
legend_font_size        = 14
marker_size             = 8
line_width              = 3
font_weight             = 'bold'
dots_per_inch           = 300

def plot_marginal_stats(marg_stats_files, labels, step_num, saveinterval, grouping, out_dirs):
    #  TODO: add location to savefile, add file format
    
    # Get array of iterations where marginal statistics were saved
    iterations = np.arange(start = saveinterval, 
                           stop = step_num + saveinterval, 
                           step = saveinterval, 
                           dtype = int)
    
    # Initialize figure
    fig, axes = plt.subplots(2, 2, figsize = (14, 5))
    axes = axes.flatten()

    # Loop over experiments in group
    for loopA in range(len(marg_stats_files)):
        
        # Loop over save intervals
        for loopB in range(len(iterations)):

            # Read file & unpack sample statistics
            stats = np.loadtxt(marg_stats_files[loopA] + str(iterations[loopB]))
            mean = stats[:, 0]  # Sample mean
            sd = stats[:, 1]    # Sample std. dev.

            # Plot mean of calibration parameters
            axes[0].plot(iterations[loopB], mean[0], markers[loopA], color = colors[loopA], markersize = marker_size)
            axes[1].plot(iterations[loopB], mean[1], markers[loopA], color = colors[loopA], markersize = marker_size)
            
            # Plot sd of calibration parameters
            axes[2].plot(iterations[loopB], sd[0], markers[loopA], color = colors[loopA], markersize = marker_size)
            axes[3].plot(iterations[loopB], sd[1], markers[loopA], color = colors[loopA], markersize = marker_size)
    
    # Set common labels for the y-axis
    for ax, ylabel in zip(axes, [r'$\mathbb{E}(\theta_1 \vert \mathcal{D})$',
                                 r'$\mathbb{E}(\theta_2 \vert \mathcal{D})$', 
                                 r'$\sqrt{\mathbb{V}(\theta_1 \vert \mathcal{D})}$', 
                                 r'$\sqrt{\mathbb{V}(\theta_2 \vert \mathcal{D})}$']):
        ax.set_ylabel(ylabel, fontsize = axis_label_font_size, fontweight = font_weight)

    # Add ground truth parameters to the sample average plots
    axes[0].axhline(gtCalParams[0], color = 'k', linewidth = line_width, label = r'$\theta_1$')
    axes[1].axhline(gtCalParams[1], color = 'k', linewidth = line_width, label = r'$\theta_2$')
    axes[0].legend(fontsize = legend_font_size)
    axes[1].legend(fontsize = legend_font_size)

    # Set x-axis label only for the bottom two subplots
    for ax in axes[-2:]:
        ax.set_xlabel('Iterations', fontsize = axis_label_font_size, fontweight = font_weight)

    # Set tick label font size
    for ax in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis = 'both', which = 'both', direction = 'in', top = True, right = True, labelsize = data_label_font_size)
    
    # Remove redundant tick labels
    axes[0].tick_params(axis = 'x', labelbottom = False)
    axes[1].tick_params(axis = 'x', labelbottom = False)

    # Set specific formatting for parameter 2
    axes[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))

    # Create a custom legend subplot
    legend_ax = fig.add_subplot(111, frame_on = False)
    legend_elements = [Line2D([0], [0], color = colors[i], marker = markers[i], markersize = marker_size, linewidth = 0, label = labels[i]) for i in range(len(labels))]
    legend = legend_ax.legend(handles = legend_elements, fontsize = legend_font_size, loc = 'upper center', ncol = len(labels), bbox_to_anchor=(0.5, 1.17))
    legend_ax.axis('off')

    plt.show()
    # # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.savefig(out_dirs + 'marginal_stats.png', bbox_inches = 'tight', dpi = dots_per_inch)

# =========
# MAIN CODE
# =========
if __name__ == '__main__':

    # Init parser
    parser = argparse.ArgumentParser(description = '.')

    # folder name
    parser.add_argument('-f', '--folder',
                        action = None,
                        # nargs='+',
                        const = None,
                        default = './',
                        type = str,
                        required = False,
                        help = 'Folder with experiment results',
                        metavar = '',
                        dest = 'folder_name')

    # experiment names
    parser.add_argument('-n', '--names',
                        action = None,
                        nargs = '+',
                        const = None,
                        default = './',
                        type = str,
                        required = True,
                        help = 'Names of numerical experiments',
                        dest = 'exp_names')
    
    # experiment names
    parser.add_argument('-l', '--labels',
                        action = None,
                        nargs = '+',
                        const = None,
                        default = './',
                        type = str,
                        required = True,
                        help = 'Names of numerical experiments to be used in plotting',
                        dest = 'labels')

    # iteration number = 1
    parser.add_argument('-i', '--iter',
                        action = None,
                        const = None,
                        default = 1,
                        type = int,
                        choices = None,
                        required = True,
                        help = 'Iteration number',
                        metavar = '',
                        dest = 'step_num')

    # plot format
    parser.add_argument('-p', '--picformat',
                        action = None,
                        const = None,
                        default = 'png',
                        type = str,
                        choices = ['png','pdf','jpg'],
                        required = False,
                        help = 'Output format for picture',
                        metavar = '',
                        dest = 'img_format')

    # Attribute to study
    parser.add_argument('-m', '--mode',
                        action = None,
                        const = None,
                        default = 'marginal_stats_group',
                        type = str,
                        choices = ['marginal_stats_group'],
                        required = False,
                        help = 'Type of plot/result to generate',
                        metavar = '',
                        dest = 'result_mode')
    
    # Grouping to study
    parser.add_argument('-g', '--grouping',
                        action = None,
                        const = None,
                        default = 'repeated',
                        type = str,
                        choices = ['TP1', 'TP15', 'prior', 'repeated measurements'],
                        required = False,
                        help = 'Type of group',
                        metavar = '',
                        dest = 'grouping')
    
    # Save interval
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
    out_dirs = [args.folder_name + x + '/' for x in args.exp_names]
    
    # Initialize
    lf_files                = []        # low-fidelity model posterior
    lf_dicr_files           = []        # low-fidelity model + discrepancy posterior
    lf_discr_noise_files    = []        # low-fidelity model + discrepancy + noise posterior
    #data_files              = []        # ?
    marg_stats_files        = []        # marginal statistics
    params_files            = []        # parameter posterior

    # Loop over experiments in grouping
    for loopA in range(len(out_dirs)):
        lf_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_outputs_lf_' + str(args.step_num))
        lf_dicr_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_outputs_lf+discr_' + str(args.step_num) )
        lf_discr_noise_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_outputs_lf+discr+noise_' + str(args.step_num))
        #data_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_data')
        marg_stats_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_marginal_stats_')
        params_files.append(out_dirs[loopA] + args.exp_names[loopA] + '_params_' + str(args.step_num))

    if(args.result_mode == 'marginal_stats_group'):
        plot_marginal_stats(marg_stats_files, args.labels, args.step_num, args.saveinterval, args.grouping, out_dirs)
    else:
        print('ERROR. Invalid execution mode')
        exit(-1)