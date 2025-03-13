import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Set global plot settings
# Set global plot settings
plt.rcParams['figure.dpi']          = 300
plt.rcParams['axes.labelsize']      = 16
plt.rcParams['xtick.labelsize']     = 15
plt.rcParams['ytick.labelsize']     = 15
plt.rcParams['legend.fontsize']     = 12
plt.rcParams['lines.linewidth']     = 3
plt.rcParams['lines.markersize']    = 16
plt.rcParams['axes.labelweight']    = 'bold'
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'
plt.rcParams['xtick.top']           = True
plt.rcParams['ytick.right']         = True
plt.rcParams['savefig.bbox']        = 'tight'

def plot_pairwise(param_data, out_dir, out_info, fig_format='png'):
    """Generate a pair plot with marginal KDEs and scatter plots for pairwise distributions using Matplotlib."""
    # Load parameter data
    param_samples = np.loadtxt(param_data)
    param_samples_mcmc = np.loadtxt(os.path.join(out_dir, "mcmc"))

    true_params = [1.0, -21.0, 0.05]
    param_names = ['Pre-exp. Factor', 'Ads. Energy', 'Noise s.d. Ratio']
    units = ['[kPa]', r'[kJ$\cdot$mol$^{-1}$]', '[ ]']

    param_samples[:,0] = param_samples[:,0] / 1000
    param_samples[:,1] = param_samples[:,1] / 1000
    param_samples_mcmc[:,0] = param_samples_mcmc[:,0] / 1000
    param_samples_mcmc[:,1] = param_samples_mcmc[:,1] / 1000

    # Number of parameters
    num_params = param_samples.shape[1]

    # Create figure
    fig, axes = plt.subplots(num_params, num_params, figsize=(4*num_params, 4*num_params))

    # Loop through each pair of parameters
    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i, j]

            if i == j:
                # Diagonal: Marginal KDE
                kde = gaussian_kde(param_samples_mcmc[:, i])
                x = np.linspace(param_samples_mcmc[:, i].min(), param_samples_mcmc[:, i].max(), 100)
                ax.hist(param_samples_mcmc[:, i], density = True, edgecolor = 'k', alpha = 0.5, label = "FA-VI")
                ax.plot(x, kde(x), 'r-', label = "MH MCMC")
                ax.axvline(true_params[i], color = "limegreen", label = "True")

                # X-label only on the last row
                if i == num_params - 1:
                    ax.set_xlabel(param_names[i]+f", $z_{i+1}$ {units[i]}")
                else:
                    ax.set_xticklabels([])

                # Special case: Ensure y-label in the upper-leftmost plot (first row, first column)
                if j == 0 and i == 0:
                    ax.set_ylabel("Density")
                    ax.yaxis.set_visible(True)  # Ensure y-axis is visible
                    ax.legend()
                else:
                    ax.set_yticklabels([])  # Hide other y-ticks
                    ax.yaxis.set_visible(False)

            elif i > j:
                # Lower triangle: Scatter plot with contours
                x_data, y_data = param_samples[:, j], param_samples[:, i]
                ax.scatter(x_data, y_data, s=10, alpha=0.5, label = 'FA-VI')

                # KDE Contours
                kde = gaussian_kde(np.vstack([x_data, y_data]))
                x = np.linspace(x_data.min(), x_data.max(), 100)
                y = np.linspace(y_data.min(), y_data.max(), 100)
                X, Y = np.meshgrid(x, y)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde(positions), X.shape)
                ax.contour(X, Y, Z, colors="r")
                ax.plot(true_params[j], true_params[i], color = "limegreen", marker = "*", linestyle='None', label = "True")

                # Add y-labels **only in the first column**
                if j == 0:
                    ax.set_ylabel(param_names[i]+f", $z_{i+1}$ {units[i]}")
                    if i==1:
                      ax.legend()
                else:
                    ax.set_yticklabels([])

                # Add x-labels **only in the last row**
                if i == num_params - 1:
                    ax.set_xlabel(param_names[j]+f", $z_{j+1}$ {units[j]}")
                else:
                    ax.set_xticklabels([])
            
            else:
                # Upper triangle: Hide plots
                ax.set_visible(False)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pair_plot_{out_info}.{fig_format}'))
    plt.close()

# =========
# MAIN CODE
# =========
if __name__ == '__main__':  

    # Init parser
    parser = argparse.ArgumentParser(description='Generate pair plot for posterior samples.')

    # folder name
    parser.add_argument('-f', '--folder', default='./', type=str, required=False,
                        help='Folder with experiment results', metavar='', dest='folder_name')

    # experiment name
    parser.add_argument('-n', '--name', default='./', type=str, required=True,
                        help='Name of numerical experiment', metavar='', dest='exp_name')

    # iteration number
    parser.add_argument('-i', '--iter', default=1, type=int, required=True,
                        help='Iteration number', metavar='', dest='step_num')

    # plot format
    parser.add_argument('-p', '--picformat', default='png', type=str, choices=['png','pdf','jpg'],
                        required=False, help='Output format for picture', metavar='', dest='img_format')

    # Enable dark mode for pictures
    parser.add_argument('-d', '--dark', action='store_true', default=False, required=False,
                        help='Generate pictures for dark background', dest='use_dark_mode')

    # Parse Commandline Arguments
    args = parser.parse_args()

    # Set file name/path
    out_dir     = os.path.join(args.folder_name, args.exp_name)
    param_file  = os.path.join(out_dir, f"{args.exp_name}_params_{args.step_num}")
    out_info    = f"{args.exp_name}_{args.step_num}"

    # Run original plotting functions if data files exist
    if os.path.isfile(param_file):
        print('Generating pair plot...')
        plot_pairwise(param_file, out_dir, out_info, fig_format=args.img_format)
    else:
        print(f'File with posterior samples not found: {param_file}')

# import os
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
# from scipy.stats import gaussian_kde
# from matplotlib.ticker import ScalarFormatter, MaxNLocator

# # Set global plot settings
# plt.rcParams['figure.figsize']      = (8, 6)
# plt.rcParams['figure.dpi']          = 300
# plt.rcParams['axes.labelsize']      = 16
# plt.rcParams['xtick.labelsize']     = 15
# plt.rcParams['ytick.labelsize']     = 15
# plt.rcParams['legend.fontsize']     = 12
# plt.rcParams['lines.linewidth']     = 3
# plt.rcParams['lines.markersize']    = 16
# plt.rcParams['axes.labelweight']    = 'bold'
# plt.rcParams['xtick.direction']     = 'in'
# plt.rcParams['ytick.direction']     = 'in'
# plt.rcParams['xtick.top']           = True
# plt.rcParams['ytick.right']         = True
# plt.rcParams['savefig.bbox']        = 'tight'

# def plot_marginals(param_data, idx1, fig_format='png'):

#   # Read in data
#   gt_params = [1000, -21.0E3, 0.05]
#   mcmc_data = np.loadtxt(os.path.join(out_dir, 'mcmc'))[:,idx1]
#   linfa_data = np.loadtxt(param_data)[:, idx1]
  
#   # Pick ranges
#   lb = [np.min(mcmc_data), np.min(linfa_data)]
#   ub = [np.max(mcmc_data), np.max(linfa_data)]
#   x = np.linspace(np.min(lb), np.max(ub), 100)
  
#   # MCMC KDE
#   kde_mcmc = gaussian_kde(mcmc_data)
#   mcmc_results = kde_mcmc(x)

#   # LINFA KDE
#   kde_linfa = gaussian_kde(linfa_data)
#   linfa_results = kde_linfa(x)

#   # Plot results
#   plt.figure(figsize=(6, 6))
#   plt.hist(linfa_data, alpha = 0.45, label = 'FAVI', density = True, edgecolor='black')
#   plt.plot(x, mcmc_results, 'm-', label = "MH MCMC")
#   # plt.plot(x, linfa_results, 'b--', label = "FAVI")
#   plt.axvline(gt_params[idx1], color = 'k', linestyle = ':', label = "Nominal")
#   if idx1 == 2:
#     plt.xlabel(r's.d. Ratio, $z_{K,'+str(idx1+1)+'}$')
#   else:
#      plt.xlabel(r'$z_{K,'+str(idx1+1)+'}$')
#   plt.ylabel("Density")
#   plt.legend(fontsize = 14)
#   plt.savefig(os.path.join(out_dir,'marginal_params_plot_' + out_info + '_'+str(idx1)+'.'+fig_format))
#   plt.close()

# def plot_marginals_aiche(param_data, idx1, fig_format='png'):

#   '''TODO: remove later'''

#   # Read in data
#   mcmc_data = np.loadtxt(os.path.join(out_dir, 'mcmc'))[:,idx1]
  
#   # Pick ranges
#   x = np.linspace(np.min(mcmc_data), np.max(mcmc_data), 100)
  
#   # KDE
#   kde_mcmc = gaussian_kde(mcmc_data)
#   mcmc_results = kde_mcmc(x)

#   # Plot results
#   plt.figure(figsize=(5, 5))
#   plt.hist(mcmc_data, alpha = 0.45, label = 'Samples', density = True, edgecolor='black')
#   plt.plot(x, mcmc_results, 'm-', label = "Estimate")

#   plt.xlabel(r'$z_k$')
#   plt.ylabel("Density")
#   plt.legend(loc = "upper right", fontsize = 14)
#   plt.savefig(os.path.join(out_dir,'marginal_params_plot_aiche_' + out_info + '_'+str(idx1)+'.'+fig_format))
#   plt.close()


# def plot_params(param_data, LL_data, idx1, idx2, out_dir, out_info, fig_format='png', use_dark_mode = False):  

#   gt_params = [1000, -21.0E3, 0.05]

#   # Read data
#   linfa_data = np.loadtxt(param_data)
#   linfa_dent_data  = np.loadtxt(LL_data)
#   linfa_samples = np.vstack([linfa_data[:,idx1], linfa_data[:,idx2]])  # Transpose to get shape (n, d)
#   mcmc_data = np.loadtxt(os.path.join(out_dir, 'mcmc'))
#   mcmc_1_data = mcmc_data[:,idx1]
#   mcmc_2_data = mcmc_data[:,idx2]
#   mcmc_samples = np.vstack([mcmc_1_data, mcmc_2_data])

#   # Create a grid to evaluate KDE
#   lb0 = [np.min(mcmc_samples[0]), np.min(linfa_samples[0])]
#   lb1 = [np.min(mcmc_samples[1]), np.min(linfa_samples[1])]
#   ub0 = [np.max(mcmc_samples[0]), np.max(linfa_samples[0])]
#   ub1 = [np.max(mcmc_samples[1]), np.max(linfa_samples[1])]

#   x = np.linspace(np.min(lb0), np.max(ub0), 100)
#   y = np.linspace(np.min(lb1), np.max(ub1), 100)
#   X, Y = np.meshgrid(x, y)
#   positions = np.vstack([X.ravel(), Y.ravel()])

#   # KDE
#   mcmc_kde = gaussian_kde(mcmc_samples)
#   linfa_kde = gaussian_kde(linfa_samples)
#   mcmc_est = np.reshape(mcmc_kde(positions), X.shape)
#   linfa_est = np.reshape(linfa_kde(positions), X.shape)

#   mean_1, std_dev_1 = np.mean(linfa_samples[0]), np.std(linfa_samples[0])
#   mean_2, std_dev_2 = np.mean(linfa_samples[1]), np.std(linfa_samples[1])

#   # Plot
#   plt.figure()
#   # plt.scatter(linfa_samples[0],  linfa_samples[1], lw = 0, s = 40, marker = 'o', c = np.exp(linfa_dent_data), cmap = "Blues_r")
#   # plt.contour(X, Y, linfa_est, colors = "r")  # Dashed blue lines for linfa_est
    

#   if idx1 == 0 and idx2 == 1:
#     mcmc_est[:,1] = mcmc_est[:,1]/1000
#     mean_2 = mean_2 / 1000
#     std_dev_2 = std_dev_2 / 1000
#     plt.xlabel(r'Std. Pressure, $z_{K,'+str(idx1 + 1)+'}$ [Pa]')
#     plt.ylabel(r'Ads. Energy, $z_{K,' + str(idx2 + 1) + '}$ [kJ $\cdot$ mol$^{-1}$]')
#     plt.hexbin(linfa_samples[0], linfa_samples[1]/1000, gridsize = 50, bins = 'log', cmap = "Blues_r")
#     plt.colorbar(label = "Frequency")
#     plt.contour(X, Y/1000, mcmc_est, colors = "m")  # Solid red lines for mcmc_est
#     plt.plot(gt_params[idx1], gt_params[idx2]/1000, '*', color = '#FFFF33', markeredgecolor = 'k', markersize = 25)
  
    
#   else:
#     plt.xlabel(r'$z_{K,'+str(idx1 + 1)+'}$')
#     plt.ylabel(r'$z_{K,'+str(idx2 + 1)+'}$')
#     plt.plot(gt_params[idx1], gt_params[idx2], '*', color = '#FFFF33', markeredgecolor = 'k', markersize = 25)
#     plt.hexbin(linfa_samples[0], linfa_samples[1], gridsize = 50, bins = 'log', cmap = "Blues_r")
#     plt.colorbar(label = "Frequency")
#     plt.contour(X, Y, mcmc_est, colors = "m")  # Solid red lines for mcmc_est
    
  
#   plt.xlim(mean_1 - 3 * std_dev_1, mean_1 + 3 * std_dev_1)
#   plt.ylim(mean_2 - 3 * std_dev_2, mean_2 + 3 * std_dev_2)
#   plt.savefig(os.path.join(out_dir,'params_plot_' + out_info + '_'+str(idx1)+'_'+str(idx2)+'.'+fig_format))
  
#   # plt.hexbin(param_data[:,idx1], param_data[:,idx2], bins = 'log') #, lw = 0, s = 7, marker = 'o')#, c = np.exp(-dent_data))
  
#   plt.close()

# # =========
# # MAIN CODE
# # =========
# if __name__ == '__main__':  

#   # Init parser
#   parser = argparse.ArgumentParser(description='.')

#   # folder name
#   parser.add_argument('-f', '--folder',
#                       action=None,
#                       const=None,
#                       default='./',
#                       type=str,
#                       required=False,
#                       help='Folder with experiment results',
#                       metavar='',
#                       dest='folder_name')

#   # folder name
#   parser.add_argument('-n', '--name',
#                       action=None,
#                       const=None,
#                       default='./',
#                       type=str,
#                       required=True,
#                       help='Name of numerical experiment',
#                       metavar='',
#                       dest='exp_name')

#   # iteration number = 1
#   parser.add_argument('-i', '--iter',
#                       action=None,
#                       const=None,
#                       default=1,
#                       type=int,
#                       choices=None,
#                       required=True,
#                       help='Iteration number',
#                       metavar='',
#                       dest='step_num')
  
#   # plot format
#   parser.add_argument('-p', '--picformat',
#                       action=None,
#                       const=None,
#                       default='png',
#                       type=str,
#                       choices=['png','pdf','jpg'],
#                       required=False,
#                       help='Output format for picture',
#                       metavar='',
#                       dest='img_format')

#   # Enable dark mode for pictures
#   parser.add_argument('-d', '--dark',
#                       action='store_true',
#                       default=False,
#                       required=False,
#                       help='Generate pictures for dark background',
#                       dest='use_dark_mode')

#   # Parse Commandline Arguments
#   args = parser.parse_args()

#   # Set file name/path
#   out_dir     = os.path.join(args.folder_name,args.exp_name)
#   param_file  = os.path.join(out_dir,args.exp_name + '_params_'     + str(args.step_num))
#   LL_file     = os.path.join(out_dir,args.exp_name + '_logdensity_' + str(args.step_num))
#   out_info    = args.exp_name + '_' + str(args.step_num)

#   # Plot 2D slice of posterior samples
#   if(os.path.isfile(param_file) and os.path.isfile(LL_file)):
#     tot_params  = np.loadtxt(param_file).shape[1] # extract total number of parameters inferred
#     print('Plotting posterior samples...')
#     for loopA in range(tot_params): # loop over total number of parameters
#       plot_marginals(param_file, loopA)
#       plot_marginals_aiche(param_file, loopA)
#       for loopB in range(loopA+1, tot_params): # get next parameter
#         plot_params(param_file,LL_file,loopA,loopB,out_dir,out_info,fig_format=args.img_format,use_dark_mode=args.use_dark_mode)
#   else:
#     print('File with posterior samples not found: '+param_file)
#     print('File with log-density not found: '+LL_file)
