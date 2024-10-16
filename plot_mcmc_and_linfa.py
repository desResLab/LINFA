import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from scipy.stats import gaussian_kde

# Set global plot settings
plt.rcParams['figure.figsize']      = (8, 6)
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

def plot_marginals(param_data, idx1, fig_format='png'):

  param_data = np.loadtxt(param_data)  # [5000, 3]
  mcmc_1_data = np.loadtxt(os.path.join(out_dir,'posterior_samples_'+str(idx1)))
  gt_params = [1000, -21.0E3, 0.05]

  plt.figure(figsize=(6, 6))
  plt.hist(param_data[:,idx1], color = 'blue', alpha = 0.25, label = 'LINFA', density = True)
  plt.hist(mcmc_1_data, color = 'red', alpha = 0.25, label = 'MCMC', density = True)
  plt.axvline(gt_params[idx1], color = 'r')
  plt.xlabel(r'$\theta_{K,'+str(idx1+1)+'}$')
  plt.legend()
  plt.savefig(os.path.join(out_dir,'marginal_params_plot_' + out_info + '_'+str(idx1)+'.'+fig_format))
  plt.close()


def plot_params(param_data, LL_data, idx1, idx2, out_dir, out_info, fig_format='png', use_dark_mode=False):  

  # Read data
  param_data = np.loadtxt(param_data)  # [5000, 3]
  dent_data  = np.loadtxt(LL_data)     # [5000, ]
  mcmc_1_data = np.loadtxt(os.path.join(out_dir,'posterior_samples_'+str(idx1)))
  mcmc_2_data = np.loadtxt(os.path.join(out_dir,'posterior_samples_'+str(idx2)))
  gt_params = [1000, -21.0E3, 0.15]

  # Combine MCMC samples
  samples = np.vstack([mcmc_1_data, mcmc_2_data])  # Transpose to get shape (n, d)

  # Perform KDE
  kde = gaussian_kde(samples)

  # Create a grid to evaluate KDE
  x = np.linspace(samples[0].min(), samples[0].max(), 100)
  y = np.linspace(samples[1].min(), samples[1].max(), 100)
  X, Y = np.meshgrid(x, y)
  positions = np.vstack([X.ravel(), Y.ravel()])
  Z = np.reshape(kde(positions), X.shape)

  # Plot
  plt.figure(figsize=(8, 6))
  plt.contour(X, Y, Z)
  plt.scatter(param_data[:,idx1], param_data[:,idx2], lw = 0, s = 7, marker = 'o', c = np.exp(dent_data))
  plt.plot(gt_params[idx1], gt_params[idx2], 'r*')
  plt.colorbar()
  plt.xlabel(r'$\theta_{K,'+str(idx1+1)+'}$')
  plt.ylabel(r'$\theta_{K,'+str(idx2+1)+'}$')
  plt.savefig(os.path.join(out_dir,'params_plot_' + out_info + '_'+str(idx1)+'_'+str(idx2)+'.'+fig_format))
  plt.close()

# =========
# MAIN CODE
# =========
if __name__ == '__main__':  

  # Init parser
  parser = argparse.ArgumentParser(description='.')

  # folder name
  parser.add_argument('-f', '--folder',
                      action=None,
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

  # Parse Commandline Arguments
  args = parser.parse_args()

  # Set file name/path
  out_dir     = os.path.join(args.folder_name,args.exp_name)
  param_file  = os.path.join(out_dir,args.exp_name + '_params_'     + str(args.step_num))
  LL_file     = os.path.join(out_dir,args.exp_name + '_logdensity_' + str(args.step_num))
  out_info    = args.exp_name + '_' + str(args.step_num)

  # Plot 2D slice of posterior samples
  if(os.path.isfile(param_file) and os.path.isfile(LL_file)):
    tot_params  = np.loadtxt(param_file).shape[1] # extract total number of parameters inferred
    print('Plotting posterior samples...')
    for loopA in range(tot_params): # loop over total number of parameters
      plot_marginals(param_file, loopA)
      for loopB in range(loopA+1, tot_params): # get next parameter
        plot_params(param_file,LL_file,loopA,loopB,out_dir,out_info,fig_format=args.img_format,use_dark_mode=args.use_dark_mode)
  else:
    print('File with posterior samples not found: '+param_file)
    print('File with log-density not found: '+LL_file)
