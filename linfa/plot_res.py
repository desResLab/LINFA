import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Plot format
fs=8
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)

def plot_log(log_file,out_dir):
  log_data = np.loadtxt(log_file)
  # loss profile
  plt.figure(figsize=(2,2))
  plt.semilogy(log_data[:,1],log_data[:,2],'b-')
  plt.xlabel('Iterations',fontsize=fs)
  plt.ylabel('log loss',fontsize=fs)
  plt.gca().tick_params(axis='both', labelsize=fs)
  plt.tight_layout()
  plt.savefig(out_dir+'log_plot.pdf',bbox_inches='tight')
  plt.close()

def plot_params(param_data,LL_data,idx1,idx2,out_dir,out_info):  
  param_data = np.loadtxt(param_data)  
  dent_data  = np.loadtxt(LL_data)
  # Plot figure
  plt.figure(figsize=(3,2))
  plt.scatter(param_data[:,idx1],param_data[:,idx2],s=1.5,lw=0,marker='o',c=np.exp(dent_data))
  plt.colorbar()
  plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
  plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
  plt.gca().tick_params(axis='both', labelsize=fs)
  plt.xlabel('$z_{K,'+str(idx1+1)+'}$',fontsize=fs)
  plt.ylabel('$z_{K,'+str(idx2+1)+'}$',fontsize=fs)  
  # Set limits based on avg and std
  avg_1 = np.mean(param_data[:,idx1])
  std_1 = np.std(param_data[:,idx1])
  avg_2 = np.mean(param_data[:,idx2])
  std_2 = np.std(param_data[:,idx2])  
  plt.xlim([avg_1-3*std_1,avg_1+3*std_1])
  plt.ylim([avg_2-3*std_2,avg_2+3*std_2])
  plt.tight_layout()
  plt.savefig(out_dir+'params_plot_' + out_info + '_'+str(idx1)+'_'+str(idx2)+'.pdf',bbox_inches='tight')
  plt.close()

def plot_outputs(sample_file,obs_file,idx1,idx2,out_dir,out_info):  
  sample_data = np.loadtxt(sample_file)  
  obs_data    = np.loadtxt(obs_file)  
  plt.figure(figsize=(2.5,2))
  plt.scatter(sample_data[:,idx1],sample_data[:,idx2],s=2,c='b',marker='o',edgecolor=None,alpha=0.1)
  plt.scatter(obs_data[idx1,:],obs_data[idx2,:],s=3,c='r',alpha=1,zorder=99)
  plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
  plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
  plt.gca().tick_params(axis='both', labelsize=fs)
  plt.xlabel('$x_{'+str(idx1+1)+'}$',fontsize=fs)
  plt.ylabel('$x_{'+str(idx2+1)+'}$',fontsize=fs)  
  # Set limits based on avg and std
  avg_1 = np.mean(sample_data[:,idx1])
  std_1 = np.std(sample_data[:,idx1])
  avg_2 = np.mean(sample_data[:,idx2])
  std_2 = np.std(sample_data[:,idx2])  
  plt.xlim([avg_1-3*std_1,avg_1+3*std_1])
  plt.ylim([avg_2-3*std_2,avg_2+3*std_2])
  plt.tight_layout()
  plt.savefig(out_dir+'data_plot_' + out_info + '_'+str(idx1)+'_'+str(idx2)+'.pdf',bbox_inches='tight')
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

  # numRealizations = 1
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

  # Parse Commandline Arguments
  args = parser.parse_args()

  # Set file name/path
  out_dir     = args.folder_name + args.exp_name + '/'
  log_file    = out_dir + 'log.txt'
  sample_file = out_dir + args.exp_name + '_samples_'    + str(args.step_num)
  param_file  = out_dir + args.exp_name + '_params_'     + str(args.step_num)
  LL_file     = out_dir + args.exp_name + '_logdensity_' + str(args.step_num)
  output_file = out_dir + args.exp_name + '_outputs_'    + str(args.step_num)
  obs_file    = out_dir + args.exp_name + '_data'
  out_info    = args.exp_name + '_' + str(args.step_num)

  # Plot loss profile
  if(os.path.isfile(log_file)):
    print('Plotting log...')
    plot_log(log_file,out_dir)
  else:
    print('Log file not found: '+log_file)

  # Plot 2D slice of posterior samples
  if(os.path.isfile(param_file) and os.path.isfile(LL_file)):
    tot_params  = np.loadtxt(param_file).shape[1]
    print('Plotting posterior samples...')
    for loopA in range(tot_params):
      for loopB in range(loopA+1, tot_params):
        plot_params(param_file,LL_file,loopA,loopB,out_dir,out_info)
  else:
    print('File with posterior samples not found: '+param_file)
    print('File with log-density not found: '+LL_file)

  # Plot 2D slice of outputs and observations
  if(os.path.isfile(output_file) and os.path.isfile(obs_file)):
    tot_outputs = np.loadtxt(output_file).shape[1]
    print('Plotting posterior predictive samples...')
    for loopA in range(tot_outputs):
      for loopB in range(loopA+1, tot_outputs):
        plot_outputs(output_file,obs_file,loopA,loopB,out_dir,out_info)
  else:
    print('File with posterior predictive samples not found: '+output_file)
    print('File with observations not found: '+obs_file)
