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
plt.rc('text', usetex = False)

def plot_output(output_file,obs_file,out_dir,out_info):

  if(os.path.isfile(output_file) and os.path.isfile(obs_file)):
    tot_outputs = np.loadtxt(output_file).shape[1]
    print('Plotting posterior predictive samples...')
    for loopA in range(tot_outputs):
      for loopB in range(loopA+1, tot_outputs):
        plot_outputs(output_file,obs_file,loopA,loopB,out_dir,out_info,fig_format=args.img_format,use_dark_mode=args.use_dark_mode)
  else:
    if not(os.path.isfile(output_file)):
      print('File with posterior predictive samples not found: '+output_file)
    if not(os.path.isfile(obs_file)):
      print('File with observations not found: '+obs_file)

# def plot_output_discrepancy(output_type,output_file,obs_file,out_dir,out_info):

#   if(os.path.isfile(output_file) and os.path.isfile(obs_file)):
#     tot_outputs = np.loadtxt(output_file).shape[1]
#     print('Plotting posterior predictive samples...')
#     for loopA in range(tot_outputs):
#       for loopB in range(loopA+1, tot_outputs):
#         plot_outputs_discr(output_type,output_file,obs_file,loopA,loopB,out_dir,out_info,fig_format=args.img_format,use_dark_mode=args.use_dark_mode)
#   else:
#     if not(os.path.isfile(output_file)):
#       print('File with posterior predictive samples ('+output_type+') not found: '+output_file)
#     if not(os.path.isfile(obs_file)):
#       print('File with observations not found: '+obs_file)

def plot_log(log_file,out_dir,fig_format='png',use_dark_mode=False):
  log_data = np.loadtxt(log_file)

  # Set dark mode
  if(use_dark_mode):
    plt.style.use('dark_background')  

  # loss profile
  plt.figure(figsize=(2,2))
  # plt.semilogy(log_data[:,1],log_data[:,2],'b-')
  plt.plot(log_data[:,1],log_data[:,2],'b-')
  plt.xlabel('Iterations',fontsize=fs)
  plt.ylabel('log loss',fontsize=fs)
  plt.gca().tick_params(axis='both', labelsize=fs)
  plt.tight_layout()
  plt.savefig(out_dir+'log_plot.'+fig_format,bbox_inches='tight',dpi=200)
  plt.close()

def plot_params(param_data,LL_data,idx1,idx2,out_dir,out_info,fig_format='png',use_dark_mode=False):  
  ## TODO: plot density instead of log likelihood -- nothing will change in the plotting, LL file will be saved as a density
  ## DS to make changes in code
  ## schedule meeting for two weeks


  # Read data
  param_data = np.loadtxt(param_data)  
  dent_data  = np.loadtxt(LL_data)

  # Set dark mode
  if(use_dark_mode):
    plt.style.use('dark_background')

  # Remove an outlier
  if False:
    arg_max = np.argmax(np.exp(-dent_data))
    dent_data[arg_max] = np.average(dent_data)

  plt.figure(figsize=(3,2))
  plt.ylabel('log likelihood')
  plt.plot(np.arange(len(dent_data)), dent_data)
  plt.xlabel('sample')
  plt.savefig(out_dir+'dent_data' + out_info + '_'+str(idx1)+'_'+str(idx2)+'.'+fig_format,bbox_inches='tight',dpi=200)


  plt.figure(figsize=(3,2))
  plt.ylabel('likelihood')
  plt.plot(np.arange(len(dent_data)), np.exp(dent_data))
  plt.xlabel('sample')
  plt.savefig(out_dir+'exp_dent_data' + out_info + '_'+str(idx1)+'_'+str(idx2)+'.'+fig_format,bbox_inches='tight',dpi=200)
  
  plt.figure(figsize=(3,2))
  plt.scatter(param_data[:,idx1],param_data[:,idx2],s=1.5,lw=0,marker='o',c = np.exp(dent_data))
  # plt.scatter(param_data[:,idx1],param_data[:,idx2],s=1.5,lw=0,marker='o')
  plt.colorbar()
  plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
  plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
  plt.gca().tick_params(axis='both', labelsize=fs)
  plt.xlabel('$z_{K,'+str(idx1+1)+'}$',fontsize=fs)
  plt.ylabel('$z_{K,'+str(idx2+1)+'}$',fontsize=fs)  
  # # Set limits based on avg and std
  # avg_1 = np.mean(param_data[:,idx1])
  # std_1 = np.std(param_data[:,idx1])
  # avg_2 = np.mean(param_data[:,idx2])
  # std_2 = np.std(param_data[:,idx2])  
  # plt.xlim([avg_1-3*std_1,avg_1+3*std_1])
  # plt.ylim([avg_2-3*std_2,avg_2+3*std_2])
  plt.tight_layout()
  plt.savefig(out_dir+'params_plot_' + out_info + '_'+str(idx1)+'_'+str(idx2)+'.'+fig_format,bbox_inches='tight',dpi=200)
  plt.close()

def plot_outputs(sample_file,obs_file,idx1,idx2,out_dir,out_info,fig_format='png',use_dark_mode=False):  

  # Read data
  sample_data = np.loadtxt(sample_file)  
  obs_data    = np.loadtxt(obs_file) 

    # Set dark mode
  if(use_dark_mode):
    plt.style.use('dark_background')

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
  # Save plot
  plt.savefig(out_dir + out_info + '_'+str(idx1)+'_'+str(idx2)+'.'+fig_format,bbox_inches='tight',dpi=200)
  plt.close()

def plot_outputs_discr(out_type,sample_file,obs_file,idx1,idx2,out_dir,out_info,fig_format='png',use_dark_mode=False):  

  # Read data
  sample_data = np.loadtxt(sample_file)  
  obs_data    = np.loadtxt(obs_file) 

    # Set dark mode
  if(use_dark_mode):
    plt.style.use('dark_background')

  plt.figure(figsize=(2.5,2))
  # THE OUTPUT ARE FOR EACH TP!!!
  plt.scatter(sample_data[:,idx1],sample_data[:,idx2],s=2,c='b',marker='o',edgecolor=None,alpha=0.1)
  # THE OBSERVATIONS ARE FOR EACH TP!!!
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
  if(out_type == 'none'):
    file_prefix = 'data_plot_'
  elif(out_type == 'disc'):
    file_prefix = 'data_plot_disc_'
  elif(out_type == 'lf'):
    file_prefix = 'data_plot_lf_'
  elif(out_type == 'lf+disc'):
    file_prefix = 'data_plot_lf+disc_'
  elif(out_type == 'lf+disc+noise'):
    file_prefix = 'data_plot_lf+disc+noise_'
  # Save plot
  plt.savefig(out_dir + file_prefix + out_info + '_'+str(idx1) + '_' + str(idx2) + '.' + fig_format,bbox_inches='tight',dpi=200)
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

  # Parse Commandline Arguments
  args = parser.parse_args()

  # Set file name/path
  out_dir     = args.folder_name + args.exp_name + '/'
  log_file    = out_dir + 'log.txt'
  sample_file = out_dir + args.exp_name + '_samples_'    + str(args.step_num)
  param_file  = out_dir + args.exp_name + '_params_'     + str(args.step_num)
  LL_file     = out_dir + args.exp_name + '_logdensity_' + str(args.step_num)

  # Output files
  output_file               = out_dir + args.exp_name + '_outputs_'                + str(args.step_num)
  output_file_lf            = out_dir + args.exp_name + '_outputs_lf_'             + str(args.step_num)
  output_file_lf_disc       = out_dir + args.exp_name + '_outputs_lf+discr_'       + str(args.step_num)
  output_file_lf_disc_noise = out_dir + args.exp_name + '_outputs_lf+discr+noise_' + str(args.step_num)

  # Observation file
  obs_file = out_dir + args.exp_name + '_data'
  out_info = args.exp_name + '_' + str(args.step_num)

  # Check is this is a case with discrepancy or not
  if(os.path.isfile(output_file_lf_disc)):
    is_discrepancy = True
  else:
    is_discrepancy = False

  # Plot loss profile
  if(os.path.isfile(log_file)):
    print('Plotting log...')
    plot_log(log_file,out_dir,fig_format=args.img_format,use_dark_mode=args.use_dark_mode)
  else:
    print('Log file not found: '+log_file)

  # Plot posterior samples
  if(os.path.isfile(param_file) and os.path.isfile(LL_file)):
    tot_params  = np.loadtxt(param_file).shape[1]
    print('Plotting posterior samples...')
    for loopA in range(tot_params):
      for loopB in range(loopA+1, tot_params):
        plot_params(param_file,LL_file,loopA,loopB,out_dir,out_info,fig_format=args.img_format,use_dark_mode=args.use_dark_mode)
  else:
    print('File with posterior samples not found: '+param_file)
    print('File with log-density not found: '+LL_file)

  # Plot posterior predictive distribution with observations 
  plot_output(output_file,obs_file,out_dir,out_info)
      