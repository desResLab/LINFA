import numpy as np
import torch
from linfa.models.TrivialModels import Trivial
from linfa.models.highdimModels import Highdim
from linfa.models.circuitModels import rcModel
from linfa.models.circuitModels import rcrModel

def eval_post_Trivial():
  # Define model
  model = Trivial()
  model.data = np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/trivial/data_trivial.txt')
  # Read samples
  param_data = torch.from_numpy(np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/trivial/update/samples30000'))
  # Evaluate posterior
  print('Evaluating models...')
  res = model.den_t(param_data,surrogate=False).numpy()
  # Save to file
  np.savetxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/trivial/update/Trivial_MAF_DENT.txt',res[:,0])

def eval_post_Highdim(runType='3p5'):
  # Define model
  model = Highdim()
  model.data = np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/hidim/data_highdim.txt')
  # Read samples
  if(runType=='3p5'):
    param_data = torch.from_numpy(np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/hidim/3p5/samples25000'))
  elif(runType=='4p5'):
    param_data = torch.from_numpy(np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/hidim/4p5/samples25000'))
  else:
    print('Invalid runType.')
    exit(-1)
  # Evaluate posterior
  print('Evaluating models...')
  res = model.den_t(param_data,surrogate=False).numpy()
  # Save to file
  if(runType=='3p5'):
    np.savetxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/hidim/3p5/Highdim_MAF_DENT.txt',res[:,0])
  elif(runType=='4p5'):
    np.savetxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/hidim/4p5/Highdim_MAF_DENT.txt',res[:,0])
  else:
    print('Invalid runType.')
    exit(-1)

def eval_post_RC():
  # Define model
  cycleTime = 1.07
  totalCycles = 10
  forcing = np.loadtxt('source/data/inlet.flow')
  model = rcModel(cycleTime, totalCycles, forcing)  # RC Model Defined
  model.data = np.loadtxt('source/data/data_rc.txt')
  # Read samples
  param_data = torch.from_numpy(np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/RC/update/samples25000'))
  # Evaluate posterior
  print('Evaluating models...')
  res = model.den_t(param_data,surrogate=False).numpy()
  # Save to file
  np.savetxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/RC/update/RC_MAF_DENT.txt',res[:,0])

def eval_post_RCR():
  # Define model
  cycleTime = 1.07
  totalCycles = 10
  forcing = np.loadtxt('../resource/data/inlet.flow')
  model = rcrModel(cycleTime, totalCycles, forcing)  # RCR Model Defined
  model.data = np.loadtxt('../resource/data/data_rc.txt')
  # Read samples
  param_data = torch.from_numpy(np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/RCR/update/samples25000'))
  # Evaluate posterior
  print('Evaluating models...')
  res = model.den_t(param_data,surrogate=False).numpy()
  # Save to file
  np.savetxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/RCR/update/RCR_MAF_DENT.txt',res[:,0])


# Main
if __name__ == '__main__':
  
  # eval_post_Trivial()

  # runType='4p5'
  # eval_post_Highdim(runType)

  # eval_post_RC()

  eval_post_RCR()


