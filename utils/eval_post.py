import numpy as np
import torch
from models.circuitModels import rcModel
from models.TrivialModels import Trivial
from models.highdimModels import Highdim

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

def eval_post_Highdim():
  # Define model
  model = Highdim()
  model.data = np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/hidim/data_highdim.txt')
  # Read samples
  param_data = torch.from_numpy(np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/hidim/3p5/samples25000'))
  # Evaluate posterior
  print('Evaluating models...')
  res = model.den_t(param_data,surrogate=False).numpy()
  # Save to file
  np.savetxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/hidim/3p5/Highdim_MAF_DENT.txt',res[:,0])

def eval_post_RC():
  # Define model
  cycleTime = 1.07
  totalCycles = 10
  forcing = np.loadtxt('source/data/inlet.flow')
  model = rcModel(cycleTime, totalCycles, forcing)  # RCR Model Defined
  model.data = np.loadtxt('source/data/data_rc.txt')
  # Read samples
  param_data = torch.from_numpy(np.loadtxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/RC/update/samples25000'))
  # Evaluate posterior
  print('Evaluating models...')
  res = model.den_t(param_data,surrogate=False).numpy()
  # Save to file
  np.savetxt('/home/dschiava/Documents/02_Documents/05_Articles/23_linfa_paper/figures_papers/RC/update/RC_MAF_DENT.txt',res[:,0])

# Main
if __name__ == '__main__':
  
  # eval_post_Trivial()

  eval_post_Highdim()

  # eval_post_RC()


