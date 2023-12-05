import matplotlib.pyplot as plt
import numpy as np
import os

directory = 'results'
iteration = '25000'
experiment = 'test_lf_with_disc_hf_data_prior_TP1'
filename1 = 'outputs_lf'
filename2 = 'outputs_lf+discr'
filename3 = 'outputs_lf+discr+noise'

path1 = os.path.join(directory, experiment, experiment + '_' + filename1 + '_' + iteration)
path2 = os.path.join(directory, experiment, experiment + '_' + filename2 + '_' + iteration)
path3 = os.path.join(directory, experiment, experiment + '_' + filename3 + '_' + iteration)

lf_model = np.loadtxt(path1)
lf_model_plus_disc = np.loadtxt(path2)
lf_model_plus_disc_plus_noise = np.loadtxt(path3)

plt.hist(lf_model[:,0], label = 'LF')
plt.hist(lf_model_plus_disc[:,0],  label = 'LF + disc')
plt.hist(lf_model_plus_disc_plus_noise[:,0],  label = 'LF + disc + noise')
plt.xlabel('Coverage')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# shape: number of ___?___ (15) x Batch size (5000)
print(np.shape(lf_model))