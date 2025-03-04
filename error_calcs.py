# import numpy as np
# import os

# # Define file path
# filename = "TP15_no_disc_error_estimation_2"
# path = os.path.join("results", filename)
# iters = 10000

# # Load data
# mcmc = np.loadtxt(os.path.join(path, 'mcmc'))
# linfa = np.loadtxt(os.path.join(path, f"{filename}_params_{iters}"))

# # Ground truth parameters
# gt_params = np.array([1.0E3, -21.0E3, 0.05])

# # Calculate RMSE for mcmc
# diff_mcmc = (mcmc - gt_params) ** 2
# rmse_mcmc = np.sqrt(np.mean(diff_mcmc))

# # Calculate RMSE for linfa
# diff_linfa = (linfa - gt_params) ** 2
# rmse_linfa = np.sqrt(np.mean(diff_linfa))

# print("RMSE for MCMC:", rmse_mcmc)
# print("RMSE for Linfa:", rmse_linfa)

import numpy as np
import os

# Define file path
filename = "TP15_no_disc_error_estimation_aiche"
path = os.path.join("results", filename)
iters = 6000

# Load data
mcmc = np.loadtxt(os.path.join(path, 'mcmc'))
linfa = np.loadtxt(os.path.join(path, f"{filename}_params_{iters}"))

# Ground truth parameters
gt_params = np.array([1.0E3, -21.0E3, 0.05])

# Calculate RMSE for mcmc
diff_mcmc = (mcmc - gt_params) ** 2
rmse_mcmc = np.sqrt(np.mean(diff_mcmc))

# Calculate RMSE for linfa
diff_linfa = (linfa - gt_params) ** 2
rmse_linfa = np.sqrt(np.mean(diff_linfa))

# Calculate the average standard deviation of β samples for mcmc
std_dev_mcmc = np.std(mcmc, axis=0)  # Standard deviation for each parameter
avg_beta_sd_mcmc = np.mean(std_dev_mcmc)  # Average standard deviation across parameters

# Calculate the average standard deviation of β samples for linfa
std_dev_linfa = np.std(linfa, axis=0)  # Standard deviation for each parameter
avg_beta_sd_linfa = np.mean(std_dev_linfa)  # Average standard deviation across parameters

print("RMSE for MCMC:", rmse_mcmc)
print("RMSE for Linfa:", rmse_linfa)
print("Average standard deviation of β samples for MCMC:", avg_beta_sd_mcmc)
print("Average standard deviation of β samples for Linfa:", avg_beta_sd_linfa)
