import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import torch
import time  # Import time module
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Abbreviations
tfd = tfp.distributions
from linfa.models.discrepancy_models import PhysChem_error

# Set global plot settings
plt.rcParams['figure.figsize']      = (8, 6)
plt.rcParams['figure.dpi']          = 300
plt.rcParams['axes.labelsize']      = 16
plt.rcParams['xtick.labelsize']     = 15
plt.rcParams['ytick.labelsize']     = 15
plt.rcParams['legend.fontsize']     = 12
plt.rcParams['lines.linewidth']     = 1
plt.rcParams['lines.markersize']    = 16
plt.rcParams['axes.labelweight']    = 'bold'
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'
plt.rcParams['xtick.top']           = True
plt.rcParams['ytick.right']         = True
plt.rcParams['savefig.bbox']        = 'tight'

def run_test(num_results, num_burnin_steps):
    
    # Set variable grid
    variable_inputs = [[350.0, 400.0, 450.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Assign as experiment model
    model = PhysChem_error(variable_inputs)

    # Read data
    model.data = np.loadtxt('observations.csv', delimiter = ',', skiprows = 1)
    data_mean = np.mean(model.data[:, 2:]) # Calculate mean of responses
        
    # Form tensors for variables and results in observations
    var_grid_in = tf.convert_to_tensor(model.data[:, :2], dtype = tf.float32)
    var_grid_out = tf.convert_to_tensor(model.data[:, 2:], dtype = tf.float32)

    def target_log_prob_fn(theta, log_sigma):
        
        # Undo transformations of parameters
        theta1 = tf.exp(theta[0])                       # Log transformation of pre-exp. factor
        theta2 = -30E3 + (tf.sigmoid(theta[1]) * 15E3)  # Tanh transformation of ads. energy
        sigma = tf.exp(log_sigma) * data_mean           # Calculate SNR as parameter

        # Assign priors
        ## Normal priors on low-fidlity model
        prior_theta1 = tfd.Normal(loc = 1.0E3, scale = 100.0).log_prob(theta1)
        prior_theta2 = tfd.Normal(loc = -21.0E3, scale = 500.0).log_prob(theta2) 
        prior_theta = prior_theta1 + prior_theta2   # Multivariate normal prior on calibration parameters
        ## Beta prior on SNR
        prior_sigma = tfd.Beta(1.0, 19.0).log_prob(sigma)

        # Convert data types
        theta_np = np.array([theta1.numpy(), theta2.numpy()])
        sigma_np = sigma.numpy()
        cal_inputs = torch.tensor(np.hstack([theta_np, sigma_np]), dtype = torch.float32)
        
        # Solve low fidelity model
        y_pred_torch = model.solve_t(cal_inputs)
        y_pred_np = y_pred_torch.detach().numpy()
        y_pred_tf = tf.convert_to_tensor(y_pred_np, dtype = tf.float32)

        # Compute likelihood
        likelihood = tfd.MultivariateNormalDiag(loc = y_pred_tf, scale_diag = sigma * tf.ones_like(y_pred_tf)).log_prob(var_grid_out)
        return tf.reduce_sum(likelihood) + tf.reduce_sum(prior_theta) + tf.reduce_sum(prior_sigma)

    # Define the Metropolis-Hastings kernel
    step_size = 0.1  
    mh_kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size)
    )
    
    # Initialize all parameters to infer
    initial_theta1 = tf.math.log(tf.ones([], dtype=tf.float32) * 1E3)
    initial_theta2 = tf.zeros([], dtype=tf.float32)  
    initial_theta = tf.stack([initial_theta1, initial_theta2])
    initial_log_sigma = tf.math.log(tf.ones([], dtype=tf.float32) * 0.05)

    # Start timing the MCMC sampling
    start_time = time.time()

    # Run MCMC sampling
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[initial_theta, initial_log_sigma],
        kernel=mh_kernel,
        trace_fn=lambda current_state, kernel_results: kernel_results.is_accepted
    )

    # End timing the MCMC sampling
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"MCMC sampling took {elapsed_time:.2f} seconds")
    np.savetxt(elapsed_time, "mcmc_time")
    exit()

    # Unpack theta samples and transform back
    theta_samples, log_sigma_samples = samples
    theta1_samples = tf.exp(theta_samples[:, 0])
    theta2_samples = -30E3 + (tf.sigmoid(theta_samples[:, 1]) * 15E3)
    sigma_samples = tf.exp(log_sigma_samples)

    return (tf.stack([theta1_samples, theta2_samples], axis=1), sigma_samples), kernel_results


def save_results(samples):

    theta_samples, sigma_samples = samples

    theta_samples_np = theta_samples.numpy()
    sigma_samples_np = sigma_samples.numpy().reshape(-1, 1)

    data = np.hstack((theta_samples_np, sigma_samples_np))
    posterior_samples = np.savetxt('results/TP15_no_disc_error_estimation_aiche/mcmc', data)

def generate_data(use_true_model = False, num_observations = 50):

    # Set variable grid
    var_grid = [[350.0, 400.0, 450.0],
                [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Create model
    model = PhysChem_error(var_grid)
    
    # Generate data
    model.genDataFile(use_true_model = use_true_model, num_observations = num_observations)


def plot_trace(samples, param_names):
    """
    Plots trace plots for the MCMC samples.
    
    Parameters:
    - samples: Tuple containing theta and sigma samples.
    - param_names: List of parameter names (e.g., ['theta_1', 'theta_2', 'sigma']).
    """
    theta_samples = samples[:,0:2]
    theta_samples[:,1] = theta_samples[:,1]/1000
    sigma_samples = samples[:,2]
    
    num_params = theta_samples.shape[1]  # Number of parameters in theta (e.g., 2)
    
    fig, axs = plt.subplots(num_params + 1, 1, sharex = True, figsize = (10,8))

    # Plot trace for each theta parameter
    for i in range(num_params):
        axs[i].plot(theta_samples[:, i], label=f'{param_names[i]}')
        axs[i].set_ylabel(f'{param_names[i]}')

    # Plot trace for sigma
    axs[num_params].plot(sigma_samples)
    axs[num_params].set_ylabel('Noise s.d. Ratio, \n $z_3$ []')

    axs[num_params].set_xlabel('Iteration')

    plt.tight_layout()
    plt.savefig('results/TP15_no_disc_error_estimation_aiche/trace')


# Main code
if __name__ == "__main__":

    # generate_data(use_true_model = False, num_observations = 1)
    
    # samples, kernel_results = run_test(10000, 500)
    
    # save_results(samples)

    samples = np.loadtxt('results/TP15_no_disc_error_estimation_aiche/mcmc')

    # Call this after running the MCMC sampling to plot the trace
    plot_trace(samples, param_names = ['Pre-exp. Factor, \n $z_1$ [Pa]', 'Ads. Energy, \n $z_2$ [kJ$\cdot$mol$^{-1}$]'])
    
    # Call the function to process the results
    # process_results(samples)






# def plot_trace_aiche(samples, param_names):

#     '''TODO: ignore this, delete l8r'''

#     # Read in data
#     theta_samples = samples[:,0:2]

#     ## Normalize all to be in [0,1]
#     theta_samples[:,0] = 0.8*(np.max(theta_samples[:,0]) - theta_samples[:,0])/(np.max(theta_samples[:,0]) - np.min(theta_samples[:,0])) + 0.1
#     theta_samples[:,1] = 0.8*(np.max(theta_samples[:,1]) - theta_samples[:,1])/(np.max(theta_samples[:,1]) - np.min(theta_samples[:,1])) + 0.1

#     if False:  
#         plt.figure(figsize = (8,3))

#         # Plot trace for each theta parameter
#         plt.plot(theta_samples[:, 0], 'r--', label = "$z_{1}$")
#         plt.plot(theta_samples[:, 1], 'b', label = "$z_{2}$")

#         plt.xlabel("Iterations")
#         plt.ylabel("Parameter Value")
#         plt.xlim(0,10000)
#         plt.ylim(0,1)
#         plt.tight_layout()
#         plt.legend(loc = "upper right", ncol = 2)
#         plt.savefig('results/TP15_no_disc_error_estimation_aiche/trace_aiche')
#         plt.close()

#     mcmc_data = np.vstack((theta_samples[:,0], theta_samples[:,1]))
#     x = np.linspace(0.2,0.8,100)
#     y = x
#     X, Y = np.meshgrid(x, y)
#     positions = np.vstack([X.ravel(), Y.ravel()])
    
#     kde_mcmc = gaussian_kde(mcmc_data)
#     mcmc_est = np.reshape(kde_mcmc(positions), X.shape)

#     plt.figure(figsize=(4,3))
#     plt.hexbin(theta_samples[:,0], theta_samples[:,1], gridsize=30, bins = "log", cmap = "Purples_r")
#     plt.colorbar(label = "Frequency")
#     plt.contour(X,Y,mcmc_est, colors = "green", linewidths = 2, levels = 3)
#     plt.xlabel("$z_1$")
#     plt.ylabel("$z_2$")
    
#     plt.savefig('results/TP15_no_disc_error_estimation_aiche/trace_2D_aiche')
#     plt.close()

# def process_results(samples):
    
#     theta_samples = samples[:,0:2]
#     sigma_samples = samples[:,2]

#     # Plot histograms of theta and sigma
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))

#     # Plot theta_1
#     axs[0].hist(theta_samples[:, 0], density=True, alpha=0.75)
#     axs[0].set_title("Posterior distribution of theta_1")
#     axs[0].set_xlabel("theta_1")
#     axs[0].set_ylabel("Density")

#     # Plot theta_2
#     axs[1].hist(theta_samples[:, 1], density=True, alpha=0.75)
#     axs[1].set_title("Posterior distribution of theta_2")
#     axs[1].set_xlabel("theta_2")
#     axs[1].set_ylabel("Density")

#     # Plot sigma
#     axs[2].hist(sigma_samples, density=True, alpha=0.75)
#     axs[2].set_title("Posterior distribution of sigma")
#     axs[2].set_xlabel("sigma")
#     axs[2].set_ylabel("Density")

#     plt.tight_layout()
#     plt.savefig('results/TP15_no_disc_error_estimation_aiche/marginals')