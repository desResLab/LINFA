import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import torch

import matplotlib.pyplot as plt

# Abbreviations
tfd = tfp.distributions
# Import the RCR model
from linfa.models.discrepancy_models import PhysChem_error

def run_test(num_results, num_burnin_steps):
    
    # Set variable grid
    variable_inputs = [[350.0, 400.0, 450.0],
                       [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Assign as experiment model
    model = PhysChem_error(variable_inputs)

    # Read data
    model.data = np.loadtxt('observations.csv', delimiter=',', skiprows=1)

    data_mean = np.mean(model.data[:,2:])
        
    # Form tensors for variables and results in observations
    var_grid_in = tf.convert_to_tensor(model.data[:,:2], dtype=tf.float32)
    var_grid_out = tf.convert_to_tensor(model.data[:,2:], dtype=tf.float32)

    def target_log_prob_fn(theta, log_sigma):
    
        # Transform log_sigma to sigma (ensuring sigma is positive)
        sigma = tf.exp(log_sigma) * data_mean
        # Transformations on theta
        theta1 = tf.exp(theta[0])  # Keep theta_1 on the log scale
        
        # Use sigmoid transformation for theta2 to map between (-15E3, -30E3)
        theta2 = -30E3 + (tf.sigmoid(theta[1]) * 15E3)  # Maps theta_2 between -22E3 and -21E3

        # Priors on transformed parameters
        prior_theta1 = tfd.Normal(loc = 1000.0, scale = 100.0).log_prob(theta1)
        prior_theta2 = tfd.Normal(loc = -21.0E3, scale = 500.0).log_prob(theta2)
        prior_theta = prior_theta1 + prior_theta2

        # Prior on sigma^2 (Beta prior as used)
        prior_sigma = tfd.Beta(1.0, 19.0).log_prob(sigma)

        # Convert theta and sigma from TensorFlow to NumPy to PyTorch tensors
        theta_np = np.array([theta1.numpy(), theta2.numpy()])
        sigma_np = sigma.numpy()

        # Stack theta and sigma for solve_t input
        cal_inputs = torch.tensor(np.hstack([theta_np, sigma_np]), dtype=torch.float32)

        # Call the PyTorch solve_t function
        y_pred_torch = model.solve_t(cal_inputs)

        # Convert PyTorch output to NumPy, then TensorFlow for further processing
        y_pred_np = y_pred_torch.detach().numpy()
        y_pred_tf = tf.convert_to_tensor(y_pred_np, dtype=tf.float32)

        # Likelihood: y_i ~ N(g(x_i, theta), sigma^2)
        likelihood = tfd.MultivariateNormalDiag(loc=y_pred_tf, scale_diag = sigma * tf.ones_like(y_pred_tf)).log_prob(var_grid_out)

        return tf.reduce_sum(likelihood) + tf.reduce_sum(prior_theta) + tf.reduce_sum(prior_sigma)

    # Define the Metropolis-Hastings kernel
    step_size = 0.1  # Adjust step size for better exploration

    mh_kernel = tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=step_size)
    )
    
    initial_theta1 = tf.math.log(tf.ones([], dtype=tf.float32) * 1200)
    initial_theta2 = tf.zeros([], dtype=tf.float32)  # Initialize at 0 to center sigmoid at midpoint of range
    initial_theta = tf.stack([initial_theta1, initial_theta2])
    initial_log_sigma = tf.math.log(tf.ones([], dtype=tf.float32) * 0.05)  # Start with sigma = 0.05

    # Run MCMC sampling
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=[initial_theta, initial_log_sigma],
        kernel=mh_kernel,
        trace_fn=lambda current_state, kernel_results: kernel_results.is_accepted
    )

    # Unpack theta samples and transform back
    theta_samples, log_sigma_samples = samples

    # Transform theta1 back to original scale (it was on log scale during sampling)
    theta1_samples = tf.exp(theta_samples[:, 0])

    # Apply the same sigmoid transformation for theta2 back to the original scale
    theta2_samples = -30E3 + (tf.sigmoid(theta_samples[:, 1]) * 15E3)

    # Transform log_sigma samples back to sigma
    sigma_samples = tf.exp(log_sigma_samples)

    return (tf.stack([theta1_samples, theta2_samples], axis=1), sigma_samples), kernel_results

def save_results(samples):

    theta_samples, sigma_samples = samples

    theta_samples_np = theta_samples.numpy()
    sigma_samples_np = sigma_samples.numpy().reshape(-1, 1)

    data = np.hstack((theta_samples_np, sigma_samples_np))
    posterior_samples = np.savetxt('results/TP15_no_disc_error_estimation/mcmc', data)

def plot_trace(samples, param_names):
    """
    Plots trace plots for the MCMC samples.
    
    Parameters:
    - samples: Tuple containing theta and sigma samples.
    - param_names: List of parameter names (e.g., ['theta_1', 'theta_2', 'sigma']).
    """
    theta_samples = samples[:,0:2]
    sigma_samples = samples[:,2]
    
    num_params = theta_samples.shape[1]  # Number of parameters in theta (e.g., 2)
    
    fig, axs = plt.subplots(num_params + 1, 1, sharex=True)

    # Plot trace for each theta parameter
    for i in range(num_params):
        axs[i].plot(theta_samples[:, i], label=f'{param_names[i]}')
        axs[i].set_ylabel(f'{param_names[i]}')
        axs[i].legend(loc='upper right')

    # Plot trace for sigma
    axs[num_params].plot(sigma_samples, label='sigma', color='tab:orange')
    axs[num_params].set_ylabel('sigma')
    axs[num_params].legend(loc='upper right')

    axs[num_params].set_xlabel('Iteration')

    plt.tight_layout()
    plt.savefig('results/TP15_no_disc_error_estimation/trace')

def process_results(samples):
    
    theta_samples = samples[:,0:2]
    sigma_samples = samples[:,2]

    # Plot histograms of theta and sigma
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot theta_1
    axs[0].hist(theta_samples[:, 0], density=True, alpha=0.75)
    axs[0].set_title("Posterior distribution of theta_1")
    axs[0].set_xlabel("theta_1")
    axs[0].set_ylabel("Density")

    # Plot theta_2
    axs[1].hist(theta_samples[:, 1], density=True, alpha=0.75)
    axs[1].set_title("Posterior distribution of theta_2")
    axs[1].set_xlabel("theta_2")
    axs[1].set_ylabel("Density")

    # Plot sigma
    axs[2].hist(sigma_samples, density=True, alpha=0.75)
    axs[2].set_title("Posterior distribution of sigma")
    axs[2].set_xlabel("sigma")
    axs[2].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig('results/TP15_no_disc_error_estimation/marginals')

def generate_data(use_true_model = False, num_observations=50):

    # Set variable grid
    var_grid = [[350.0, 400.0, 450.0],
                [1.0, 2.0, 3.0, 4.0, 5.0]]

    # Create model
    model = PhysChem_error(var_grid)
    
    # Generate data
    model.genDataFile(use_true_model = use_true_model, num_observations = num_observations)

# Main code
if __name__ == "__main__":

    # generate_data(use_true_model = False, num_observations = 1)
    
    samples, kernel_results = run_test(10000, 1000)
    
    save_results(samples)

    samples = np.loadtxt('results/TP15_no_disc_error_estimation/mcmc')

    # Call this after running the MCMC sampling to plot the trace
    plot_trace(samples, param_names = ['theta_1', 'theta_2'])
    
    # Call the function to process the results
    process_results(samples)


