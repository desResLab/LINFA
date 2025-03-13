import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


# Set global plot settings
plt.rcParams['figure.figsize']      = (8, 6)
plt.rcParams['figure.dpi']          = 300
plt.rcParams['axes.labelsize']      = 16
plt.rcParams['xtick.labelsize']     = 15
plt.rcParams['ytick.labelsize']     = 15
plt.rcParams['legend.fontsize']     = 12
plt.rcParams['lines.linewidth']     = 3
plt.rcParams['lines.markersize']    = 8
plt.rcParams['axes.labelweight']    = 'bold'
plt.rcParams['xtick.direction']     = 'in'
plt.rcParams['ytick.direction']     = 'in'
plt.rcParams['xtick.top']           = True
plt.rcParams['ytick.right']         = True
plt.rcParams['savefig.bbox']        = 'tight'


if False:
    U = np.linspace(-4,4, 200)
    u = stats.norm.pdf(U, loc = 0, scale = 1)
    pi = 0.2
    z = pi * stats.norm.pdf(U, loc = -2, scale = 0.5) + (1 - pi) * stats.norm.pdf(U, loc = 1, scale = 0.5)

    plt.figure()
    plt.plot(U,u, linewidth = 3)
    plt.fill_between(U, u * 0, u, alpha = 0.5)
    plt.savefig("aiche_delete_l8r_blue", dpi = 300)
    plt.close()

    plt.figure()
    plt.plot(U,z, linewidth = 3, color = "r")
    plt.fill_between(U, u * 0, z, alpha = 0.5, color = "r")
    plt.savefig("aiche_delete_l8r_orange", dpi = 300)
    plt.close()

if False:
    X = np.linspace(-5, 5, 200)
    shift = 6
    scale = 1
    x1 = stats.norm.pdf(X, 0 , 1)
    x2 = 0.75 * stats.norm.pdf(X - shift, 0 - shift, 1.5 * scale)
    x3 = 0.9 * stats.norm.pdf(X + shift, 0 + shift, 1.25 * scale)

    plt.figure(figsize=(10, 4))

    
    plt.plot(X - shift, x2, linewidth = 3, color = "r")
    plt.fill_between(X - shift, x2 * 0, x2, alpha = 0.3, color = "r")

    plt.plot(X + shift, x3, linewidth = 3, color = "b")
    plt.fill_between(X + shift, x3 * 0, x3, alpha = 0.3, color = "b")

    plt.plot(X, x1, linewidth = 3, color = "g")
    plt.fill_between(X, x1 * 0, x1, alpha = 0.3, color = "g")

    plt.plot(X, x1*0, linewidth = 3, color = "k")
    plt.plot(X - shift, x1*0, linewidth = 3, color = "k")
    plt.plot(X + shift, x1*0, linewidth = 3, color = "k")
    plt.savefig("kl_divergence", dpi = 300)

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Observed data
    np.random.seed(0)
    x_observed = np.random.normal(loc=2.5, scale=1.0, size=10)  # Data with unknown mean and known variance

    # Known parameters
    sigma = 1.0  # Known standard deviation of the observations
    n = len(x_observed)  # Number of observations
    sample_mean = np.mean(x_observed)

    # Prior parameters
    mu_0 = 0.0  # Prior mean
    tau_0 = 1.0  # Prior standard deviation

    # Posterior parameters
    tau_n_sq = 1 / (1 / tau_0**2 + n / sigma**2)  # Posterior variance
    mu_n = tau_n_sq * (mu_0 / tau_0**2 + n * sample_mean / sigma**2)  # Posterior mean
    tau_n = np.sqrt(tau_n_sq)  # Posterior standard deviation

    # Range of mu values
    mu_values = np.linspace(1.5, 4.5, 200)

    # Calculate unnormalized posterior (prior * likelihood)
    prior = norm.pdf(mu_values, mu_0, tau_0)
    likelihood = norm.pdf(mu_values, sample_mean, sigma / np.sqrt(n))
    unnormalized_posterior = prior * likelihood

    # Scale up unnormalized posterior for visibility
    unnormalized_posterior_scaled = unnormalized_posterior * 200  # Adjust this factor as needed

    # Calculate normalized posterior
    normalized_posterior = norm.pdf(mu_values, mu_n, tau_n)

    # Plotting
    plt.figure(figsize=(5, 5))

    # Unnormalized posterior with shading (scaled up)
    plt.plot(mu_values, unnormalized_posterior_scaled, label="Unnormalized", linestyle="--", color="blue")
    plt.fill_between(mu_values, unnormalized_posterior_scaled, color="blue", alpha=0.2)

    # Normalized posterior with shading
    plt.plot(mu_values, normalized_posterior, label="Normalized", color="red")
    plt.fill_between(mu_values, normalized_posterior, color="red", alpha=0.2)

    # Annotation to highlight normalization importance
    plt.text(4.0, 0.8, "Area = 1", color="red", fontsize=16, ha="center", backgroundcolor="white")
    plt.text(4.0, 0.5, "Area < 1", color="blue", fontsize=16, ha="center", backgroundcolor="white")

    # Labels and legend
    plt.xlabel(r"$\mu$")
    plt.ylabel("Density")
    plt.legend(loc="upper right", fontsize = 14)
    plt.ylim(0,1.4)

    plt.savefig("normalized_density_fig", dpi = 300)


import os
import torch
from linfa.models.discrepancy_models import PhysChem_error
import matplotlib.pyplot as plt



data = np.loadtxt("observations.csv", skiprows=1, delimiter=',')
unique_values = np.unique(data[:, 0])
mrkrs = ['o', 'v', 's']
colores = ['m', 'red', 'orange']
# Plot each unique value as a separate line
plt.figure(figsize=(5, 5))


# Add temperatures and pressures for each evaluation

samples = np.loadtxt('results/TP15_no_disc_error_estimation_aiche/TP15_no_disc_error_estimation_aiche_params_6000')
temps = [350.0, 400.0, 450.0]
pressures =  np.linspace(0.0, 5.5).tolist()
for i, temp in enumerate(temps):
    for j, sample in enumerate(samples):
        variable_inputs = [[temp], pressures]
        langmuir = PhysChem_error(variable_inputs)
        ssl = langmuir.solve_t(torch.tensor(sample))
        plt.plot(pressures, ssl, color = colores[i], linewidth = 0.1, alpha = 0.2)
    ssl_true = langmuir.solve_t(torch.tensor([1000, -21E3, 0.05]))
    if i == 0:
        plt.plot(pressures, ssl_true, 'k--', label = "True")
        plt.plot([], [], 'k-', alpha = 0.2, label = "Estimated")
    else:
        plt.plot(pressures, ssl_true, 'k--')

for i, val in enumerate(unique_values):
    subset = data[data[:, 0] == val]
    plt.plot(subset[:, 1], subset[:, 2], color = colores[i], marker = mrkrs[i], markeredgecolor = 'k', linestyle = "None", label=f'{int(val)} K')

plt.xlabel(r'Pressure, $P$ [Pa]')
plt.ylabel(r'Coverage, [ ]')
plt.xlim(0,5.5)
plt.ylim(0,1.0)
plt.legend()
plt.savefig("results/TP15_no_disc_error_estimation_aiche/fxn_pred")




# import os
# import torch
# from linfa.models.discrepancy_models import PhysChem
# import matplotlib.pyplot as plt

# samples = np.loadtxt('results/TP15_no_disc_error_estimation_aiche/TP15_no_disc_error_estimation_aiche_outputs_lf+noise_6000')


# samples = np.loadtxt('results/TP15_no_disc_error_estimation_aiche/TP15_no_disc_error_estimation_aiche_samples_6000')
# observations = np.loadtxt("observations.csv", skiprows=1, delimiter=',')

# # Set variable grid
# T = [300.0, 400.0, 450.0]
# P = [1.0, 2.0, 3.0, 4.0, 5.0]

# samples = samples.reshape(3,5,5000)

# # Plot the samples
# for i in range(5000):
#     plt.plot(P, samples[0, :, i], 'm-', linewidth=0.005)
#     plt.plot(P, samples[1, :, i], 'r-', linewidth=0.005)
#     plt.plot(P, samples[2, :, i], color="orange", linestyle='-', linewidth=0.005)

# # Plot observations
# plt.plot(observations[:, 1], observations[:, 2], 'ko')

# # Add custom legend entries by plotting representative lines
# plt.plot([], [], 'm-', label="350 K")       # Magenta line
# plt.plot([], [], 'r-', label="400 K")       # Red line
# plt.plot([], [], color="orange", linestyle='-', label="450 K")  # Orange line
# plt.plot([], [], 'ko', label="Observations")  # Black circles for observations

# # Add legend, labels, and limits
# plt.legend()
# plt.xlim(1, 5)
# plt.xlabel("Pressure, $P$ [Pa]")
# plt.ylabel("Coverage")
# plt.savefig("function", dpi=300)

# for i in range(5000):
#     plt.plot(P, samples[0,:,i],'m-', linewidth = 0.005)
#     plt.plot(P, samples[1,:,i],'r-', linewidth = 0.005)
#     plt.plot(P, samples[2,:,i],color = "orange", linestyle = '-', linewidth = 0.005)
# plt.plot(observations[:,1], observations[:,2], 'ko')
# plt.xlim(1,5)
# plt.xlabel("Pressure, $P$ [Pa]")
# plt.ylabel("Coverage")
# plt.savefig("function", dpi = 300)
# exit()