import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import gpflow
from gpflow import set_trainable
from gpflow.ci_utils import reduce_in_tests

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

tf.random.set_seed(123)

# Generate synthetic data
rng = np.random.RandomState(42)
N = 15
def synthetic_data(num: int, rng: np.random.RandomState):
    X = rng.rand(num, 1)
    Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + rng.randn(num, 1) * 0.1 + 3
    return X, Y
data = (X, Y) = synthetic_data(N, rng)

# Visualize synthetic data
plt.figure(figsize=(12, 6))
plt.plot(X, Y, "kx", mew=2)
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.title("toy data")
plt.show()

# Define GP model
kernel = gpflow.kernels.Matern52(lengthscales = 0.3)
mean_function = gpflow.mean_functions.Linear(1.0, 0.0)
model = gpflow.models.GPR(data, kernel, mean_function, noise_variance=0.01)

# Do optimization
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(model.training_loss, model.trainable_variables)
print(f"log posterior density at optimum: {model.log_posterior_density()}")

# Define priors
# tfp.distributions dtype is inferred from parameters - so convert to 64-bit
model.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
model.mean_function.A.prior = tfd.Normal(f64(0.0), f64(10.0))
model.mean_function.b.prior = tfd.Normal(f64(0.0), f64(10.0))

gpflow.utilities.print_summary(model)

num_burnin_steps = reduce_in_tests(300)
num_samples = reduce_in_tests(1000)

# Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
hmc_helper = gpflow.optimizers.SamplingHelper(
    model.log_posterior_density, model.trainable_parameters
)

hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=hmc_helper.target_log_prob_fn,
    num_leapfrog_steps=10,
    step_size=0.01,
)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc,
    num_adaptation_steps=10,
    target_accept_prob=f64(0.75),
    adaptation_rate=0.1,
)


@tf.function
def run_chain_fn():
    return tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=num_burnin_steps,
        current_state=hmc_helper.current_state,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
    )


samples, traces = run_chain_fn()
parameter_samples = hmc_helper.convert_to_constrained_values(samples)

param_to_name = {
    param: name
    for name, param in gpflow.utilities.parameter_dict(model).items()
}

def plot_samples(samples, parameters, y_axis_label):
    plt.figure(figsize=(8, 4))
    for val, param in zip(samples, parameters):
        plt.plot(tf.squeeze(val), label=param_to_name[param])
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlabel("HMC iteration")
    plt.ylabel(y_axis_label)


plot_samples(samples, model.trainable_parameters, "unconstrained values")
plot_samples(
    parameter_samples,
    model.trainable_parameters,
    "constrained parameter values",
)

def marginal_samples(samples, parameters, y_axis_label):
    fig, axes = plt.subplots(
        1, len(param_to_name), figsize=(15, 3), constrained_layout=True
    )
    for ax, val, param in zip(axes, samples, parameters):
        ax.hist(np.stack(val).flatten(), bins=20)
        ax.set_title(param_to_name[param])
    fig.suptitle(y_axis_label)
    plt.show()


marginal_samples(
    samples, model.trainable_parameters, "unconstrained variable samples"
)
marginal_samples(
    parameter_samples,
    model.trainable_parameters,
    "constrained parameter samples",
)

def plot_joint_marginals(samples, parameters, y_axis_label):
    name_to_index = {
        param_to_name[param]: i for i, param in enumerate(parameters)
    }
    f, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axs[0].plot(
        samples[name_to_index[".likelihood.variance"]],
        samples[name_to_index[".kernel.variance"]],
        "k.",
        alpha=0.15,
    )
    axs[0].set_xlabel("noise_variance")
    axs[0].set_ylabel("signal_variance")

    axs[1].plot(
        samples[name_to_index[".likelihood.variance"]],
        samples[name_to_index[".kernel.lengthscales"]],
        "k.",
        alpha=0.15,
    )
    axs[1].set_xlabel("noise_variance")
    axs[1].set_ylabel("lengthscale")

    axs[2].plot(
        samples[name_to_index[".kernel.lengthscales"]],
        samples[name_to_index[".kernel.variance"]],
        "k.",
        alpha=0.1,
    )
    axs[2].set_xlabel("lengthscale")
    axs[2].set_ylabel("signal_variance")
    f.suptitle(y_axis_label)
    plt.show()


plot_joint_marginals(
    samples, model.trainable_parameters, "unconstrained variable samples"
)
plot_joint_marginals(
    parameter_samples, model.trainable_parameters, "parameter samples"
)

# plot the function posterior
xx = np.linspace(-0.1, 1.1, 100)[:, None]
plt.figure(figsize=(12, 6))

for i in range(0, num_samples, 20):
    for var, var_samples in zip(hmc_helper.current_state, samples):
        var.assign(var_samples[i])
    f = model.predict_f_samples(xx, 1)
    plt.plot(xx, f[0, :, :], "C0", lw=2, alpha=0.3)

plt.plot(X, Y, "kx", mew=2)
plt.xlim(xx.min(), xx.max())
plt.ylim(0, 6)
plt.xlabel("$x$")
plt.ylabel("$f|X,Y$")
plt.title("Posterior GP samples")

plt.show()

