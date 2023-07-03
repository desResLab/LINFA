[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
 ![example workflow](https://github.com/desResLab/LINFA/actions/workflows/test_publish_pypi.yml/badge.svg) 
[![Documentation Status](https://readthedocs.org/projects/linfa-vi/badge/?version=latest)](https://linfa-vi.readthedocs.io/en/latest/?badge=latest)

## LINFA

LINFA is a library for variational inference with normalizing flow and adaptive annealing. It is designed to accommodate computationally expensive models and difficult-to-sample posterior distributions with dependent parameters.

The code for the masked autoencoders for density estimation (MADE), masked autoregressive flow (MAF) and real non volume-preserving transformation (RealNVP) is based on the implementation provided by [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows). 

### Installation

To install LINFA type

```
pip install linfa-vi
```

### Documentation 

The documentation can be found on [readthedocs](https://linfa-vi.readthedocs.io/en/latest/)

### References

Background theory and examples for LINFA are discussed in the two papers: 

- Y. Wang, F. Liu and D.E. Schiavazzi, *[Variational Inference with NoFAS: Normalizing Flow with Adaptive Surrogate for Computationally Expensive Models](https://www.sciencedirect.com/science/article/abs/pii/S0021999122005162)*
- E.R. Cobian, J.D. Hauenstein, F. Liu and D.E. Schiavazzi, *[AdaAnn: Adaptive Annealing Scheduler for Probability Density Approximation](https://www.dl.begellhouse.com/journals/52034eb04b657aea,796f39cb1acf1296,6f85fe1149ff41d9.html?sgstd=1)*

### Requirements

* PyTorch 1.13.1
* Numpy 1.22
* Matplotlib 3.6 (only plot functionalities `linfa.plot_res`)

### Numerical Benchmarks

LINFA includes five numerical benchmarks:

* Trivial example.
* High dimensional example (Sobol' function).
* Two-element Windkessel model (a.k.a. RC model).
* Three-element Windkessel model (a.k.a. RCR model).
* Friedman 1 dataset example.

The implementation of the lumped parameter network models (RC and RCR models) follows closely from the code developed by [the Schiavazzi Lab at the University of Notre Dame](https://github.com/desResLab/supplMatHarrod20).

To run the tests type
```sh
python -m unittest linfa.tests.test_linfa.linfa_test_suite.NAME_example
```
where `NAME` need to be replaced by
* `trivial` for the trivial example (Ex 1).
* `highdim` for the high-dimensional example (Ex 2).
* `rc` for the RC model (Ex 3).
* `rcr` for the RCR model (Ex 4).
* `adaann` for the Friedman model example (Ex 5).

At regular intervals set by the parameter `experiment.save_interval` LINFA writes a few results files. The sub-string `NAME` refers to the experiment name specified in the `experiment.name` variable, and `IT` indicates the iteration at which the file is written. The results files are

* `log.txt` contains the log profile information, i.e.
  * Iteration number.
  * Annealing temperature at each iteration.
  * Loss function at each iteration.
* `NAME_grid_IT` contains the inputs where the true model was evaluated. 
* `NAME_params_IT` contains the batch of input parameters $\boldsymbol{z}_{K}$ in the physical space generated at iteration `IT`. 
* `NAME_samples_IT` contains the batch of normalized parameters (parameter values before the coordinate transformation) generated at iteration `IT`.
* `NAME_logdensity_IT` contains the value of the log posterior density corresponding to each parameter realization. 
* `NAME_outputs_IT` contains the true model (or surrogate model) outputs for each batch sample at iteration `IT`.
* `NAME_IT.nf` contains a backup of the normalizing flow parameters at iteration `IT`.

A post processing script is also available to plot all results. To run it type

```sh
python linfa.plot_res -n NAME -i IT
```
where `NAME` and `IT` are again the experiment name and iteration number corresponding to the result file of interest. 

### Usage

To use LINFA with your model you need to specify the following components:

* A computational model.
* A surrogate model.
* A log-likelihood model.
* An optional transformation. 

In addition you need to specify a list of options as discussed in the [documentation](https://linfa-vi.readthedocs.io/en/latest/content/linfa_options.html).

### Tutorial

A (tutorial)[] is also available which will guide you through the definition of each of the quantities and function required by LINFA. 

### Citation

Did you use LINFA? Please cite our paper using:
```
@misc{TO BE FINALIZED!!!,
      title={LINFA: a Python library for variational inference with normalizing flow and annealing}, 
      author={Yu Wang, Emma R. Cobian, Fang Liu, Jonathan D. Hauenstein, Daniele E. Schiavazzi},
      year={2022},
      eprint={2201.03715},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}