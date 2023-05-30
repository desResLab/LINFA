[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
 ![example workflow](https://github.com/desResLab/LINFA/actions/workflows/test_publish_pypi.yml/badge.svg) 
[![Documentation Status](https://readthedocs.org/projects/mrina/badge/?version=latest)](https://mrina.readthedocs.io/en/latest/?badge=latest)

## LINFA

LINFA is a library for variational inference with normalizing flow and adaptive annealing. It combines the 

In particular, masked autoregressive flow (MAF) and RealNVP are used in the code, which are 
implemented by [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows). 


### Documentation 

The documentation can be found on [readmydocs]()

### References

Background theory and examples for LINFA are discussed in the two papers: 

- Y. Wang, F. Liu and D.E. Schiavazzi, *[Variational Inference with NoFAS: Normalizing Flow with Adaptive Surrogate for Computationally Expensive Models](https://www.sciencedirect.com/science/article/abs/pii/S0021999122005162)*
- E.R. Cobian, J.D. Hauenstein, F. Liu and D.E. Schiavazzi, *[AdaAnn: Adaptive Annealing Scheduler for Probability Density Approximation](https://www.dl.begellhouse.com/journals/52034eb04b657aea,796f39cb1acf1296,6f85fe1149ff41d9.html?sgstd=1)*

### Requirements

* PyTorch 1.4.0
* Numpy 1.19.2
* Scipy 1.6.1

### Available numerical benchmarks

LINFA includes five numerical benchmarks:

* Trivial example.
* High dimensional example (Sobol' function).
* Two-element Windkessel model (a.k.a RC model).
* Three-element Windkessel model (a.k.a RCR model).
* Friedman 1 dataset example.

The implementation of the lumped parameter network models (RC and RCR models) follows the code from [the Schiavazzi Lab at the University of Notre Dame](https://github.com/desResLab/supplMatHarrod20).


To run the trivial example (Ex 1) type:

```sh
# python -m unittest tests/test_linfa.py linfa_test_suite.trivial_example
```

To run the high-dimensional example (Ex 2) type:

```sh
# python -m unittest tests/test_linfa.py linfa_test_suite.highdim_example
```

To run the RC model (Ex 3) type:

```sh
# python -m unittest tests/test_linfa.py linfa_test_suite.rc_example
```

To run the RCR model (Ex 4) type:

```sh
# python -m unittest tests/test_linfa.py linfa_test_suite.rcr_example
```

To run the Friedman model example (Ex 5) type:

```sh
# python -m unittest tests/test_linfa.py linfa_test_suite.adaann_example
```

### Usage

To use LINFA with your model you need to specify the following components:

* A computational model.
* A surrogate model.
* A log-likelihood model.
* An optional transformation. 

In addition you need to specify a list of options as discussed in the (documentation)[].

### Citation

Did you use LINFA? Cite us using:

COMPLETE WHEN ON ARCHIVE!!!

```
@misc{linfa,
      title={LINFA: a Python library for variational inference with normalizing flow and annealing}, 
      author={Yu Wang, Emma R. Cobian, Fang Liu, Jonathan D. Hauenstein, Daniele E. Schiavazzi},
      year={2022},
      eprint={2201.03715},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}