# A Package

Variational Inference with NoFAS: Normalizing Flow with Adaptive Surrogate for Computationally Expensive Models

NoFAS estimates the posterior distribution of hidden variables for a computationally expensive model using normalizing 
flow with an adaptive surrogate. In particular, masked autoregressive flow (MAF) and RealNVP are used in the code, which are 
implemented by [Kamen Bliznashki](https://github.com/kamenbliznashki/normalizing_flows). 
The code includes four numerical experiments: closed form mapping, RC, RCR, and non-isomorphic Sobol function.
RC and RCR are implemented by [the Schiavazzi Lab at the University of Notre Dame](https://github.com/desResLab/supplMatHarrod20).

Need AdaANN
## Paper
The methodology and the numerical examples are discussed in the paper:
Y. Wang, F. Liu and D.E. Schiavazzi, *[Variational Inference with NoFAS: Normalizing Flow with Adaptive Surrogate for Computationally Expensive Models](https://arxiv.org/pdf/2108.12657.pdf)*

Need AdaANN
## Requirements:
* PyTorch 1.4.0
* Numpy 1.19.2
* Scipy 1.6.1

## Usage

## Recommended Hyperparameters of NoFAS
All experiments used RMSprop as the optimizer equipped exponential decay scheduler with decay factor 0.9999. All normalizing flows use ReLU activations and 
maximum number of iterations is 25001. All MADE autoencoders contain 1 hidden layer with 100 nodes.

| Experiment  | NF type | NF layers | batch size | budget | updating size | updating period | learning rate |
| ----------- | ------- |-----------| ---------- | ------ | ------------- | --------------- | ------------- |
| closed form | RealNVP | 5         | 200        | 64     | 2             | 1000            | 0.002         |
| RC          | MAF     | 5         | 250        | 64     | 2             | 1000            | 0.003         |
| RCR         | MAF     | 15        | 500        | 216    | 2             | 300             | 0.003         |
| 5-dim       | RealNVP | 15        | 250        | 1023   | 12            | 250             | 0.0005        |
