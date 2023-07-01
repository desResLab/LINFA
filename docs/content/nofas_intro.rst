Normalizing flow with adaptive surrogate (NoFAS)
================================================

LINFA is designed to accommodate black-box models :math:`\boldsymbol{f}: \boldsymbol{\mathcal{Z}} \to \boldsymbol{\mathcal{X}}` between the random inputs :math:`\boldsymbol{z} = (z_1, z_2, \cdots, z_d)^T \in \boldsymbol{\mathcal{Z}}` and the outputs :math:`(x_1, x_2,\cdots,x_m)^T \in \boldsymbol{\mathcal{X}}`, and assumes :math:`n` observations :math:`\boldsymbol x = \{\boldsymbol x_i\}_{i=1}^n \subset \boldsymbol{\mathcal{X}}` to be available. 

Our goal is to infer :math:`\boldsymbol z` and to quantify its uncertainty given :math:`\boldsymbol{x}`. We employ a variational Bayesian paradigm and sample from the posterior distribution :math:`p(\boldsymbol z\vert \boldsymbol x)\propto \ell_{\boldsymbol z}(\boldsymbol x,\boldsymbol{f})\,p(\boldsymbol z)`, with prior :math:`p(\boldsymbol z)` via normalizing flows. 

This requires the evaluation of the gradient of the ELBO :eq:`equ:ELBO` with respect to the NF parameters :math:`\boldsymbol{\lambda}`, replacing :math:`p(\boldsymbol x, \boldsymbol z_K)` with :math:`p(\boldsymbol x\vert\boldsymbol z_K)\,p(\boldsymbol z)=\ell_{\boldsymbol z_K}(\boldsymbol{x},\boldsymbol{f})\,p(\boldsymbol z)`, and approximating the expectations with their MC estimates. 

However, the likelihood function needs to be evaluated at every MC realization, which can be costly if the model :math:`\boldsymbol{f}(\boldsymbol{z})` is computationally expensive. In addition, automatic differentiation through a legacy (e.g. physics-based) solver may be an impractical, time-consuming, or require the development of an adjoint solver.

Our solution is to replace the model :math:`\boldsymbol{f}` with a computationally inexpensive surrogate :math:`\widehat{\boldsymbol{f}}: \boldsymbol{\mathcal{Z}} \times \boldsymbol{\mathcal{W}} \to \boldsymbol{\mathcal{X}}` parameterized by the weigths :math:`\boldsymbol{w} \in \boldsymbol{\mathcal{W}}`, whose derivatives can be obtained at a relatively low computational cost, but intrinsic bias in the selected surrogate formulation, a limited number of training examples, and locally optimal :math:`\boldsymbol{w}` can compromise the accuracy of :math:`\widehat{\boldsymbol{f}}`.

To resolve these issues, LINFA implements NoFAS, which updates the surrogate model adaptively by smartly weighting the samples of :math:`\boldsymbol{z}` from NF thanks to a **memory-aware** loss function. Once a newly updated surrogate is obtained, the likelihood function is updated, leading to a new posterior distribution that will be approximated by VI-NF, producing, in turn, new samples for the next surrogate model update, and so on. 
Additional details can be found in :cite:p:`wang2022variational`.
