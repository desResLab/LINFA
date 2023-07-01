Normalizing flow with adaptive surrogate (NoFAS)
================================================

LINFA is designed to accommodate black-box models :math:`\bm{f}: \bm{\mathcal{Z}} \to \bm{\mathcal{X}}` between the random inputs :math:`\bm{z} = (z_1, z_2, \cdots, z_d)^T \in \bm{\mathcal{Z}}` and the outputs :math:`(x_1, x_2,\cdots,x_m)^T \in \bm{\mathcal{X}}`, and assumes :math:`n` observations :math:`\bm x = \{\bm x_i\}_{i=1}^n \subset \bm{\mathcal{X}}` to be available. 

Our goal is to infer :math:`\bm z` and to quantify its uncertainty given :math:`\bm{x}`. We employ a variational Bayesian paradigm and sample from the posterior distribution :math:`p(\bm z\vert \bm x)\propto \ell_{\bm z}(\bm x,\bm{f})\,p(\bm z)`, with prior :math:`p(\bm z)` via normalizing flows. 

This requires the evaluation of the gradient of the ELBO :eq:`equ:ELBO` with respect to the NF parameters :math:`\bm{\lambda}`, replacing :math:`p(\bm x, \bm z_K)` with :math:`p(\bm x\vert\bm z_K)\,p(\bm z)=\ell_{\bm z_K}(\bm{x},\bm{f})\,p(\bm z)`, and approximating the expectations with their MC estimates. 

However, the likelihood function needs to be evaluated at every MC realization, which can be costly if the model :math:`\bm{f}(\bm{z})` is computationally expensive. In addition, automatic differentiation through a legacy (e.g. physics-based) solver may be an impractical, time-consuming, or require the development of an adjoint solver.

Our solution is to replace the model :math:`\bm{f}` with a computationally inexpensive surrogate :math:`\widehat{\bm{f}}: \bm{\mathcal{Z}} \times \bm{\mathcal{W}} \to \bm{\mathcal{X}}` parameterized by the weigths :math:`\bm{w} \in \bm{\mathcal{W}}`, whose derivatives can be obtained at a relatively low computational cost, but intrinsic bias in the selected surrogate formulation, a limited number of training examples, and locally optimal :math:`\bm{w}` can compromise the accuracy of :math:`\widehat{\bm{f}}`.

To resolve these issues, LINFA implements NoFAS, which updates the surrogate model adaptively by smartly weighting the samples of :math:`\bm{z}` from NF thanks to a **memory-aware** loss function. Once a newly updated surrogate is obtained, the likelihood function is updated, leading to a new posterior distribution that will be approximated by VI-NF, producing, in turn, new samples for the next surrogate model update, and so on. 
Additional details can be found in [wang2022variational]_.
