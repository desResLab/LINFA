Variational Inference with Normalizing Flow
===========================================

Consider the problem of estimating (in a Bayesian sense) the parameters :math:`\boldsymbol{z}\in\boldsymbol{\mathcal{Z}}` of a physics-based or statistical model

.. math::
   \boldsymbol{x} = \boldsymbol{f}(\boldsymbol{z}) + \boldsymbol{\varepsilon},

from the observations :math:`\boldsymbol{x}\in\boldsymbol{\mathcal{X}}` and a known statistical characterization of the error :math:`\boldsymbol{\varepsilon}`.

We tackle this problem with variational inference and normalizing flow. A **normalizing flow** (NF) is a nonlinear transformation :math:`F:\mathbb{R}^{d}\times \boldsymbol{\Lambda} \to \mathbb{R}^{d}` designed to map an easy-to-sample **base** distribution :math:`q_{0}(\boldsymbol{z}_{0})` into a close approximation :math:`q_{K}(\boldsymbol{z}_{K})` of a desired target posterior density :math:`p(\boldsymbol{z}|\boldsymbol{x})`. This transformation can be determined by composing :math:`K` bijections 

.. math::
   \boldsymbol{z}_{K} = F(\boldsymbol{z}_{0}) = F_{K} \circ F_{K-1} \circ \cdots \circ F_{k} \circ \cdots \circ F_{1}(\boldsymbol{z}_{0}),

and evaluating the transformed density through the change of variable formula (see :cite:p:`villani2009optimal`).

In the context of variational inference, we seek to determine an **optimal** set of parameters :math:`\boldsymbol{\lambda}\in\boldsymbol{\Lambda}` so that :math:`q_{K}(\boldsymbol{z}_{K})\approx p(\boldsymbol{z}|\boldsymbol{x})`. Given observations :math:`\boldsymbol{x}\in\mathcal{\boldsymbol{X}}`, a likelihood function :math:`l_{\boldsymbol{z}}(\boldsymbol{x})` (informed by the distribution of the error :math:`\boldsymbol{\varepsilon}`) and prior :math:`p(\boldsymbol{z})`, a NF-based approximation :math:`q_K(\boldsymbol{z})` of the posterior distribution :math:`p(\boldsymbol{z}|\boldsymbol{x})` can be computed by maximizing the lower bound to the log marginal likelihood :math:`\log p(\boldsymbol{x})` (the so-called **evidence lower bound** or **ELBO**), or, equivalently, by minimizing a **free energy bound** (see, e.g., :cite:p:`rezende2015variational`).

.. math::
   \begin{split}
   \mathcal{F}(\boldsymbol x)& = \mathbb{E}_{q_K(\boldsymbol z_K)}\left[\log q_K(\boldsymbol z_K) - \log p(\boldsymbol x, \boldsymbol z_K)\right]\\
   & = \mathbb{E}_{q_0(\boldsymbol z_0)}[\log q_0(\boldsymbol z_0)] - \mathbb{E}_{q_0(\boldsymbol z_0)}[\log p(\boldsymbol x, \boldsymbol z_K)] - \mathbb{E}_{q_0(\boldsymbol z_0)}\left[\sum_{k=1}^K \log \left|\det \frac{\partial \boldsymbol z_k}{\partial \boldsymbol z_{k-1}}\right|\right].
   \end{split}
   :label: equ:ELBO

For computational convenience, normalizing flows transformations are selected to be easily invertible and their Jacobian determinant can be computed with a cost that grows linearly with the problem dimensionality. Approaches in the literature include RealNVP :cite:p:`dinh2016density`, GLOW :cite:p:`kingma2018glow` and autoregressive transformations such as MAF :cite:p:`papamakarios2018masked` and IAF :cite:p:`kingma2016improved`.
