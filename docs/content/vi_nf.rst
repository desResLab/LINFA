Variational Inference with Normalizing Flow
===========================================

Consider the problem of estimating (in a Bayesian sense) the parameters :math:`\bm{z}\in\bm{\mathcal{Z}}` of a physics-based or statistical model

.. math::
   \bm{x} = \bm{f}(\bm{z}) + \bm{\varepsilon},

from the observations :math:`\bm{x}\in\bm{\mathcal{X}}` and a known statistical characterization of the error :math:`\bm{\varepsilon}`.

We tackle this problem with variational inference and normalizing flow. A **normalizing flow** (NF) is a nonlinear transformation :math:`F:\mathbb{R}^{d}\times \bm{\Lambda} \to \mathbb{R}^{d}` designed to map an easy-to-sample **base** distribution :math:`q_{0}(\bm{z}_{0})` into a close approximation :math:`q_{K}(\bm{z}_{K})` of a desired target posterior density :math:`p(\bm{z}|\bm{x})`. This transformation can be determined by composing :math:`K` bijections 

.. math::
   \bm{z}_{K} = F(\bm{z}_{0}) = F_{K} \circ F_{K-1} \circ \cdots \circ F_{k} \circ \cdots \circ F_{1}(\bm{z}_{0}),

and evaluating the transformed density through the change of variable formula (see [villani2009optimal]_).

In the context of variational inference, we seek to determine an **optimal** set of parameters :math:`\bm{\lambda}\in\bm{\Lambda}` so that :math:`q_{K}(\bm{z}_{K})\approx p(\bm{z}|\bm{x})`. Given observations :math:`\bm{x}\in\mathcal{\bm{X}}`, a likelihood function :math:`l_{\bm{z}}(\bm{x})` (informed by the distribution of the error :math:`\bm{\varepsilon}`) and prior :math:`p(\bm{z})`, a NF-based approximation :math:`q_K(\bm{z})` of the posterior distribution :math:`p(\bm{z}|\bm{x})` can be computed by maximizing the lower bound to the log marginal likelihood :math:`\log p(\bm{x})` (the so-called **evidence lower bound** or **ELBO**), or, equivalently, by minimizing a **free energy bound** (see, e.g., [rezende2015variational]_).

.. math::
   \begin{split}
   \mathcal{F}(\bm x)& = \mathbb{E}_{q_K(\bm z_K)}\left[\log q_K(\bm z_K) - \log p(\bm x, \bm z_K)\right]\\
   & = \mathbb{E}_{q_0(\bm z_0)}[\log q_0(\bm z_0)] - \mathbb{E}_{q_0(\bm z_0)}[\log p(\bm x, \bm z_K)] - \mathbb{E}_{q_0(\bm z_0)}\left[\sum_{k=1}^K \log \left|\det \frac{\partial \bm z_k}{\partial \bm z_{k-1}}\right|\right].
   \end{split}
   :label: equ:ELBO

For computational convenience, normalizing flows transformations are selected to be easily invertible and their Jacobian determinant can be computed with a cost that grows linearly with the problem dimensionality. Approaches in the literature include RealNVP [dinh2016density]_, GLOW [kingma2018glow]_ and autoregressive transformations such as MAF [papamakarios2018masked] and IAF [kingma2016improved].
