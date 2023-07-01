Adaptive Annealing
==================

Annealing is a technique to parametrically smooth a target density to improve sampling efficiency and accuracy during inference. In the discrete case, this is achieved by incrementing an *inverse temperature* :math:`t_{k}` and setting :math:`p_k(\boldsymbol{z},\boldsymbol{x}) = p^{t_k}(\boldsymbol{z},\boldsymbol{x}),\,\,\text{for } k=0,\dots,K`, where :math:`0 < t_{0} < \cdots < t_{K} \le 1`. The result of exponentiation produces a smooth unimodal distribution for a sufficiently small :math:`t_0`, recovering the target density as :math:`t_{k}` approaches 1. In other words, annealing provides a continuous deformation from an easier to approximate unimodal distribution to a desired target density.

A linear annealing scheduler (see, e.g. :cite:t:`rezende2015variational`) with fixed temperature increments is often used in practice, where :math:`t_j=t_{0} + j (1-t_{0})/K` for :math:`j=0,\ldots,K` with constant increments :math: `\epsilon = (1-t_{0})/K`. Intuitively, small temperature changes are desirable to carefully explore the parameter spaces at the beginning of the annealing process, whereas larger changes can be taken as :math:`t_{k}` increases, after annealing has helped to capture important features of the target distribution (e.g., locating all the relevant modes).

The proposed AdaAnn scheduler determines the increment :math:`\epsilon_{k}` that approximately produces a pre-defined change in the KL divergence between two distributions annealed at :math:`t_{k}` and :math:`t_{k+1}=t_{k}+\epsilon_{k}`, respectively. Letting the KL divergence equal a constant :math:`\tau^2/2`, where :math:`\tau` is referred to as the **KL tolerance**, the step size :math:`\epsilon_k` becomes 

.. math::
   \epsilon_k = \tau/ \sqrt{\mathbb{V}_{p^{t_k}}[\log p(\boldsymbol z,\boldsymbol{x})]}. 
   :label: equ:adaann

The denominator is large when the support of the annealed distribution :math:`p^{t_{k}}(\boldsymbol{z},\boldsymbol{x})` is wider than the support of the target :math:`p(\boldsymbol{z},\boldsymbol{x})`, and progressively reduces with increasing :math:`t_{k}`. Further detail on the derivation of the expression for :math:`\epsilon_{k}` can be found in :cite:t:`cobian2023adaann`.
