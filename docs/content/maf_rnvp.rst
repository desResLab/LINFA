MAF and RealNVP
===============

LINFA implements two widely used normalizing flow formulations, MAF [papamakarios2018masked]_ and RealNVP [dinh2016density]_.

MAF belongs to the class of **autoregressive** normalizing flows. Given the latent variable :math:`\boldsymbol{z} = (z_{1},z_{2},\dots,z_{d})` it assumes :math:`p(z_i|z_{1},\dots,z_{i-1}) = \phi[(z_i - \mu_i) / e^{\alpha_i}]`, where :math:`\phi` is the standard normal density, :math:`\mu_i = f_{\mu_i}(z_{1},\dots,z_{i-1})`, :math:`\alpha_i = f_{\alpha_i}(z_{1},\dots,z_{i-1}),\,i=1,2,\dots,d`, and :math:`f_{\mu_i}` and :math:`f_{\alpha_i}` are masked autoencoder neural networks (MADE, [germain2015made]_). 

In a MADE autoencoder the network connectivities are multiplied by Boolean masks so the input-output relation maintains a lower triangular structure, making the computation of the Jacobian determinant particularly simple. MAF transformations are then composed of multiple MADE layers, possibly interleaved by batch normalization layers [ioffe2015batch], typically used to add stability during training and increase network accuracy [papamakarios2018masked].

RealNVP is another widely used flow where, at each layer the first :math:`d'` variables are left unaltered while the remaining :math:`d-d'` are subject to an affine transformation of the form :math:`\widehat{\bm{z}}_{d'+1:d} = \bm{z}_{d'+1:d}\,\odot\,e^{\bm{\alpha}} + \bm{\mu}`, where :math:`\bm{\mu} = f_{\mu}(\bm{z}_{1:d'})` and :math:`\bm{\alpha} = f_{\alpha}(\bm{z}_{d'+1:d})` are MADE autoencoders. In this context, MAF could be seen as a generalization of RealNVP by setting :math:`\mu_i=\alpha_i=0` for :math:`i\leq d'` [papamakarios2018masked]_.
