Three-element Wndkessel Model
=============================

The three-parameter Windkessel or **RCR** model is characterized by proximal and distal resistance parameters :math:`R_{p}, R_{d} \in [100, 1500]`$ Barye :math:`\cdot` s/ml and one capacitance parameter :math:`C \in [1\times 10^{-5}, 1\times 10^{-2}]` ml/Barye.

This model is not identifiable. The average distal pressure is only affected by the total system resistance, i.e. the sum :math:`R_{p}+R_{d}`, leading to a negative correlation between these two parameters. Thus, an increment in the proximal resistance is compensated by a reduction in the distal resistance (so the average distal pressure remains the same) which, in turn, reduces the friction encountered by the flow exiting the capacitor. An increase in the value of :math:`C` is finally needed to restore the average, minimum and maximum pressure. This leads to a positive correlation between :math:`C` and :math:`R_{d}`.

The output consists of the maximum, minimum and average values of the proximal pressure :math:`P_{p}(t)`, i.e., :math:`(P_{p,\text{min}}, P_{p,\text{max}}, P_{p,\text{avg}})` over one heart cycle. The true parameters are :math:`z^{*}_{K,1} = R^{*}_{p} = 1000` Barye :math:`\cdot` s/ml, :math:`z^{*}_{K,2}=R^{*}_{d} = 1000` Barye :math:`\cdot` s/ml and :math:`C^{*} = 5\times 10^{-5}` ml/Barye and the proximal pressure is computed from the solution of the algebraic-differential system

.. math::
   Q_{p} = \frac{P_{p} - P_{c}}{R_{p}},\quad Q_{d} = \frac{P_{c}-P_{d}}{R_{d}},\quad \frac{d\, P_{c}}{d\,t} = \frac{Q_{p}-Q_{d}}{C},

where the distal pressure is set to :math:`P_{d}=55` mmHg.

Synthetic observations are generated from :math:`N(\boldsymbol\mu, \boldsymbol\Sigma)`, where :math:`\mu=(f_{1}(\boldsymbol{z}^{*}),f_{2}(\boldsymbol{z}^{*}),f_{3}(\boldsymbol{z}^{*}))^T = (P_{p,\text{min}}, P_{p,\text{max}}, P_{p,\text{ave}})^T = (100.96, 148.02,116.50)^T` and :math:`\boldsymbol\Sigma`` is a diagonal matrix with entries :math:`(5.05, 7.40, 5.83)^T`. The budgeted number of true model solutions is 216; the fixed surrogate model is evaluated on a :math:`6\times 6\times 6 = 216` pre-grid while the adaptive surrogate is evaluated with a pre-grid of size :math:`4\times 4\times 4 = 64` and the other 152 evaluations are adaptively selected.

The results are presented in :numref:`fig_rcr_res`. The posterior samples obtained through NoFAS capture well the non-linear correlation among the parameters and generate a fairly accurate posterior predictive distribution that overlaps with the observations. Additional details can be found in :cite:p:`wang2022variational`.

.. _fig_rcr_res:

.. figure:: imgs/rcr/log_plot_rcr-1.png
.. figure:: imgs/rcr/data_plot_rcr_25000_0_1-1.png
.. figure:: imgs/rcr/data_plot_rcr_25000_0_2-1.png
.. figure:: imgs/rcr/params_plot_rcr_25000_0_1-1.png
.. figure:: imgs/rcr/params_plot_rcr_25000_0_2-1.png
.. figure:: imgs/rcr/params_plot_rcr_25000_1_2-1.png

   Results from the RCR model. Loss profile (top), posterior predictive distribution (center) and posterior samples (bottom).
