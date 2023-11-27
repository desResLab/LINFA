LINFA options
=============

.. list-table:: General parameters
   :widths: 25 25 25
   :header-rows: 1

   * - Option
     - Type
     - Description

   * - ``name``
     - ``str``
     - Name of the experiment

   * - ``flow_type``
     - ``str``
     - Type of normalizing flow (``'maf'`` or ``'realnvp'``)

   * - ``n_blocks``
     - ``int``
     - Number of normalizing flow layers (default ``5``)

   * - ``hidden_size``
     - ``int``
     - Number of neurons in MADE hidden layer (default ``100``)

   * - ``n_hidden``
     - ``int``
     - Number of hidden layers in MADE (default ``1``)

   * - ``activation_fn``
     - ``str``
     - | Activation function for MADE network 
       | used by MAF (default: ``'relu'``)

   * - ``input_order``
     - ``str``
     - | Input order for MADE mask creation 
       | (default: ``'sequential'`` or ``'random'``)

   * - ``batch_norm_order``
     - ``bool``
     - | Adds batchnorm layer after each MAF or 
       | RealNVP layer (default ``True``)

   * - ``save_interval``
     - ``int``
     - | How often to save results from the normalizing flow iterations. 
       | Saved results include posterior samples, loss profile, 
       | samples from the posterior predictive distribution with 
       | observations and marginal statistics

   * - ``input_size``
     - ``int``
     - Input dimensionality (default ``2``)

   * - ``batch_size``
     - ``int``
     - | Number of samples from the basic distribution 
       | generated at each iteration (default ``100``)

   * - ``true_data_num``
     - ``int``
     - | Number of additional true model evaluations at 
       | each surrogate model update (default ``2``)

   * - ``n_iter``
     - ``int``
     - Total number of NF iterations (default ``25001``)


.. list-table:: Optimizer and learning rate parameters
   :widths: 25 25 50
   :header-rows: 1

   * - Option
     - Type
     - Description

   * - ``optimizer``
     - ``string``
     - Type of SGD optimizer (default ``'Adam'``)

   * - ``lr``
     - ``float``
     - Learning rate (default ``0.003``)

   * - ``lr_decay``
     - ``float``
     - Learning rate decay (default ``0.9999``)

   * - ``lr_scheduler``
     - ``string``
     - | Type of learning rate scheduler 
       | (``'StepLR'`` or ``'ExponentialLR'``)

   * - ``lr_step``
     - ``int``
     - | Number of steps before learning rate 
       | reduction for the step scheduler

   * - ``log_interval``
     - ``int``
     - | Number of iterations between successive 
       | loss printouts (default ``10``)


.. list-table:: Output parameters
   :widths: 25 25 50
   :header-rows: 1

   * - Option
     - Type
     - Description

   * - ``output_dir``
     - ``string``
     - | Name of output folder where 
       | results files are written

   * - ``log_file``
     - ``string``
     - | Name of the log file which stores the iteration number, 
       | annealing temperature and value of the loss function at each iteration

   * - ``seed``
     - ``int``
     - Seed for random number generator

.. list-table:: Surrogate model parameters (NoFAS)
   :widths: 25 25 50
   :header-rows: 1

   * - Option
     - Type
     - Description

   * - ``n_sample``
     - ``int``
     - | Batch size used to generate results 
       | after ``save_interval`` iterations

   * - ``calibrate_interval``
     - ``int``
     - | Number of NF iteration between successive 
       | updates of the surrogate model (default ``1000``)

   * - ``budget``
     - ``int``
     - Maximum allowable number of true model evaluations

   * - ``surr_pre_it``
     - ``int``
     - | Number of pre-training iterations 
       | for surrogate model (default ``40000``)
   
   * - ``surr_upd_it``
     - ``int``
     - Number of iterations for the surrogate model update (default ``6000``)
   
   * - ``surr_folder``
     - ``str``
     - Folder where the surrogate model is stored (default ``'./'``)
   
   * - ``use_new_surr``
     - ``bool``
     - | Start by pre-training a new surrogate
       | and ignore existing surrogates (default ``True``)

   * - ``store_surr_interval``
     - ``int``
     - Save interval for surrogate model (None for no save, default ``None``)

.. list-table:: Parameters for the adaptive annealing scheduler (AdaAnn)
   :widths: 25 25 50
   :header-rows: 1

   * - Option
     - Type
     - Description

   * - ``annealing``
     - ``bool``
     - | Flag to activate the annealing scheduler. 
       | If this is ``False``, the target posterior 
       | distribution is left unchanged during 
       | the iterations

   * - ``scheduler``
     - ``string``
     - | Type of annealing scheduler 
       | (default ``'AdaAnn'`` or ``'fixed'``)

   * - ``tol``
     - ``float``
     - | KL tolerance. It is kept constant during inference and used 
       | in the numerator of equation :eq:`equ:adaann`.

   * - ``t0``
     - ``float``
     - Initial inverse temperature.

   * - ``N``
     - ``int``
     - Number of batch samples during annealing.

   * - ``N_1``
     - ``int``
     - Number of batch samples at :math:`t=1`.

   * - ``T_0``
     - ``int``
     - Number of initial parameter updates at :math:`t_0`.

   * - ``T``
     - ``int``
     - | Number of parameter updates after each temperature update. 
       | During such updates the temperature is kept fixed.

   * - ``T_1``
     - ``int``
     - Number of parameter updates at :math:`t=1`

   * - ``M``
     - ``int``
     - | Number of Monte Carlo samples used to evaluate 
       | the denominator in equation :eq:`equ:adaann`


.. list-table:: Device parameters
   :widths: 25 25 50
   :header-rows: 1

   * - Option
     - Type
     - Description

   * - ``no_cuda``
     - ``bool``
     - Do not use GPU acceleration
