LINFA options
*************

.. list-table:: General parameters
   :widths: 25 25 50
   :header-rows: 1

   * - Option
     - Type
     - Description

   * - ``name`` 
     - str
     - experiment name

   * - flow\_type
     - str
     - type of normalizing flow (maf,realnvp)

   * - n\_blocks
     - int
     - Number of normalizing flow layers (default 5)

   * - hidden\_size
     - int
     - Number of neurons in MADE hidden layer (default 100)

   * - n\_hidden
     - int
     - Number of hidden layers in MADE (default 1)

   * - activation\_fn
     - str
     - Activation function used (WHERE???) (default: 'relu')

   * - input\_order
     - str
     - Input order for mask creation {\bf EXPLAIN!!!!} (default: 'sequential')

   * - batch\_norm\_order
     - bool
     - Order to decide if batch normalization layer is used (default True)

   * - sampling\_interval
     - int
     - How often to sample from normalizing flow

   * - input\_size
     - int
     - Dimensionality of input (default: 2)

   * - batch\_size
     - int
     - Number of samples from the basic distribution generated at each iteration (default 100)

   * - true\_data\_num
     - double
     - number of true model evaluated (default: 2)

   * - n\_iter
     - int
     - Number of iterations (default 25001)\\


.. \begin{table}[!ht]
.. \centering
.. \caption{Optimizer and learning rate parameters}
.. \begin{tabular}{p{4cm} p{2cm} p{8cm}} 
.. \toprule
.. {\bf Option} & {\bf Type} & {\bf Description}\\
.. \midrule
.. \emph{\texttt{optimizer}} & string & type of optimizer used (default: 'Adam')\\
.. \emph{\texttt{lr}} & float & Learning rate (default 0.003)\\
.. \emph{\texttt{lr\_decay}} & float & Learning rate decay (default 0.9999)\\
.. \emph{\texttt{lr\_scheduler}} & string & type of learning rate scheduler used\\
.. \emph{\texttt{lr\_step}} & int & Number of steps for learning rate step scheduler\\
.. \emph{\texttt{log\_interval}} & int & Number of interval between two successive plots of loss summary plot interval (default 10)\\
.. \bottomrule
.. \end{tabular}
.. \end{table}

.. \begin{table}[!ht]
.. \centering
.. \caption{Output parameters}
.. \begin{tabular}{p{4cm} p{2cm} p{8cm}} 
.. \toprule
.. {\bf Option} & {\bf Type} & {\bf Description}\\
.. \midrule
.. \emph{\texttt{output\_dir}} & string & output folder\\
.. \emph{\texttt{results\_file}} & string & {\bf\color{red}What exactly is writing in the result file??}\\
.. \emph{\texttt{log\_file}} & string & name of the log file which stores {\bf\color{red} What are we writing in the log file??}\\
.. \emph{\texttt{samples\_file}} & string & Name of the file where all samples are stored {\bf\color{red} do we keep track of the samples at all iterations? How do we distinguish between iterations??}\\
.. \emph{\texttt{seed}} & int & Seed for random number generator\\
.. \bottomrule
.. \end{tabular}
.. \end{table}

.. \begin{table}[!ht]
.. \centering
.. \caption{Surrogate model parameters (NoFAS)}
.. \begin{tabular}{p{4cm} p{2cm} p{8cm}} 
.. \toprule
.. {\bf Option} & {\bf Type} & {\bf Description}\\
.. \midrule
.. \emph{\texttt{use\_surrogate}} & bool & decide if the surrogate model is used\\
.. \emph{\texttt{n\_sample}} & int & Total number of iterations {\bf\color{red}Not clear what this is exactly...}\\
.. \emph{\texttt{calibrate\_interval}} & int & How often to update surrogate model (default 1000)\\
.. \emph{\texttt{budget}} & int & Maximum allowable number of true model evaluation\\
.. \bottomrule
.. \end{tabular}
.. \end{table}

.. \begin{table}[!ht]
.. \centering
.. \caption{Parameters for the adaptive annealing scheduler (AdaAnn)}\label{tab:adaann}
.. \begin{tabular}{p{4cm} p{2cm} p{8cm}} 
.. \toprule
.. {\bf Option} & {\bf Type} & {\bf Description}\\
.. \midrule
.. \emph{\texttt{annealing}} & bool & is used to activate an annealing scheduler. If this is \emph{\texttt{False}}, the target posterior distribution is left unchanged during the iterations.\\
.. \emph{\texttt{scheduler}} & string & defines the type of annealing scheduler. This includes a \emph{\texttt{fixed}} scheduler as well as the \emph{\texttt{AdaAnn}} adaptive scheduler (default 'AdaAnn').\\
.. \emph{\texttt{tol}} & float & KL tolerance. It is kept constant during inference and used in the numerator of~\eqref{equ:adaann}.\\
.. \emph{\texttt{t0}} & float & Initial inverse temperature.\\
.. \emph{\texttt{N}} & int & Number of batch samples during annealing.\\
.. \emph{\texttt{N\_1}} & int & Number of batch samples at $t=1$.\\
.. \emph{\texttt{T\_0}} & int & Number of initial parameter updates at initial $t_0$.\\
.. \emph{\texttt{T}} & int & Number of parameter updates during annealing.\\
.. \emph{\texttt{T\_1}} & int & Number of parameter updates at $t=1$.\\
.. \emph{\texttt{M}} & int & Number of Monte Carlo samples used to evaluate the denominator in~\eqref{equ:adaann}.\\
.. \bottomrule
.. \end{tabular}
.. \end{table}

.. \begin{table}[!ht]
.. \centering
.. \caption{Device parameters}
.. \begin{tabular}{p{4cm} p{2cm} p{8cm}} 
.. \toprule
.. {\bf Option} & {\bf Type} & {\bf Description}\\
.. \midrule
.. \emph{\texttt{no\_cuda}} & bool & Do not use GPU acceleration\\
.. \bottomrule
.. \end{tabular}
.. \end{table}