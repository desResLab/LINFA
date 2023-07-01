Friedman 1 model
================

We consider a modified version of the Friedman 1 dataset [friedman1991multivariate]_ to examine the performance of  our adaptive annealing scheduler in a high-dimensional context. According to the original model in [friedman1991multivariate]_, the data are generated as

.. math::
   \textstyle y_i = \mu_i(\boldsymbol{\beta})+ \epsilon_i, \mbox{ where }
   \mu_i(\boldsymbol{\beta})=\beta_1\text{sin}(\pi x_{i,1}x_{i,2})+ \beta_2(x_{i,3}-\beta_3)^2+\sum_{j=4}^{10}\beta_jx_{i,j}, 
   :label: eqn:friedman1

where :math:`\epsilon_i\sim\mathcal{N}(0,1)`. We made a slight modification to the model in~\eqref{eqn:friedman1} as

.. math::
   \mu_i(\boldsymbol{\beta}) = \textstyle \beta_1\text{sin}(\pi x_{i,1}x_{i,2})+ \beta_2^2(x_{i,3}-\beta_3)^2+\sum_{j=4}^{10}\beta_jx_{i,j},
   :label: eqn:friedman1_modified

and set the true parameter combination to :math:`\boldsymbol{\beta}=(\beta_1,\ldots,\beta_{10})=(10,\pm \sqrt{20}, 0.5, 10, 5, 0, 0, 0, 0, 0)`. Note that both :eq:`eqn:friedman1` and :eq:`eqn:friedman1_modified` contain linear, non-linear, and interaction terms of the input variables :math:`X_1` to :math:`X_{10}`, five of which (:math:`X_6` to :math:`X_{10}`) are irrelevant to :math:`Y`. 
Each :math:`X` is drawn independently from :math:`\mathcal{U}(0,1)`. We used R package `tgp` [gramacy2007tgp]_ to generate a Friedman 1 dataset with a sample size of :math:`n=1000`.

We impose a non-informative uniform prior :math:`p(\boldsymbol{\beta})` and, unlike the original modal, we now expect a bimodal posterior distribution of :math:`\boldsymbol{\beta}`. Results in terms of marginal statistics and their convergence for the mode with positive :math:`z_{K,2}` are illustrated in Table :eq:`table:Friedman_bimodal_stats` and Figure :fig:`fig:adaann_res`.

\begin{minipage}{\textwidth}
  \begin{minipage}[b]{0.4\textwidth}
    \centering
    \resizebox{.8\textwidth}{!}{%
    \begin{tabular}[2in]{l c c c c}
    \toprule
    \textbf{True} & \multicolumn{2}{c}{\textbf{Mode 1}}\\
    \textbf{Value} & Post. Mean & Post. SD\\
    \midrule
    $\beta_1 = 10$   & 10.0285 & 0.1000\\
    $\beta_2 = \pm \sqrt{20}$ & 4.2187 & 0.1719\\
    $\beta_3 = 0.5$  & 0.4854 & 0.0004\\
    $\beta_4 = 10$   & 10.0987 & 0.0491\\
    $\beta_5 = 5$    & 5.0182 & 0.1142\\
    $\beta_6 = 0$    & 0.1113 & 0.0785\\
    $\beta_7 = 0$    & 0.0707 & 0.0043\\
    $\beta_8 = 0$    & -0.1315 & 0.1008\\
    $\beta_9 = 0$    & 0.0976 & 0.0387\\
    $\beta_{10} = 0$ & 0.1192 & 0.0463\\
    \bottomrule
    \end{tabular}}  
    \captionof{table}{Posterior mean and standard deviation for positive mode in the modified Friedman test case.}\label{table:Friedman_bimodal_stats}    
  \end{minipage}
\hfill
\begin{minipage}[b]{0.58\textwidth}
\centering
\includegraphics[width=0.4\textwidth]{imgs/adaann/log_plot.pdf}
\includegraphics[width=0.58\textwidth]{imgs/adaann/adaann.pdf}
\captionof{figure}{Loss profile (left) and posterior marginal statistics for positive mode in the modified Friedman test case.}\label{fig:adaann_res}
\end{minipage}
\end{minipage}
