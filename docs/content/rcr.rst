Three-element Wndkessel Circulatory Model
=========================================

The three-parameter Windkessel or \emph{RCR} model is characterized by proximal and distal resistance parameters $R_{p}, R_{d} \in [100, 1500]$ Barye$\cdot$ s/ml and one capacitance parameter $C \in [1\times 10^{-5}, 1\times 10^{-2}]$ ml/Barye.  
%
This model is not identifiable. The average distal pressure is only affected by the total system resistance, i.e. the sum $R_{p}+R_{d}$, leading to a negative correlation between these two parameters. Thus, an increment in the proximal resistance is compensated by a reduction in the distal resistance (so the average distal pressure remains the same) which, in turn, reduces the friction encountered by the flow exiting the capacitor. An increase in the value of $C$ is finally needed to restore the average, minimum and maximum pressure. This leads to a positive correlation between $C$ and $R_{d}$.

The output consists of the maximum, minimum and average values of the proximal pressure $P_{p}(t)$, i.e., $(P_{p,\text{min}}, P_{p,\text{max}}, P_{p,\text{avg}})$ over one heart cycle.
%
The true parameters are $z^{*}_{K,1} = R^{*}_{p} = 1000$ Barye$\cdot$s/ml, $z^{*}_{K,2}=R^{*}_{d} = 1000$ Barye$\cdot$s/ml and $C^{*} = 5\times 10^{-5}$ ml/Barye and the proximal pressure is computed from the solution of the algebraic-differential system
%
\begin{equation}
Q_{p} = \frac{P_{p} - P_{c}}{R_{p}},\quad Q_{d} = \frac{P_{c}-P_{d}}{R_{d}},\quad \frac{d\, P_{c}}{d\,t} = \frac{Q_{p}-Q_{d}}{C},
\end{equation}
%
where the distal pressure is set to $P_{d}=55$ mmHg.
%
Synthetic observations are generated from $N(\bm\mu, \bm\Sigma)$, where $\mu=(f_{1}(\bm{z}^{*}),f_{2}(\bm{z}^{*}),f_{3}(\bm{z}^{*}))^T = (P_{p,\text{min}}, P_{p,\text{max}}, P_{p,\text{ave}})^T = (100.96,$ $148.02,$ $ 116.50)^T$ and $\bm\Sigma$ is a diagonal matrix with entries $(5.05, 7.40, 5.83)^T$. The budgeted number of true model solutions is $216$; the fixed surrogate model is evaluated on a $6\times 6\times 6 = 216$ pre-grid while the adaptive surrogate is evaluated with a pre-grid of size $4\times 4\times 4 = 64$ and the other 152 evaluations are adaptively selected.

The results are presented in Figure~\ref{fig:rcr_res}. The posterior samples obtained through NoFAS capture well the non-linear correlation among the parameters and generate a fairly accurate posterior predictive distribution that overlaps with the observations. Additional details can be found in~\cite{wang2022variational}.
%
\begin{figure}[!ht]
\centering
\includegraphics[scale=0.7]{imgs/rcr/log_plot_rcr.pdf}
\includegraphics[scale=0.7]{imgs/rcr/data_plot_rcr_25000_0_1.pdf}
\includegraphics[scale=0.7]{imgs/rcr/data_plot_rcr_25000_0_2.pdf}\\
\includegraphics[scale=0.65]{imgs/rcr/params_plot_rcr_25000_0_1.pdf}
\includegraphics[scale=0.65]{imgs/rcr/params_plot_rcr_25000_0_2.pdf}
\includegraphics[scale=0.65]{imgs/rcr/params_plot_rcr_25000_1_2.pdf}
\caption{Results from the RCR model. Loss profile (left), posterior predictive distribution (center) and posterior samples (right).}\label{fig:rcr_res}
\end{figure}
