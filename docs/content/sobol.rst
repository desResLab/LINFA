High-dimensional Sobol' Function
================================

We consider a map $f: \mathbb{R}^{5}\to\mathbb{R}^{4}$ expressed as
\begin{equation}
f(\bm{z}) = \bm{A}\,\bm{g}(e^{\bm{z}}),
\end{equation}
where $g_i(\bm{r}) = (2\cdot |2\,a_{i} - 1| + r_i) / (1 + r_i)$ with $r_i > 0$ for $i=1,\dots,5$ is the \emph{Sobol}  function~\cite{sobol2003theorems} and $\bm{A}$ is a $4\times5$ matrix. We also set
\begin{equation*}
\bm{a} = (0.084, 0.229, 0.913, 0.152, 0.826)^T \mbox{ and }\bm{A} = \frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 1 & 0 & 0 & 0\\
0 & 1 & 1 & 0 & 0\\
0 & 0 & 1 & 1 & 0\\
0 & 0 & 0 & 1 & 1\\
\end{pmatrix}.
\end{equation*}
%
The true parameter vector is set at $\bm{z}^{*} = (2.75,$ $-1.5, 0.25,$ $-2.5,$ $1.75)^T$. While the Sobol function is bijective and analytic, $f$ is over-parameterized and non identifiabile.
%
This is also confirmed by the fact that the curve segment $\gamma(t) = g^{-1}(g(\bm z^*) + \bm v\,t)\in Z$ gives the same model solution as $\bm{x}^{*} = f(\bm{z}^{*}) = f(\gamma(t)) \approx (1.4910,$ $1.6650,$ $1.8715,$ $1.7011)^T$ for $t \in (-0.0153, 0.0686]$, where $\bm v = (1,-1,1,-1,1)^T$. 
%
This is consistent with the one-dimensional null-space of the matrix $\bm A$.
%
We also generate synthetic observations from the Gaussian distribution
%
\begin{equation}
\bm{x} = \bm{x}^{*} + 0.01\cdot |\bm{x}^{*}| \odot \bm{x}_{0},\,\,\text{and}\,\,\bm{x}_{0} \sim \mathcal{N}(0,\bm I_5).
\end{equation}
%
Results are shown in Figure~\ref{fig:highdim}.
%
\begin{figure}[!ht]
\centering
\includegraphics[scale=0.7]{imgs/highdim/log_plot.pdf}
\includegraphics[scale=0.7]{imgs/highdim/data_plot_highdim_25000_0_2.pdf}
\includegraphics[scale=0.7]{imgs/highdim/data_plot_highdim_25000_2_3.pdf}\\
%
\includegraphics[scale=0.7]{imgs/highdim/params_plot_highdim_25000_0_1.pdf}
\includegraphics[scale=0.7]{imgs/highdim/params_plot_highdim_25000_1_2.pdf}
\includegraphics[scale=0.7]{imgs/highdim/params_plot_highdim_25000_3_4.pdf}
\caption{Results from the high-dimensional model. Loss profile, posterior samples and posterior predictive distribution.}\label{fig:highdim}
\end{figure}
