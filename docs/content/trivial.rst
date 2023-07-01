Simple two-dimensional map with Gaussian likelihood
===================================================

A model $f:\mathbb{R}^{2}\to \mathbb{R}^{2}$ is chosen in this experiment having the closed-form expression
%
\begin{equation}
f(\bm z) = f(z_{1},z_{2}) = (z_1^3 / 10 + \exp(z_2 / 3), z_1^3 / 10 - \exp(z_2 / 3))^T.
\end{equation}
%
Observations $\bm{x}$ are generated as
\begin{equation}\label{eqn:exp1}
\bm{x} = \bm{x}^{*} + 0.05\,|\bm{x}^{*}|\,\odot\bm{x}_{0},
\end{equation}
where $\bm{x}_{0} \sim \mathcal{N}(0,\bm I_2)$ and $\odot$ is the Hadamard product. We set the \emph{true} model parameters at $\bm{z}^{*} = (3, 5)^T$, with output $\bm{x}^{*} = f(\bm z^{*})=(7.99, -2.59)^{T}$, and simulate 50 sets of observations from~\eqref{eqn:exp1}. The likelihood of $\bm z$ given $\bm{x}$ is assumed Gaussian and we adopt a noninformative uniform prior $p(\bm z)$. We allocate a budget of $4\times4=16$ model solutions to the pre-grid and use the rest to adaptively calibrate $\widehat{f}$ using $2$ samples every $1000$ normalizing flow iterations.

Results in terms of loss profile, variational approximation and posterior predictive distribution are shown in Figure~\ref{fig:trivial}.
%
\begin{figure}[!ht]
\centering
\includegraphics[scale=0.7]{imgs/trivial/log_plot_trivial.pdf}
\includegraphics[scale=0.75]{imgs/trivial/target_plot_trivial.pdf}
\includegraphics[scale=0.7]{imgs/trivial/sample_plot_trivial.pdf}
\caption{Results from the trivial model. Loss profile (left), posterior samples (center) and posterior predictive distribution (right).}\label{fig:trivial}
\end{figure}
