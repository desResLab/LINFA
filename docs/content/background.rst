Background Theory
*****************


% ==================
###Background}\label{sec:background}
% ==================

% ======================================================
\subsection{Variational inference with normalizing flow}
% ======================================================

A normalizing flow (NF) is a nonlinear transformation designed to map an easy-to-sample base distribution $q_{0}(\bm{z})$ defined using the \emph{latent variables} $\bm{z}_{0}\in\bm{\mathcal{Z}}$, e.g. a standard Gaussian, into a close approximation $q_{K}(\bm{z}_{K})$ of a desired target posterior density $p(\bm{z}|\bm{x})$, where $\bm{x}\in\bm{\mathcal{X}}$ represent available observations that are approximated by model outputs
\begin{equation}
\bm{x} = f(\bm{z}) + \epsilon.
\end{equation}
This transformation can be determined by composing $K$ bijections and evaluating the transformed density through the change of variable formula~\citep[see][]{villani2009optimal}.
%
In the context of variational inference, we seek to determine an \emph{optimal} set of parameters $\bm{\lambda}$ so that $q_{K}\approx p(\cdot)$. 
%
For computational convenience, normalizing flows transformations are selected so are easily inverted and their Jacobian determinant can be computed with a cost that grows linearly with the problem dimensionality. Approaches in the literature include RealNVP~\citep{dinh2016density}, GLOW~\citep{kingma2018glow} and autoregressive transformations such as MAF~\citep{papamakarios2018masked} and IAF~\citep{kingma2016improved}.

Given observations $\bm{x}\in\mathcal{\bm{X}}$, a likelihood function $l_{\bm{z}}(\bm{x})$ (informed by the distribution of the error $\epsilon$) and prior $p(\bm{z})$, a NF-based approximation $q_K(\bm{z})$ of the posterior distribution $p(\bm{z}|\bm{x})$ can be computed by maximizing the lower bound to the log marginal likelihood $\log p(\bm{x})$ (the so-called \emph{evidence lower bound} or ELBO), or, equivalently, by minimizing a \emph{free energy bound}~\citep[see, e.g.,][]{rezende2015variational}.

% ==========================
\subsection{MAF and RealNVP}
% ==========================

LINFA implements two widely used normalizing flow formulations, MAF~\citep{papamakarios2018masked} and RealNVP~\citep{dinh2016density}.
MAF belongs to the class of \emph{autoregressive} normalizing flows. Given the latent variable $\bm{z} = (z_{1},z_{2},\dots,z_{d})$ the approach assumes $p(z_i|z_{1},\dots,z_{i-1}) = \phi[(z_i - \mu_i) / e^{\alpha_i}]$, where $\phi$ is the standard normal density, $\mu_i = f_{\mu_i}(z_{1},\dots,z_{i-1})$, $\alpha_i = f_{\alpha_i}(z_{1},\dots,z_{i-1})$, and $f_{\mu_i}$ and $f_{\alpha_i}$ are masked autoencoder neural networks~\citep[MADE,][]{germain2015made}. 
%
In a MADE autoencoder the network connectivities are multiplied by Boolean masks so the input-output relation maintains a a lower triangular structure, making the computation of the Jacobian determinant particularly simple. 
%
MAF transformations are then composed of multiple MADE layers, possibly interleaved by batch normalization layers~\citep{ioffe2015batch}, typically used to add stability during training and increase network accuracy~\citep{papamakarios2018masked}.

% REALNVP
RealNVP is another widely used flow where, at each layer the first $d'$ variables are left unaltered while the remaining $d-d'$ are subject to an affine transformation of the form $\widehat{\bm{z}}_{d'+1:d} = \bm{z}_{d'+1:d}\,\odot\,e^{\bm{\alpha}} + \bm{\mu}$, where $\bm{\mu} = f_{\mu}(\bm{z}_{1:d'})$ and $\bm{\alpha} = f_{\alpha}(\bm{z}_{d'+1:d})$ are MADE autoencoders. 
%
In this context, MAF could be seen as a generalization of RealNVP by setting $\mu_i=\alpha_i=0$ for $i\leq d'$~\citep{papamakarios2018masked}.


% ===========================================================
\subsection{Normalizing flow with adaptive surrogate (NoFAS)}
% ===========================================================

LINFA is designed to accommodate black-box models $f: \bm{\mathcal{Z}} \to \bm{\mathcal{X}}$ between the random inputs $\bm{z} = (z_1, z_2, \cdots, z_d)^T \in \bm{\mathcal{Z}}$ and the outputs $(x_1, x_2,\cdots,x_m)^T \in \bm{\mathcal{X}}$, and assume $n$ observations $\bm x = \{\bm x_i\}_{i=1}^n \subset \bm{\mathcal{X}}$ to be available. 
%
Our goal is to infer $\bm z$ and to quantify its uncertainty given $\bm{x}$. 
We employ a variational Bayesian paradigm and sample from the posterior distribution $p(\bm z\vert \bm x)\propto \ell(\bm z; f,\bm x)\,\pi(\bm z)$, with prior $\pi(\bm z)$ via normalizing flows. 

This requires the evaluation of the gradient of the ELBO with respect to the NF parameters $\bm{\lambda}$, replacing $p(\bm x, \bm z_K)$ with $p(\bm x\vert\bm z_K)\,\pi(\bm z_k)$ $=\ell(\bm z_K; f,\bm x)\,\pi(\bm z_k)$, and approximating the expectations with their MC estimates. 
%
However, the likelihood function needs to be evaluated at every MC realization, which can be costly if the model $f(\bm{z})$ is computationally expensive. In addition, automatic differentiation through a legacy (e.g. physics-based) solver may be an impractical, time-consuming, or require the development of an adjoint solver.

Our solution is to replace the model $f$ with a computationally inexpensive surrogate $\widehat{f}: \bm{\mathcal{Z}} \times \bm{\Omega} \to \bm{\mathcal{X}}$ parameterized by $\bm{\omega} \in \bm{\Omega}$, whose derivatives can be obtained at a relatively low computational cost, but intrinsic bias in the selected surrogate formulation, a limited number of training examples, and locally optimal $\bm{\omega}$ can compromise the accuracy of $\widehat{f}$.

To resolve these issues, LINFA implements NoFAS, which updates the surrogate model adaptively by smartly weighting the samples of $\bm{z}$ from NF thanks to a loss function with \emph{memory-aware terms}.
Once a newly updated surrogate is obtained, the likelihood function is updated, leading to a new posterior distribution that will be approximated by VI-NF, producing, in turn, new samples for the next surrogate model update, and so on. 
Additional details can be found in~\cite{wang2022variational}.

% =============================
\subsection{Adaptive Annealing}
% =============================

Annealing is a technique to parametrically smooth a target density to improve sampling efficiency and accuracy during inference. 
%
In the discrete case, this is achieved by incrementing an \emph{inverse temperature} $t_{k}$ and setting $p_k(\bm{Z},\bm{X}) = p^{t_k}(\bm{Z},\bm{X}),\,\,\text{for } k=0,\dots,K$, where $0 < t_{0} < \cdots < t_{K} \le 1$.
%
The result of exponentiation produces a smooth unimodal distribution for a sufficiently small $t_0$, recovering the target density for $t_{K}$ close to 1. In other words, annealing provides a continuous deformation from an easier to approximate unimodal distribution to a desired target density.

An a-priori defined linear annealing scheduler~\citep[see, e.g.,][]{rezende2015variational} is often used in practice, where \mbox{$t_j=t_{0} + j (1-t_{0})/K$} for \mbox{$j=0,\ldots,K$}
with constant increments 
\mbox{$\epsilon = (1-t_{0})/K$}. 
%
Intuitively, small temperature changes are desirable to carefully explore the parameter spaces at the beginning of the annealing process, whereas larger changes can be taken as $t_{k}$ increases, after annealing has helped the approximate distribution to capture important features of the target distribution (e.g., locating all the relevant modes).

The proposed AdaAnn scheduler determines the increment $\epsilon_{k}$ that approximately produces a pre-defined change in the KL divergence between two distributions tempered at~$t_{k}$ and $t_{k+1}=t_{k}+\epsilon_{k}$, respectively.
%
Letting the KL divergence equal a constant $\tau^2/2$, where $\tau$ is referred to as the KL divergence tolerance, the step size $\epsilon_k$ becomes 
%
\begin{equation}\label{equ:adaann}
\epsilon_k = \tau/ \sqrt{\mathbb{V}_{p^{t_k}}[\log p(\bm Z)]}. 
\end{equation}
%
The denominator is large when the support of the annealed distribution $p^{t_{k}}(\bm{z})$ is wider than the support of the target $p(\bm{z})$, and progressively reduces with increasing $t_{k}$.
%
Further detail on the derivation of the expression for $\epsilon_{k}$ can be found in~\cite{cobian2023adaann}.

% ====================
\section{Capabilities}\label{sec:capabilities}
% ====================

LINFA is designed as a general inference engine and allows the user to define computational models, surrogates, likelihood formulation and input transformations. 

% Model agnostic
\noindent{\bf Supports user-defined computational models} - LINFA can accommodate any type of model, from analytically defined posteriors with gradient computed through automatic differentiation, to legacy computational solvers for which the solution gradient is not available nor easy to compute. New models are created by adding new classes derived from the \emph{computational\_model} class, and providing an implementation for the methods below
%
\vspace{-3pt}
%
\begin{itemize}\itemsep -3pt
\item \emph{\texttt{def genDataFile(dataSize=50, dataFileName="...", store=True)}} - This function is used to generate synthetic observations. It will compute the model output corresponding to the default parameter values and add zero-mean noise with user-specified standard deviation. Observations will be stored so the likelihood can be computed with \texttt{evalNegLL\_t}.
%
\item \emph{\texttt{def solve\_t(params)}} - This function solves the model for multiple values of the input parameters specified in the matrix \emph{\texttt{params}}.
%
\item \emph{\texttt{def evalNegLL\_t(params, surrogate=True)}} - Evaluates the negative log-likelihood at multiple realizations stored in \emph{\texttt{params}}. If available, a surrogate model can be specified to reduce the computational cost.
%
\item \emph{\texttt{def den\_t(params, surrogate=True)}} - Evaluate the posterior density using the inputs in \emph{\texttt{params}} using the true model or a surrogate.
\end{itemize}

\noindent{\bf Supports user-defined surrogate models} - For computational models that are too expensive to allow for online inference, LINFA provides functionalities to create, train and fine-tune a \emph{Surrogate model}. The \emph{\texttt{Surrogate}} class is provided, which implements the following functionalities: 
%
\vspace{-3pt}
%
\begin{itemize}\itemsep -3pt
\item A new surrogate model can be created using the \emph{\texttt{Surrogate}} constructor. 

% or \emph{\texttt{Surrogate(model\_name, model\_func, input\_size, output\_size, limits=None, memory\_len=20, surrogate=None)}}. 
%
\item The limits (i.e. upper and lower bounds) for the model inputs can be either interrogated or modified. They are stored as a list of lists using the format \emph{\texttt{[[low\_0, high\_0], [low\_1, high\_1], ...]}}.

\item A \emph{pre-grid} is defined as an a-priori selected 

%Generation or acquisiton of a pre-grid 
%    def pre_grid(self):
%            @pre_grid.setter
%    def pre_grid(self, pre_grid):
%    def gen_grid(self, input_limits=None, gridnum=4, store=True):

\item Surrogate model I/O. The two functions \emph{\texttt{surrogate\_save()}} and \emph{\texttt{surrogate\_load()}} are provided to save a snapshot of a given surrogate or to read it from a file. 
%
% def pre_train(self, max_iters, lr, lr_exp, record_interval, store=True, reg=False):
% def update(self, x, max_iters=10000, lr=0.01, lr_exp=0.999, record_interval=500, store=False, tol=1e-5, reg=False):
\item The \emph{\texttt{pre\_train()}} function is provided to perform an initial training of the surrogate model on the pre-grid. In addition, the \emph{\texttt{update}} function is also available for re-train the model once additional traning examples are available. 
%
\item The \emph{\texttt{forward(x)}} function is also available to evaluate the surrogate model at $n$ input realizations each defined as a vector of dimension $d$ stored in a $n\times d$ matrix.
\end{itemize}

\noindent{\bf Supports user-defined likelihood} - A user-defined likelihood function can be specified through the \emph{\texttt{log\_density(x, model, surrogate)}} function and then assigning it to the \emph{\texttt{experiment.model\_logdensity}} function as \emph{\texttt{exp.model\_logdensity = lambda x: log\_density(x, model, exp.surrogate)}}.

\noindent{\bf Supports user-defined transformations} - {\bf\color{red}Complete!!!}

% Adaptive annealing schedule
\noindent{\bf Automatic definition of temperature increments for posterior annealing} - LINFA offers a number of options for the definition of annealing schedulers. Currently implemented schedulers include a \emph{fixed scheduler} and adaptive \emph{AdaAnn} scheduler from~\citep{cobian2023adaann}. The parameters are described in Table~\label{tab:adaann}. {\bf\color{red}Define a class??}

% ============================
\section{Numerical benchmarks}\label{sec:benchmarks}
% ============================

% ==============================================================
\subsection{Simple two-dimensional map with Gaussian likelihood}
% ==============================================================

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
where $\bm{x}_{0} \sim \mathcal{N}(0,\bm I_2)$ and $\odot$ is the Hadamard product. 
%
We set the \emph{true} model parameters at $\bm{z}^{*} = (3, 5)^T$, with output $\bm{x}^{*} = f(\bm z^{*})=(7.99, -2.59)^{T}$, and simulate 50 sets of observations from~\eqref{eqn:exp1}. The likelihood of $\bm z$ given $\bm{x}$ is assumed Gaussian and we adopt a noninformative uniform prior $\pi(\bm z)$.
%
We allocate a budget of $4\times4=16$ model solutions to the pre-grid and use the rest to adaptively calibrate $\widehat{f}$ using $2$ samples every $1000$ normalizing flow iterations.

Results in terms of loss profile, variational approximation and posterior predictive distribution are shown in Figure~\ref{fig:trivial}.
%
\begin{figure}[!ht]
\centering
\includegraphics[scale=0.8]{imgs/trivial/log_plot_trivial.pdf}
\includegraphics[scale=0.85]{imgs/trivial/target_plot_trivial.pdf}
\includegraphics[scale=0.8]{imgs/trivial/sample_plot_trivial.pdf}
\caption{Results from the trivial model. Loss profile (left), posterior samples (center) and posterior predictive distribution (right).}\label{fig:trivial}
\end{figure}

% ===================================
\subsection{High-dimensional example}
% ===================================

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
\includegraphics[scale=0.8]{imgs/highdim/log_plot_highdim.pdf}
\includegraphics[scale=0.8]{imgs/highdim/sample_plot_highdim.pdf}
\includegraphics[scale=0.8]{imgs/highdim/target_plot_highdim_0_1.pdf}\\
\includegraphics[scale=0.7]{imgs/highdim/target_plot_highdim_0_1.pdf}
\includegraphics[scale=0.7]{imgs/highdim/target_plot_highdim_0_2.pdf}
\includegraphics[scale=0.7]{imgs/highdim/target_plot_highdim_0_3.pdf}\\
\caption{{\bf\color{red}COMPLETE!!!}}\label{fig:highdim}
\end{figure}

% =======================================
\subsection{Two-element Windkessel Model}
% =======================================

The two-element Windkessel model (often referred to as the \emph{RC} model) is the simplest representation of the human systemic circulation and requires two parameters, i.e., a resistance $R \in [100, 1500]$ Barye$\cdot$ s/ml and a capacitance $C \in [1\times 10^{-5}, 1 \times 10^{-2}]$ ml/Barye. 
%
We provide a periodic time history of the aortic flow (see~\cite{wang2022variational} for additional details) and use the RC model to predict the time history of the proximal pressure $P_{p}(t)$, specifically its maximum (max), minimum (min) and average (ave) values over a typical heart cycle, while assuming the distal resistance $P_{d}(t)$ as a constant in time, equal to 55 mmHg. 
%
In our experiment, we set the true resistance and capacitance as $z_{1}^{*}=R^{*} = 1000$ Barye$\cdot$ s/ml and $z_{2}^{*}=C^{*} = 5\times 10^{-5}$ ml/Barye and determine $P_{p}(t)$ from a RK4 numerical solution of the following algebraic-differential system of two equations
%
\begin{equation}\label{equ:RC}
Q_{d} = \frac{P_{p}-P_{d}}{R},\quad \frac{d P_{p}}{d t} = \frac{Q_{p} - Q_{d}}{C},
\end{equation}
%
where $Q_{p}$ is the flow entering the RC system and $Q_{d}$ is the distal flow.
%
Synthetic observations are generated by adding Gaussian noise to the true model solution $\bm{x}^{*}=(P_{p,\text{min}},$ $P_{p,\text{max}},$ $P_{p,\text{ave}})= (78.28, 101.12,  85.75)$, i.e., $\widetilde{\bm{x}}$ follows a multivariate Gaussian distribution with mean $\bm{x}^{*}$ and a diagonal covariance matrix with entries $0.05\,x_{i}^{*}$, where $i=1,2,3$ corresponds to the maximum, minimum, and average pressures, respectively. 
%
The aim is to quantify the uncertainty in the RC model parameters given 50 repeated pressure measurements. We imposed a non-informative prior on $R$ and $C$. 
%
\begin{figure}[!ht]
\centering
\includegraphics[scale=0.8]{imgs/rc/log_plot_rc.pdf}
\includegraphics[scale=0.8]{imgs/rc/sample_plot_rc.pdf}
\includegraphics[scale=0.8]{imgs/rc/target_plot_rc_0_1.pdf}
\caption{Results from the RC model. Loss profile (left), posterior predictive distribution (center) and posterior samples (right).}\label{fig:rc_res}
\end{figure}

% ============================================================
\subsection{Three-element Wndkessel Circulatory Model (NoFAS)}
% ============================================================

The three-parameter Windkessel or \emph{RCR} model is characterized by proximal and distal resistance parameters $R_{p}, R_{d} \in [100, 1500]$ Barye$\cdot$ s/ml and one capacitance parameter $C \in [1\times 10^{-5}, 1\times 10^{-2}]$ ml/Barye.  
%
Even if it consists of a relatively simple lumped parameter formulation, the RCR circuit model is not identifiable. The average distal pressure is only affected by the total system resistance, i.e. the sum $R_{p}+R_{d}$, leading to a negative correlation between these two parameters. Thus, an increment in the proximal resistance is compensated by a reduction in the distal resistance (so the average distal pressure remains the same) which, in turn, reduces the friction encountered by the flow exiting the capacitor. An increase in the value of $C$ is finally needed to restore the average, minimum and maximum pressure. This leads to a positive correlation between $C$ and $R_{d}$.

The output consists of the proximal pressure $P_{p}(t)$, specifically its maximum, minimum and average values $(P_{p,\text{min}}, P_{p,\text{max}}, P_{p,\text{ave}})$ over one heart cycle.
%
The true parameters are $z^{*}_{1} = R^{*}_{p} = 1000$ Barye$\cdot$s/ml, $z^{*}_{2}=R^{*}_{d} = 1000$ Barye$\cdot$s/ml and $C^{*} = 5\times 10^{-5}$ ml/Barye and the proximal pressure is computed from the solution of the algebraic-differential system
%
\begin{equation}
Q_{p} = \frac{P_{p} - P_{c}}{R_{p}},\quad Q_{d} = \frac{P_{c}-P_{d}}{R_{d}},\quad \frac{d\, P_{c}}{d\,t} = \frac{Q_{p}-Q_{d}}{C},
\end{equation}
%
where the distal pressure is set to $P_{d}=55$ mmHg.
%
Synthetic observations are generated from $N(\bm\mu, \bm\Sigma)$, where $\mu=(f_{1}(\bm{z}^{*}),f_{2}(\bm{z}^{*}),f_{3}(\bm{z}^{*}))^T = (P_{p,\text{min}}, P_{p,\text{max}}, P_{p,\text{ave}})^T = (100.96,$ $148.02,$ $ 116.50)^T$ and $\bm\Sigma$ is a diagonal matrix with entries $(5.05, 7.40, 5.83)^T$. The budgeted number of true model solutions is $216$; the fixed surrogate model is evaluated on a $6\times 6\times 6 = 216$ pre-grid while the adaptive surrogate is evaluated with a pre-grid of size $4\times 4\times 4 = 64$ and the other 152 evaluations are adaptively selected. 
The NF architecture and hyper-parameter specifications are the same as for the RC model, except a more frequent surrogate update of $c = 300$ and a larger batch size $b = 500$.

The results are presented in Figure~\ref{fig:rcr_res}. The posterior samples obtained through NoFAS capture well the non-linear correlation among the parameters and generate a fairly accurate posterior predictive distribution that overlaps with the observations but has a slightly larger dispersion, as expected. Additional details can be found in~\cite{wang2022variational}.
%
\begin{figure}[!ht]
\centering
\includegraphics[scale=0.8]{imgs/rcr/log_plot_rcr.pdf}
\includegraphics[scale=0.8]{imgs/rcr/sample_plot_rcr.pdf}
\includegraphics[scale=0.8]{imgs/rcr/target_plot_rcr_0_1.pdf}\\
\includegraphics[scale=0.8]{imgs/rcr/target_plot_rcr_0_2.pdf}
\includegraphics[scale=0.8]{imgs/rcr/target_plot_rcr_1_2.pdf}
\caption{Results from the RCR model. Loss profile (left), posterior predictive distribution (center) and posterior samples (right).}\label{fig:rcr_res}
\end{figure}

% ====================================
\subsection{Friedman 1 model (AdaAnn)}
% ====================================

We consider a modified version of the Friedman 1 dataset~\citep{friedman1991multivariate} to examine the performance of  our adaptive annealing scheduler in a high-dimensional context. 
According to the original model in~\cite{friedman1991multivariate}, the data are generated as
%
\begin{equation}\label{eqn:friedman1}
\textstyle y_i = \mu_i(\boldsymbol{\beta})+ \epsilon_i, \mbox{ where }
\mu_i(\boldsymbol{\beta})=\beta_1\text{sin}(\pi x_{i1}x_{i2})+ \beta_2(x_{i3}-\beta_3)^2+\sum_{j=4}^{10}\beta_jx_{ij}, 
\end{equation}
%
where $\boldsymbol{\beta}=(\beta_1,\ldots,\beta_{10})=(10,20, 0.5, 10, 5, 0, 0, 0, 0, 0)$ and $\epsilon_i\sim\mathcal{N}(0,1)$. 

We made a slight modification to the model in~\eqref{eqn:friedman1} by setting
\begin{equation} \label{eqn:friedman1_modified}
\mu_i(\boldsymbol{\beta}) = \textstyle \beta_1\text{sin}(\pi x_{i1}x_{i2})+ \beta_2^2(x_{i3}-\beta_3)^2+\sum_{j=4}^{10}\beta_jx_{ij},
\end{equation}
%
where $\boldsymbol{\beta}=(\beta_1,\ldots,\beta_{10})=(10,\pm \sqrt{20}, 0.5, 10, 5, 0, 0, 0, 0, 0)$. Note that both~\eqref{eqn:friedman1} and \eqref{eqn:friedman1_modified} contain linear, non-linear, and interaction terms of the input variables $X_1$ to $X_{10}$, five of which ($X_6$ to $X_{10}$) are irrelevant to $Y$. Each $X$ is drawn independently from $\mathcal{U}(0,1)$. We used R package \texttt{tgp} \citep{gramacy2007tgp} to generate a Friedman 1 dataset with a sample size of $n$=1000.
%
We impose a non-informative uniform prior $p(\boldsymbol{\beta})\propto$ and, unlike the original modal, we now expect a bimodal posterior distribution of $\boldsymbol{\beta}$.
%
\begin{table}[!ht]
\caption{Posterior mean and standard deviation of the parameters for bimodal posterior in Example 6.}\label{table:Friedman_bimodal_stats}
\centering
    \begin{tabular}[2in]{lc c cc c c}
    \toprule
    \textbf{True}  && \multicolumn{2}{c}{\textbf{Mode 1}} && \multicolumn{2}{c}{\textbf{Mode 2}} \\
    \cline{3-4}\cline{6-7}\noalign{\vspace{4pt}}
    \textbf{Value} && Post. Mean & Post. SD && Post. Mean & Post. SD\\
    \midrule
    $\beta_1 = 10$   && 9.9865 & 0.0901 && 9.9829 & 0.0920\\
    $\beta_2 = \pm \sqrt{20}$   && 4.5095 & 0.0461 && -4.5070\,\, & 0.0456\\
    $\beta_3 = 0.5$  && 0.4978 & 0.0027 && 0.4978 & 0.0027\\
    $\beta_4 = 10$   && 10.1330\,\,\, & 0.1049 && 10.1255\,\,\, & 0.1051\\
    $\beta_5 = 5$    && 5.0273 & 0.1058 && 5.0289 & 0.1038\\
    $\beta_6 = 0$    && 0.0594 & 0.1043 && 0.0572 & 0.1022\\
    $\beta_7 = 0$    && -0.0419\,\, & 0.1023 && -0.0299\,\, & 0.1024\\
    $\beta_8 = 0$    && -0.0883\,\, & 0.1052 && -0.0827\,\, & 0.1052\\
    $\beta_9 = 0$    && -0.0715\,\, & 0.1055 && -0.0665\,\, & 0.1060\\
    $\beta_{10} = 0$ && 0.0162 & 0.1008 && 0.0104 & 0.1027\\
    \bottomrule
    \end{tabular}
\end{table}

% ==================================
\subsection{NoFAS with AdaAnn (RCR)}
% ==================================

% ==================================
\section{Conclusion and Future Work}\label{sec:conclusions}
% ==================================

LINFA is designed to be extensible through implementation of new child classes for a number of abstract classes. Some interesting direction for future work are listed below.

% Possible extension of the linfa library
% Differential privacy 
\noindent{\bf Differential privacy for variational inference} - Future versions will support user-defined privacy preserving gradient descent algorithms. This will allow to perform inference while limiting the information about the original dataset disclosed to third parties. Additional information can be found in~\cite{su2023differentially}.

% Additional annealing schedulers
\noindent{\bf Additional annealing schedulers} - LINFA is designed to be flexible with respect to the annealing schedulers and provides an interface that is easy to extend to include additional schedulers.

% Dimensionality reduction
\noindent{\bf Dimensionality reduction} - 

% ELBO
\noindent{\bf Flexible definition of the loss function} - The ELBO loss typically used in variational inference has known limitations, some of which are related to its close connection with the KL distance. 

% Acknowledgements should go at the end, before appendices and references
\acks{The authors gratefully acknowledge the support 
by the NSF Big Data Science \& Engineering grant \#1918692
and the computational resources provided through the Center for
Research Computing at the University of Notre Dame. DES also acknowledges support from
NSF CAREER grant \#1942662.}

% Manual newpage inserted to improve layout of sample file - not
% needed in general before appendices/bibliography.
% \newpage

\appendix

% ====================
\section*{Appendix A - Code options}
% ====================

\begin{table}[!ht]
\centering
\caption{General parameters}
\begin{tabular}{p{4cm} p{2cm} p{8cm}} 
\toprule
{\bf Option} & {\bf Type} & {\bf Description}\\
\midrule
\emph{\texttt{name}} & str & experiment name\\
\emph{\texttt{flow\_type}} & str & type of normalizing flow (maf,realnvp)\\ 
\emph{\texttt{n\_blocks}} & int & Number of normalizing flow layers (default 5)\\
\emph{\texttt{hidden\_size}} & int & Number of neurons in MADE hidden layer (default 100)\\
\emph{\texttt{n\_hidden}} & int & Number of hidden layers in MADE (default 1)\\
\emph{\texttt{activation\_fn}} & str & Activation function used (WHERE???) (default: 'relu')\\
\emph{\texttt{input\_order}} & str & Input order for mask creation {\bf EXPLAIN!!!!} (default: 'sequential')\\
\emph{\texttt{batch\_norm\_order}} & bool & Order to decide if batch normalization layer is used (default True)\\
\emph{\texttt{sampling\_interval}} & int & How often to sample from normalizing flow\\
\emph{\texttt{input\_size}} & int & Dimensionality of input (default: 2)\\
\emph{\texttt{batch\_size}} & int & Number of samples from the basic distribution generated at each iteration (default 100)\\
\emph{\texttt{true\_data\_num}} & double & number of true model evaluated (default: 2)\\
\emph{\texttt{n\_iter}} & int & Number of iterations (default 25001)\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!ht]
\centering
\caption{Optimizer and learning rate parameters}
\begin{tabular}{p{4cm} p{2cm} p{8cm}} 
\toprule
{\bf Option} & {\bf Type} & {\bf Description}\\
\midrule
\emph{\texttt{optimizer}} & string & type of optimizer used (default: 'Adam')\\
\emph{\texttt{lr}} & float & Learning rate (default 0.003)\\
\emph{\texttt{lr\_decay}} & float & Learning rate decay (default 0.9999)\\
\emph{\texttt{lr\_scheduler}} & string & type of learning rate scheduler used\\
\emph{\texttt{lr\_step}} & int & Number of steps for learning rate step scheduler\\
\emph{\texttt{log\_interval}} & int & Number of interval between two successive plots of loss summary plot interval (default 10)\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!ht]
\centering
\caption{Output parameters}
\begin{tabular}{p{4cm} p{2cm} p{8cm}} 
\toprule
{\bf Option} & {\bf Type} & {\bf Description}\\
\midrule
\emph{\texttt{output\_dir}} & string & output folder\\
\emph{\texttt{results\_file}} & string & {\bf\color{red}What exactly is writing in the result file??}\\
\emph{\texttt{log\_file}} & string & name of the log file which stores {\bf\color{red} What are we writing in the log file??}\\
\emph{\texttt{samples\_file}} & string & Name of the file where all samples are stored {\bf\color{red} do we keep track of the samples at all iterations? How do we distinguish between iterations??}\\
\emph{\texttt{seed}} & int & Seed for random number generator\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!ht]
\centering
\caption{Surrogate model parameters (NoFAS)}
\begin{tabular}{p{4cm} p{2cm} p{8cm}} 
\toprule
{\bf Option} & {\bf Type} & {\bf Description}\\
\midrule
\emph{\texttt{use\_surrogate}} & bool & decide if the surrogate model is used\\
\emph{\texttt{n\_sample}} & int & Total number of iterations {\bf\color{red}Not clear what this is exactly...}\\
\emph{\texttt{calibrate\_interval}} & int & How often to update surrogate model (default 1000)\\
\emph{\texttt{budget}} & int & Maximum allowable number of true model evaluation\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!ht]
\centering
\caption{Parameters for the adaptive annealing scheduler (AdaAnn)}\label{tab:adaann}
\begin{tabular}{p{4cm} p{2cm} p{8cm}} 
\toprule
{\bf Option} & {\bf Type} & {\bf Description}\\
\midrule
\emph{\texttt{annealing}} & bool & is used to activate an annealing scheduler. If this is \emph{\texttt{False}}, the target posterior distribution is left unchanged during the iterations.\\
\emph{\texttt{scheduler}} & string & defines the type of annealing scheduler. This includes a \emph{\texttt{fixed}} scheduler as well as the \emph{\texttt{AdaAnn}} adaptive scheduler (default 'AdaAnn').\\
\emph{\texttt{tol}} & float & KL tolerance. It is kept constant during inference and used in the numerator of~\eqref{equ:adaann}.\\
\emph{\texttt{t0}} & float & Initial inverse temperature.\\
\emph{\texttt{N}} & int & Number of batch samples during annealing.\\
\emph{\texttt{N\_1}} & int & Number of batch samples at $t=1$.\\
\emph{\texttt{T\_0}} & int & Number of initial parameter updates at initial $t_0$.\\
\emph{\texttt{T}} & int & Number of parameter updates during annealing.\\
\emph{\texttt{T\_1}} & int & Number of parameter updates at $t=1$.\\
\emph{\texttt{M}} & int & Number of Monte Carlo samples used to evaluate the denominator in~\eqref{equ:adaann}.\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!ht]
\centering
\caption{Device parameters}
\begin{tabular}{p{4cm} p{2cm} p{8cm}} 
\toprule
{\bf Option} & {\bf Type} & {\bf Description}\\
\midrule
\emph{\texttt{no\_cuda}} & bool & Do not use GPU acceleration\\
\bottomrule
\end{tabular}
\end{table}

\vskip 0.2in
\bibliography{linfa}

\end{document}
