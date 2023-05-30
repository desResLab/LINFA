Introduction
************

Generating samples from a posterior distribution is a fundamental task in Bayesian inference.

Development of sampling-based algorithms of the Markov chain Monte Carlo family in the mid-eighties~\citep{geman1984stochastic} have made such task accessible for the solution of Bayesian inverse problems to a wide audience in a wide range of research fields. 

However, the number of samples required by these approaches becomes significant and, even though a number of metrics have been proposed over the years, the convergence of Markov chains to their stationary posterior distribution is not always easy to quantify.

More recent paradigms have been proposed in the context of variational inference~\cite{wainwright2008graphical}, where an optimization problem is formulated to determine the optimal member of a parametric family of distributions that can approximate a target posterior.

In addition, flexible approaches to parametrize variational distributions through a composition of transformations rooted in optimal transport theory~\cite{villani2009optimal} have reached popularity under the name of \emph{normalizing flows}~\cite{kobyzev2020normalizing,papamakarios2021normalizing}. 

The combination of variational inference and normalizing flow transformation has received significant recent interest in the context of general algorithm for the solution of inverse problems~\cite{rezende2015variational}.

However, is it often the case in practice that the underlying statistical model is computationally expensive to solve, for example when it requires the solution of an ordinary or partial differential equation. In such cases, inference can easily become intractable. Additionally, strong and nonlinear dependence between model parameters may results in difficult-to-sample posterior distributions characterized by features at multiple scales or by multiple modes. 

The LINFA library is specifically designed for cases where the model evaluation is computationally expensive. In such cases, the construction of an adaptively trained surrogate model is key to reduce the computational cost of inference. 
% Difficult to sample posterior distribution
In addition, LINFA provides recently developed adaptive annealing schedulers, with automatically assigned temperature increments.

This paper is organized as follows. The theoretical background is summarized in Section~\ref{sec:background} followed by a list of the capabilities in Section~\ref{sec:capabilities}. Numerical tests are described in Section~\ref{sec:benchmarks}, where each test is specifically designed to test one or more components of the library. Conclusions and future work are finally discussed in~\ref{sec:conclusions}.

.. bibliography:: references.bib
