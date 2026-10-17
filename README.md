# MLP Regression: MLE vs MAP

This project studies a two-hidden-layer ReLU MLP on a synthetic nonlinear regression task. It compares ordinary maximum-likelihood training with L2-regularized maximum-a-posteriori (MAP) training.

## Contents
- `notebook.ipynb`: experiments, plots, learning curves, regularization sensitivity, and convergence.
- `src.py`: shared synthetic-data utilities and the MLP model definition.

## Setup
Install Python 3.10+ and the dependencies:

```bash
pip install numpy matplotlib seaborn tensorflow jupyter
```

Open the notebook from this directory so that `from src import ...` works:

```bash
jupyter notebook notebook.ipynb
```

Run all cells from top to bottom. The experiments use fixed random seeds where appropriate for reproducibility.

## Methodology

### 1. Synthetic Data Generation

The synthetic regression dataset is generated according to

\[
y_i=f^*(x_i)+\epsilon_i,
\]

where the underlying function is

\[
f^*(x)=\sin(2\pi x)+0.3x.
\]

The input samples are drawn from a uniform distribution:

\[
x_i\sim\mathcal{U}(0,1),
\]

and the observation noise follows a Gaussian distribution:

\[
\epsilon_i\sim\mathcal{N}(0,\sigma^2).
\]

### 2. Neural Network Architecture

The regression model is a multilayer perceptron (MLP) with two hidden layers using ReLU activations and a single linear output neuron.

For an input \(x\), the network computes

\[
\begin{aligned}
h_1 &= \operatorname{ReLU}(W_1x+b_1),\\
h_2 &= \operatorname{ReLU}(W_2h_1+b_2),\\
\hat y &= W_3h_2+b_3.
\end{aligned}
\]

The hidden-layer width is selected according to the experiment.

### 3. Maximum Likelihood Estimation (MLE)

Assuming conditionally independent Gaussian observation noise,

\[
y_i\mid x_i,\theta
\sim
\mathcal{N}(f_\theta(x_i),\sigma^2),
\]

the mean negative log-likelihood is

\[
\mathcal{L}_{\mathrm{NLL}}(\theta)
=
\frac{1}{N}\sum_{i=1}^{N}
\left[
\frac{1}{2}\log(2\pi\sigma^2)
+
\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}
\right].
\]

MLE estimates the network parameters by minimizing this objective:

\[
\theta_{\mathrm{MLE}}
=
\arg\min_\theta
\mathcal{L}_{\mathrm{NLL}}(\theta).
\]

For fixed \(\sigma^2\), minimizing the Gaussian negative log-likelihood is equivalent to minimizing the mean squared error.

### 4. Maximum A Posteriori Estimation (MAP)

MAP estimation introduces a zero-mean Gaussian prior on the network weights:

\[
W_j\sim\mathcal{N}(0,\tau^2I).
\]

Ignoring constants independent of the parameters, the resulting objective is

\[
\mathcal{L}_{\mathrm{MAP}}
=
\mathcal{L}_{\mathrm{NLL}}
+
\frac{1}{2N\tau^2}
\sum_j\|W_j\|_2^2.
\]

Thus, MAP estimation is implemented by adding L2 regularization to the mean Gaussian negative log-likelihood. In the implementation, the regularizer is applied to layer kernels (weights), not biases.

The coefficient used in the experiment is

\[
\lambda_{\mathrm{MAP}}=\frac{1}{2N\tau^2},
\]

with \(\tau=1\).

### 5. Evaluation and Regularization Study

Mean squared error (MSE) is used to evaluate prediction accuracy:

\[
\operatorname{MSE}
=
\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2.
\]

A separate regularization experiment compares training and test MSE across several L2 coefficients, including the coefficient associated with the MAP formulation. The MSE used for evaluation measures prediction error only and does not include the regularization penalty.
