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

The dataset is generated according to

$$
y_i = f^*(x_i) + \epsilon_i
$$

where the underlying function is

$$
f^*(x) = \sin(2\pi x) + 0.3x
$$

The input samples and observation noise are drawn from

$$
x_i \sim \mathcal{U}(0,1),
\qquad
\epsilon_i \sim \mathcal{N}(0,\sigma^2)
$$

### 2. Neural Network Architecture

The MLP consists of two hidden layers with ReLU activation and one linear output neuron.

$$
h_1 = \operatorname{ReLU}(W_1x+b_1)
$$

$$
h_2 = \operatorname{ReLU}(W_2h_1+b_2)
$$

$$
\hat{y} = W_3h_2+b_3
$$

### 3. Maximum Likelihood Estimation (MLE)

Assuming Gaussian observation noise,

$$
y_i \mid x_i,\theta
\sim
\mathcal{N}(f_\theta(x_i),\sigma^2)
$$

the mean negative log-likelihood is

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[
\frac{1}{2}\log(2\pi\sigma^2)
+
\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}
\right]
$$

MLE estimates the parameters by minimizing this objective:

$$
\theta_{\mathrm{MLE}}
=
\arg\min_{\theta}
\mathcal{L}_{\mathrm{NLL}}(\theta)
$$

For fixed noise variance, minimizing Gaussian NLL is equivalent to minimizing MSE.

### 4. Maximum A Posteriori Estimation (MAP)

A zero-mean Gaussian prior is imposed on the network weights:

$$
W_j \sim \mathcal{N}(0,\tau^2 I)
$$

The resulting MAP objective is

$$
\mathcal{L}_{\mathrm{MAP}}
=
\mathcal{L}_{\mathrm{NLL}}
+
\frac{1}{2N\tau^2}
\sum_j \|W_j\|_2^2
$$

Therefore, the regularization coefficient is

$$
\lambda_{\mathrm{MAP}}
=
\frac{1}{2N\tau^2}
$$

In the implementation, $\tau=1$, and L2 regularization is applied to layer weights (kernels), not biases.

### 5. Evaluation and Regularization Study

Prediction performance is evaluated using mean squared error:

$$
\operatorname{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

A separate experiment compares training and test MSE across several L2 regularization coefficients, including the MAP-associated coefficient. The reported MSE measures prediction error only; it does not include the regularization penalty.
