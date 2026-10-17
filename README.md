# MLP Regression with MLE and MAP

This project studies nonlinear regression with a Multi-Layer Perceptron (MLP). It compares training with a Gaussian negative log-likelihood (MLE setting) and with an L2-regularized objective motivated by Maximum a Posteriori (MAP) estimation. It also explores how different L2 regularization strengths affect regression performance.

## Project overview

The experiments use synthetic one-dimensional data generated from a known nonlinear function with additive Gaussian noise:

$$
x_i \sim \mathcal{U}(0,1)
$$

$$
y_i = f^*(x_i) + \epsilon_i,\qquad \epsilon_i \sim \mathcal{N}(0,\sigma_{\text{noise}}^2)
$$

The underlying function is

$$
f^*(x) = \sin(2\pi x) + 0.3x.
$$

The dataset is generated with 75 samples and a noise standard deviation of 0.12 in the notebook.

## Model architecture

The regression model is a feed-forward MLP with two hidden Dense layers using ReLU activations and a single linear output unit:

$$
h_1 = \mathrm{ReLU}(W_1x+b_1)
$$

$$
h_2 = \mathrm{ReLU}(W_2h_1+b_2)
$$

$$
\hat{y} = W_3h_2+b_3
$$

Here, $W_j$ and $b_j$ denote the weights and biases of layer $j$. The L2 regularizer is applied to the Dense-layer kernels (weights), not to the biases.

## Maximum Likelihood Estimation (MLE)

Assume the target is conditionally Gaussian given the input:

$$
y_i \mid x_i,\theta \sim \mathcal{N}(f_\theta(x_i),\sigma^2)
$$

The mean negative log-likelihood is

$$
\mathcal{L}_{\mathrm{NLL}}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\left[\frac{1}{2}\log(2\pi\sigma^2)+\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right].
$$

Maximum likelihood estimation chooses the model parameters that minimize this objective:

$$
\theta_{\mathrm{MLE}} = \arg\min_{\theta}\mathcal{L}_{\mathrm{NLL}}(\theta).
$$

The notebook implements this Gaussian negative log-likelihood as a custom TensorFlow loss. The observation-noise standard deviation used in this experiment is $\sigma=0.2$.

## Maximum a Posteriori (MAP) estimation

MAP estimation combines the data likelihood with a prior over the model weights. With an independent zero-mean Gaussian prior on the weights,

$$
W_j \sim \mathcal{N}(0,\tau^2 I),
$$

minimizing the negative log-posterior gives the negative log-likelihood plus an L2 penalty on the weights. With the mean-NLL convention used here, the objective is

$$
\mathcal{L}_{\mathrm{MAP}}(\theta) = \mathcal{L}_{\mathrm{NLL}}(\theta) + \frac{1}{2N\tau^2}\sum_j \|W_j\|_2^2.
$$

Therefore, the corresponding L2 coefficient is

$$
\lambda_{\mathrm{MAP}}=\frac{1}{2N\tau^2}.
$$

The notebook uses $\tau=1.0$ and computes this coefficient from the number of training examples. The regularization term applies to the layer weights, consistent with the model's kernel regularizers.

**Important:** the loss recorded during MAP training includes both the data-fit term and the regularization penalty. It is not the same quantity as the unregularized MSE.

## MSE and regularization-strength experiment

Mean squared error (MSE) is also used to evaluate regression predictions:

$$
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2.
$$

The notebook compares MSE-based training and evaluates several L2 coefficients, including zero regularization and the coefficient derived from the MAP prior. The sweep is an experiment over regularization strengths; only the coefficient calculated from $\lambda_{\mathrm{MAP}}=1/(2N\tau^2)$ has the stated MAP-prior interpretation under the mean-NLL objective.

## Repository contents

- `notebook.ipynb` — data generation, model definition, training experiments, and visualizations.
- `src.py` — Python source code for the project.

## Requirements

The project uses Python with NumPy, Matplotlib, and TensorFlow. Install the dependencies in your Python environment:

```bash
pip install numpy matplotlib tensorflow
```

If you maintain a `requirements.txt` file, install from it instead:

```bash
pip install -r requirements.txt
```

## Running the project

Open `notebook.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run the cells in order.

## Scope

This is an experimental study on synthetic one-dimensional regression data. The purpose is to compare likelihood-based training, prior-motivated L2 regularization, and the effect of regularization strength. Results depend on the random sample, model initialization, and training configuration.
