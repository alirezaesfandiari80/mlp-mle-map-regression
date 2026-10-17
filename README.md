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

## Methodology note
For Gaussian observations with known standard deviation `sigma` and an independent zero-mean Gaussian prior with variance `tau**2` on the network kernels, the average negative log-posterior is the mean Gaussian negative log-likelihood plus an L2 penalty with coefficient `1 / (2 * n * tau**2)`. Where the notebook uses mean squared error instead of Gaussian NLL, the equivalent scaled coefficient is `sigma**2 / (n * tau**2)`. Bias parameters are not L2-regularized in this implementation.
