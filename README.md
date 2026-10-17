# MLP Regression: MLE vs. MAP

In this project, I explore how a Multi-Layer Perceptron (MLP) can learn nonlinear patterns from data. I use a simple synthetic dataset to experiment with different training objectives and see how regularization affects the model's predictions.

## What I did

- Generated synthetic data from a nonlinear function with Gaussian noise.
- Built an MLP with two hidden layers and ReLU activation functions.
- Trained the model using a Gaussian negative log-likelihood loss.
- Explored MAP estimation by adding L2 regularization to the model's weights.
- Compared different regularization strengths using mean squared error (MSE).

## MLE and MAP

For the MLE experiment, I used a Gaussian likelihood to model the noise in the observations. The model learns by minimizing the negative log-likelihood.

For the MAP experiment, I added an L2 penalty to the weights. This corresponds to assuming a Gaussian prior over the model parameters and encourages smaller weight values.

I also experimented with different regularization strengths to see how they affect the model's performance on training and test data.

## Tools

Python, NumPy, Matplotlib, TensorFlow, and Jupyter Notebook.

## Running the project

Open `notebook.ipynb` in Jupyter Notebook or VS Code and run the cells in order to reproduce the experiments and visualizations.
