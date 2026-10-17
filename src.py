"""Utilities and model definition for the MLP regression experiments."""

import numpy as np
import tensorflow as tf


def true_func(x):
    """Underlying noiseless function used to generate synthetic targets."""
    x = np.asarray(x)
    return np.sin(2 * np.pi * x) + 0.3 * x


def generate_data(n_sample, noise_std, seed=None):
    """Generate uniform inputs and Gaussian-noisy observations."""
    if n_sample <= 0:
        raise ValueError("n_sample must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_sample, 1))
    epsilon = rng.normal(loc=0.0, scale=noise_std, size=(n_sample, 1))
    y = true_func(X) + epsilon
    return X, y


class MLPRegression(tf.keras.Model):
    """Two-hidden-layer ReLU MLP with optional L2 kernel regularization."""

    def __init__(self, hid_dim=64, l2_lambda=0.0):
        super().__init__()
        regularizer = (
            tf.keras.regularizers.L2(l2_lambda)
            if l2_lambda > 0 else None
        )
        self.hidden_layer1 = tf.keras.layers.Dense(
            hid_dim, activation="relu", kernel_regularizer=regularizer
        )
        self.hidden_layer2 = tf.keras.layers.Dense(
            hid_dim, activation="relu", kernel_regularizer=regularizer
        )
        self.output_layer = tf.keras.layers.Dense(
            1, kernel_regularizer=regularizer
        )

    def call(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        return self.output_layer(x)
