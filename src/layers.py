"""
Couches (layers) pour les réseaux de neurones
"""

import numpy as np


class Layer:
    """Classe de base pour toutes les couches"""

    def __init__(self):
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, X, training=True):
        """Forward pass"""
        raise NotImplementedError

    def backward(self, dout):
        """Backward pass"""
        raise NotImplementedError


class Dense(Layer):
    """Couche Dense (Fully Connected)"""

    def __init__(self, input_size, output_size, activation=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # He initialization
        self.params['W'] = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.params['b'] = np.zeros((1, output_size))

    def forward(self, X, training=True):
        self.cache['X'] = X
        Z = np.dot(X, self.params['W']) + self.params['b']
        self.cache['Z'] = Z

        if self.activation == 'relu':
            out = np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            out = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif self.activation == 'softmax':
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            out = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:
            out = Z

        self.cache['A'] = out
        return out

    def backward(self, dout):
        X = self.cache['X']
        Z = self.cache['Z']
        A = self.cache['A']
        m = X.shape[0]

        if self.activation == 'relu':
            dZ = dout * (Z > 0)
        elif self.activation == 'sigmoid':
            dZ = dout * A * (1 - A)
        else:
            dZ = dout

        self.grads['W'] = np.dot(X.T, dZ) / m
        self.grads['b'] = np.sum(dZ, axis=0, keepdims=True) / m
        dX = np.dot(dZ, self.params['W'].T)

        return dX


class Dropout(Layer):
    """Dropout pour régularisation"""

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, X, training=True):
        if training:
            self.cache['mask'] = (np.random.rand(*X.shape) > self.dropout_rate)
            out = X * self.cache['mask'] / (1 - self.dropout_rate)
        else:
            out = X
        return out

    def backward(self, dout):
        mask = self.cache['mask']
        return dout * mask / (1 - self.dropout_rate)
