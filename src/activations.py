"""
Fonctions d'activation pour les réseaux de neurones

Implémente les activations classiques avec leurs dérivées
"""

import numpy as np


def sigmoid(Z):
    """
    Sigmoid: Ã(x) = 1 / (1 + e^(-x))
    Output: [0, 1]
    """
    return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))


def sigmoid_derivative(Z, A=None):
    """Dérivée de sigmoid: Ã'(x) = Ã(x)(1 - Ã(x))"""
    if A is None:
        A = sigmoid(Z)
    return A * (1 - A)


def tanh(Z):
    """
    Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Output: [-1, 1]
    """
    return np.tanh(Z)


def tanh_derivative(Z, A=None):
    """Dérivée de tanh: tanh'(x) = 1 - tanh²(x)"""
    if A is None:
        A = tanh(Z)
    return 1 - A ** 2


def relu(Z):
    """
    ReLU: f(x) = max(0, x)
    Output: [0, )
    """
    return np.maximum(0, Z)


def relu_derivative(Z):
    """Dérivée de ReLU: 1 si x > 0, 0 sinon"""
    return (Z > 0).astype(float)


def leaky_relu(Z, alpha=0.01):
    """
    Leaky ReLU: f(x) = max(±x, x)
    Output: (-, )
    """
    return np.where(Z > 0, Z, alpha * Z)


def leaky_relu_derivative(Z, alpha=0.01):
    """Dérivée de Leaky ReLU"""
    return np.where(Z > 0, 1, alpha)


def softmax(Z):
    """
    Softmax: transforme scores en probabilités

    Output: [0, 1] avec somme = 1
    """
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def linear(Z):
    """Activation linéaire (identité)"""
    return Z


def linear_derivative(Z):
    """Dérivée de l'identité = 1"""
    return np.ones_like(Z)


# Dictionnaire des activations
ACTIVATIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'softmax': softmax,
    'linear': linear,
    None: linear
}

DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'tanh': tanh_derivative,
    'relu': relu_derivative,
    'leaky_relu': leaky_relu_derivative,
    'linear': linear_derivative,
    None: linear_derivative
}


def get_activation(name):
    """Retourne la fonction d'activation"""
    return ACTIVATIONS.get(name, linear)


def get_derivative(name):
    """Retourne la dérivée de l'activation"""
    return DERIVATIVES.get(name, linear_derivative)
