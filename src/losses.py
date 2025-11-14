"""
Fonctions de coût (loss functions) pour l'entraînement
"""

import numpy as np


def cross_entropy(Y_true, Y_pred, epsilon=1e-7):
    """
    Cross-Entropy Loss pour classification multi-classes

    Args:
        Y_true: vrais labels (one-hot encoded) - shape: (n, classes)
        Y_pred: prédictions (probabilités) - shape: (n, classes)
        epsilon: valeur pour éviter log(0)

    Returns:
        loss: erreur moyenne
    """
    n = Y_true.shape[0]
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(Y_true * np.log(Y_pred)) / n
    return loss


def binary_cross_entropy(Y_true, Y_pred, epsilon=1e-7):
    """
    Binary Cross-Entropy pour classification binaire

    Args:
        Y_true: vrais labels (0 ou 1) - shape: (n, 1)
        Y_pred: prédictions (probabilités) - shape: (n, 1)

    Returns:
        loss: erreur moyenne
    """
    n = Y_true.shape[0]
    Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred)) / n
    return loss


def mse(Y_true, Y_pred):
    """
    Mean Squared Error

    Args:
        Y_true: vraies valeurs - shape: (n, d)
        Y_pred: prédictions - shape: (n, d)

    Returns:
        loss: erreur quadratique moyenne
    """
    n = Y_true.shape[0]
    loss = np.sum((Y_true - Y_pred) ** 2) / n
    return loss


def mae(Y_true, Y_pred):
    """
    Mean Absolute Error

    Args:
        Y_true: vraies valeurs - shape: (n, d)
        Y_pred: prédictions - shape: (n, d)

    Returns:
        loss: erreur absolue moyenne
    """
    n = Y_true.shape[0]
    loss = np.sum(np.abs(Y_true - Y_pred)) / n
    return loss


def huber_loss(Y_true, Y_pred, delta=1.0):
    """
    Huber Loss (robuste aux outliers)

    Args:
        Y_true: vraies valeurs
        Y_pred: prédictions
        delta: seuil

    Returns:
        loss: erreur de Huber
    """
    error = Y_true - Y_pred
    is_small_error = np.abs(error) <= delta

    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)

    loss = np.where(is_small_error, squared_loss, linear_loss)
    return np.mean(loss)


# Dérivées pour backpropagation
def cross_entropy_derivative_softmax(Y_true, Y_pred):
    """
    Dérivée de cross-entropy + softmax (simplifié)

    Pour softmax + cross-entropy, la dérivée est simplement:
    L/Z = Y_pred - Y_true
    """
    return Y_pred - Y_true


def mse_derivative(Y_true, Y_pred):
    """
    Dérivée de MSE

    L/Y_pred = 2(Y_pred - Y_true) / n
    """
    n = Y_true.shape[0]
    return 2 * (Y_pred - Y_true) / n


# Dictionnaire des loss functions
LOSSES = {
    'cross_entropy': cross_entropy,
    'binary_cross_entropy': binary_cross_entropy,
    'mse': mse,
    'mae': mae,
    'huber': huber_loss
}


def get_loss(name):
    """Retourne la fonction de loss"""
    return LOSSES.get(name, cross_entropy)
