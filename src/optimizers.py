"""
Optimiseurs pour l'entraînement des réseaux de neurones
"""

import numpy as np


class Optimizer:
    """Classe de base pour les optimiseurs"""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """Met à jour les paramètres"""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent"""

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads['d' + key]
        return params


class SGDMomentum(Optimizer):
    """SGD avec Momentum"""

    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        if not self.velocity:
            for key in params.keys():
                self.velocity[key] = np.zeros_like(params[key])

        for key in params.keys():
            self.velocity[key] = self.momentum * self.velocity[key] + grads['d' + key]
            params[key] -= self.learning_rate * self.velocity[key]

        return params


class Adam(Optimizer):
    """Adam Optimizer"""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        if not self.m:
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.t += 1

        for key in params.keys():
            grad = grads['d' + key]

            # Mise à jour des moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # Correction du biais
            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)

            # Mise à jour des paramètres
            params[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

        return params


class RMSprop(Optimizer):
    """RMSprop Optimizer"""

    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads):
        if not self.cache:
            for key in params.keys():
                self.cache[key] = np.zeros_like(params[key])

        for key in params.keys():
            grad = grads['d' + key]
            self.cache[key] = self.beta * self.cache[key] + (1 - self.beta) * (grad ** 2)
            params[key] -= self.learning_rate * grad / (np.sqrt(self.cache[key]) + self.epsilon)

        return params
