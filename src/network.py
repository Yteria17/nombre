"""
Classe NeuralNetwork complète pour la classification

Architecture modulaire et extensible
"""

import numpy as np
import pickle
from pathlib import Path


class NeuralNetwork:
    """
    Réseau de neurones multi-couches from scratch

    Supporte:
    - Architecture flexible
    - Différents optimiseurs
    - Dropout
    - Sauvegarde/chargement
    """

    def __init__(self, layer_dims, learning_rate=0.01, optimizer='sgd'):
        """
        Initialise le réseau

        Args:
            layer_dims: liste des dimensions [input, hidden1, hidden2, ..., output]
            learning_rate: taux d'apprentissage
            optimizer: 'sgd', 'momentum', 'adam'
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.parameters = self._initialize_parameters()
        self.history = {'loss': [], 'train_acc': [], 'val_acc': []}

        # Initialiser l'optimiseur
        self._init_optimizer()

    def _initialize_parameters(self):
        """Initialise les poids avec He initialization"""
        np.random.seed(42)
        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            # He initialization pour ReLU
            parameters[f'W{l}'] = np.random.randn(
                self.layer_dims[l-1], self.layer_dims[l]
            ) * np.sqrt(2.0 / self.layer_dims[l-1])

            parameters[f'b{l}'] = np.zeros((1, self.layer_dims[l]))

        return parameters

    def _init_optimizer(self):
        """Initialise les paramètres de l'optimiseur"""
        if self.optimizer_name == 'momentum':
            self.velocity = {}
            for key in self.parameters.keys():
                self.velocity[key] = np.zeros_like(self.parameters[key])
            self.momentum = 0.9

        elif self.optimizer_name == 'adam':
            self.m = {}
            self.v = {}
            for key in self.parameters.keys():
                self.m[key] = np.zeros_like(self.parameters[key])
                self.v[key] = np.zeros_like(self.parameters[key])
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0

    def relu(self, Z):
        """ReLU activation"""
        return np.maximum(0, Z)

    def softmax(self, Z):
        """Softmax activation"""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation

        Args:
            X: données d'entrée (n_samples, n_features)

        Returns:
            A_final: prédictions (probabilités)
            cache: valeurs intermédiaires
        """
        cache = {'A0': X}
        A = X

        # Couches cachées (toutes sauf la dernière)
        for l in range(1, len(self.layer_dims) - 1):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            Z = np.dot(A, W) + b
            A = self.relu(Z)

            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        # Couche de sortie
        l = len(self.layer_dims) - 1
        W = self.parameters[f'W{l}']
        b = self.parameters[f'b{l}']

        Z = np.dot(A, W) + b
        A = self.softmax(Z)

        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A

        return A, cache

    def compute_loss(self, Y_true, Y_pred):
        """Cross-entropy loss"""
        n_samples = Y_true.shape[0]
        epsilon = 1e-7
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(Y_true * np.log(Y_pred)) / n_samples
        return loss

    def backward(self, Y, cache):
        """
        Backpropagation

        Args:
            Y: vrais labels (one-hot)
            cache: valeurs de la forward pass

        Returns:
            gradients: dictionnaire des gradients
        """
        gradients = {}
        n_samples = Y.shape[0]
        L = len(self.layer_dims) - 1

        # Gradient de la couche de sortie (softmax + cross-entropy)
        dZ = cache[f'A{L}'] - Y

        # Backprop couche par couche
        for l in reversed(range(1, L + 1)):
            A_prev = cache[f'A{l-1}']

            # Gradients des paramètres
            gradients[f'dW{l}'] = np.dot(A_prev.T, dZ) / n_samples
            gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / n_samples

            # Gradient pour la couche précédente
            if l > 1:
                dA_prev = np.dot(dZ, self.parameters[f'W{l}'].T)
                # Dérivée de ReLU
                dZ = dA_prev * (cache[f'Z{l-1}'] > 0)

        return gradients

    def update_parameters(self, gradients):
        """
        Met à jour les paramètres selon l'optimiseur choisi
        """
        if self.optimizer_name == 'sgd':
            self._update_sgd(gradients)
        elif self.optimizer_name == 'momentum':
            self._update_momentum(gradients)
        elif self.optimizer_name == 'adam':
            self._update_adam(gradients)

    def _update_sgd(self, gradients):
        """SGD basique"""
        for key in self.parameters.keys():
            self.parameters[key] -= self.learning_rate * gradients['d' + key]

    def _update_momentum(self, gradients):
        """SGD avec momentum"""
        for key in self.parameters.keys():
            self.velocity[key] = self.momentum * self.velocity[key] + gradients['d' + key]
            self.parameters[key] -= self.learning_rate * self.velocity[key]

    def _update_adam(self, gradients):
        """Adam optimizer"""
        self.t += 1

        for key in self.parameters.keys():
            grad = gradients['d' + key]

            # Mise à jour des moments
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # Correction du biais
            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)

            # Mise à jour
            self.parameters[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

    def predict(self, X):
        """
        Fait des prédictions

        Returns:
            predictions: labels prédits
        """
        A, _ = self.forward(X)
        return np.argmax(A, axis=1)

    def accuracy(self, X, y):
        """Calcule l'accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def one_hot_encode(self, y, n_classes=10):
        """One-hot encoding"""
        one_hot = np.zeros((y.shape[0], n_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128, verbose=True):
        """
        Entraîne le réseau

        Args:
            X_train, y_train: données d'entraînement
            X_val, y_val: données de validation
            epochs: nombre d'époques
            batch_size: taille des mini-batches
            verbose: afficher les détails
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size

        # One-hot encode labels
        Y_train = self.one_hot_encode(y_train)

        if verbose:
            print("\n" + "="*70)
            print("<“ DÉBUT DE L'ENTRAÎNEMENT")
            print("="*70)
            print(f"\nConfiguration:")
            print(f"  " Architecture: {' ’ '.join(map(str, self.layer_dims))}")
            print(f"  " Learning rate: {self.learning_rate}")
            print(f"  " Optimizer: {self.optimizer_name}")
            print(f"  " Batch size: {batch_size}")
            print(f"  " Époques: {epochs}")
            print(f"  " Exemples: {n_samples:,}")
            print("\n" + "="*70 + "\n")

        for epoch in range(epochs):
            # Mélanger les données
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            epoch_loss = 0

            # Mini-batch training
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                # Forward
                A, cache = self.forward(X_batch)

                # Loss
                loss = self.compute_loss(Y_batch, A)
                epoch_loss += loss

                # Backward
                gradients = self.backward(Y_batch, cache)

                # Update
                self.update_parameters(gradients)

            # Métriques
            avg_loss = epoch_loss / n_batches
            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)

            # Sauvegarder
            self.history['loss'].append(avg_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            if verbose:
                print(f"Epoch {epoch+1:2d}/{epochs} - "
                      f"Loss: {avg_loss:.4f} - "
                      f"Train Acc: {train_acc:.4f} - "
                      f"Val Acc: {val_acc:.4f}")

        if verbose:
            print("\n" + "="*70)
            print(" ENTRAÎNEMENT TERMINÉ")
            print("="*70)

    def save(self, filepath):
        """Sauvegarde le modèle"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'parameters': self.parameters,
                'layer_dims': self.layer_dims,
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer_name,
                'history': self.history
            }, f)

        print(f"\n=¾ Modèle sauvegardé: {filepath}")

    @staticmethod
    def load(filepath):
        """Charge un modèle sauvegardé"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        model = NeuralNetwork(
            data['layer_dims'],
            data['learning_rate'],
            data.get('optimizer', 'sgd')
        )
        model.parameters = data['parameters']
        model.history = data.get('history', {'loss': [], 'train_acc': [], 'val_acc': []})

        print(f"\n=Â Modèle chargé: {filepath}")
        return model
