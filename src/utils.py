"""
Utilitaires pour le chargement et la préparation des données MNIST
"""

import numpy as np
import gzip
import os
from urllib import request
import pickle


def download_mnist(data_dir='data'):
    """
    Télécharge le dataset MNIST depuis le serveur officiel de Yann LeCun

    Args:
        data_dir (str): Répertoire où sauvegarder les fichiers
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    # Créer le répertoire s'il n'existe pas
    os.makedirs(data_dir, exist_ok=True)

    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Téléchargement de {filename}...")
            url = base_url + filename
            request.urlretrieve(url, filepath)
            print(f" {filename} téléchargé")
        else:
            print(f" {filename} déjà présent")


def load_mnist_images(filepath):
    """
    Charge les images MNIST depuis un fichier .gz

    Args:
        filepath (str): Chemin vers le fichier .gz

    Returns:
        np.ndarray: Array d'images de forme (n_samples, 28, 28)
    """
    with gzip.open(filepath, 'rb') as f:
        # Les 16 premiers octets sont des métadonnées
        # magic number (4 bytes), nombre d'images (4 bytes),
        # hauteur (4 bytes), largeur (4 bytes)
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    # Reshape en (n_images, 28, 28)
    return data.reshape(-1, 28, 28)


def load_mnist_labels(filepath):
    """
    Charge les labels MNIST depuis un fichier .gz

    Args:
        filepath (str): Chemin vers le fichier .gz

    Returns:
        np.ndarray: Array de labels de forme (n_samples,)
    """
    with gzip.open(filepath, 'rb') as f:
        # Les 8 premiers octets sont des métadonnées
        # magic number (4 bytes), nombre de labels (4 bytes)
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data


def load_mnist(data_dir='data', flatten=False, normalize=True):
    """
    Charge le dataset MNIST complet (train + test)

    Args:
        data_dir (str): Répertoire contenant les fichiers MNIST
        flatten (bool): Si True, aplatit les images (28x28) en vecteurs (784,)
        normalize (bool): Si True, normalise les pixels de [0, 255] à [0, 1]

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            - X_train: images d'entraînement (60000, 28, 28) ou (60000, 784)
            - y_train: labels d'entraînement (60000,)
            - X_test: images de test (10000, 28, 28) ou (10000, 784)
            - y_test: labels de test (10000,)
    """
    # Télécharger si nécessaire
    download_mnist(data_dir)

    # Charger les données
    print("\nChargement des données MNIST...")
    X_train = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    X_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))

    print(f" Données chargées: {len(X_train)} images d'entraînement, {len(X_test)} images de test")

    # Normalisation
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    # Aplatissement
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)  # (60000, 784)
        X_test = X_test.reshape(X_test.shape[0], -1)      # (10000, 784)

    return X_train, y_train, X_test, y_test


def one_hot_encode(labels, num_classes=10):
    """
    Convertit les labels en encodage one-hot

    Exemple: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    Args:
        labels (np.ndarray): Array de labels (n_samples,)
        num_classes (int): Nombre de classes

    Returns:
        np.ndarray: Array one-hot de forme (n_samples, num_classes)
    """
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot


def create_batches(X, y, batch_size=32, shuffle=True):
    """
    Crée des mini-batches pour l'entraînement

    Args:
        X (np.ndarray): Données d'entrée
        y (np.ndarray): Labels
        batch_size (int): Taille des batches
        shuffle (bool): Mélanger les données

    Yields:
        tuple: (X_batch, y_batch)
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]

        yield X[batch_indices], y[batch_indices]


def save_model(model, filepath):
    """
    Sauvegarde un modèle entraîné

    Args:
        model: Le modèle à sauvegarder (doit avoir une méthode get_params())
        filepath (str): Chemin du fichier de sauvegarde
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f" Modèle sauvegardé dans {filepath}")


def load_model(filepath):
    """
    Charge un modèle sauvegardé

    Args:
        filepath (str): Chemin du fichier

    Returns:
        Le modèle chargé
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    print(f" Modèle chargé depuis {filepath}")
    return model


def get_data_stats(X, y):
    """
    Affiche des statistiques sur le dataset

    Args:
        X (np.ndarray): Images
        y (np.ndarray): Labels
    """
    print(f"\n{'='*50}")
    print(f"STATISTIQUES DU DATASET")
    print(f"{'='*50}")
    print(f"Forme des données: {X.shape}")
    print(f"Forme des labels: {y.shape}")
    print(f"Type des données: {X.dtype}")
    print(f"Valeur min: {X.min():.4f}, max: {X.max():.4f}")
    print(f"\nDistribution des classes:")

    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"  Classe {label}: {count:5d} images ({percentage:.1f}%)")

    print(f"{'='*50}\n")
