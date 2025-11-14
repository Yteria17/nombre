"""
Fonctions de visualisation pour l'analyse des réseaux de neurones

Fournit des outils pour visualiser:
- Courbes d'apprentissage (loss, accuracy)
- Matrice de confusion
- Prédictions sur échantillons
- Poids des neurones
- Exemples mal classifiés
- Distribution des classes
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_training_history(history, figsize=(14, 5), save_path=None):
    """
    Affiche les courbes d'apprentissage (loss et accuracy)

    Args:
        history: dictionnaire {'loss': [...], 'train_acc': [...], 'val_acc': [...]}
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history['loss']) + 1)

    # Loss
    ax1.plot(epochs, history['loss'], 'b-o', linewidth=2, markersize=4, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Evolution de la Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'g-o', linewidth=2, markersize=4, label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-o', linewidth=2, markersize=4, label='Val Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Evolution de l\'Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Graphique sauvegardé: {save_path}")

    plt.show()


def plot_confusion_matrix(cm, class_names=None, figsize=(10, 8), save_path=None):
    """
    Affiche la matrice de confusion avec heatmap

    Args:
        cm: matrice de confusion (numpy array)
        class_names: noms des classes
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Normaliser pour pourcentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Matrice absolue
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Nombre de prédictions'}, ax=ax1)
    ax1.set_xlabel('Prédiction', fontsize=12)
    ax1.set_ylabel('Vraie Classe', fontsize=12)
    ax1.set_title('Matrice de Confusion (valeurs absolues)', fontsize=14, fontweight='bold')

    # Matrice normalisée
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Pourcentage (%)'}, ax=ax2,
                vmin=0, vmax=100)
    ax2.set_xlabel('Prédiction', fontsize=12)
    ax2.set_ylabel('Vraie Classe', fontsize=12)
    ax2.set_title('Matrice de Confusion (pourcentages)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Matrice sauvegardée: {save_path}")

    plt.show()


def plot_sample_predictions(X, y_true, y_pred, y_probs=None, n_samples=25, figsize=(12, 12), save_path=None):
    """
    Affiche des exemples avec leurs prédictions

    Args:
        X: images (n, 784) ou (n, 28, 28)
        y_true: vraies classes
        y_pred: prédictions
        y_probs: probabilités optionnelles
        n_samples: nombre d'échantillons à afficher
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    # Reshape si nécessaire
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)

    n_samples = min(n_samples, len(X))
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]

    for i in range(n_samples):
        ax = axes[i]
        ax.imshow(X[i], cmap='gray')

        # Titre avec couleur selon correcte/incorrecte
        is_correct = y_true[i] == y_pred[i]
        color = 'green' if is_correct else 'red'

        title = f"Vrai: {y_true[i]} | Pred: {y_pred[i]}"
        if y_probs is not None:
            prob = y_probs[i, y_pred[i]] * 100
            title += f"\n({prob:.1f}%)"

        ax.set_title(title, color=color, fontsize=9, fontweight='bold')
        ax.axis('off')

    # Cacher les axes vides
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Exemples de Prédictions', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Exemples sauvegardés: {save_path}")

    plt.show()


def plot_weights_visualization(weights, n_neurons=64, figsize=(12, 12), save_path=None):
    """
    Visualise les poids de la première couche (ce que les neurones 'voient')

    Args:
        weights: matrice de poids (784, n_neurons)
        n_neurons: nombre de neurones à afficher
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    n_neurons = min(n_neurons, weights.shape[1])
    n_cols = int(np.ceil(np.sqrt(n_neurons)))
    n_rows = int(np.ceil(n_neurons / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_neurons > 1 else [axes]

    # Normaliser pour meilleure visualisation
    weights_normalized = (weights - weights.min()) / (weights.max() - weights.min())

    for i in range(n_neurons):
        ax = axes[i]
        weight_image = weights_normalized[:, i].reshape(28, 28)
        ax.imshow(weight_image, cmap='RdBu', interpolation='nearest')
        ax.set_title(f'Neurone {i}', fontsize=8)
        ax.axis('off')

    # Cacher les axes vides
    for i in range(n_neurons, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Visualisation des Poids (1ère Couche)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Poids sauvegardés: {save_path}")

    plt.show()


def plot_misclassified_examples(X, y_true, y_pred, y_probs=None, n_samples=16, figsize=(12, 10), save_path=None):
    """
    Affiche les exemples mal classifiés

    Args:
        X: images (n, 784) ou (n, 28, 28)
        y_true: vraies classes
        y_pred: prédictions
        y_probs: probabilités optionnelles
        n_samples: nombre d'exemples à afficher
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    # Trouver les indices mal classifiés
    misclassified_indices = np.where(y_true != y_pred)[0]

    if len(misclassified_indices) == 0:
        print("Aucune erreur trouvée ! <‰")
        return

    # Reshape si nécessaire
    if X.ndim == 2 and X.shape[1] == 784:
        X = X.reshape(-1, 28, 28)

    n_samples = min(n_samples, len(misclassified_indices))

    # Trier par confiance décroissante (erreurs confiantes = plus intéressantes)
    if y_probs is not None:
        confidences = [y_probs[i, y_pred[i]] for i in misclassified_indices]
        sorted_indices = misclassified_indices[np.argsort(confidences)[::-1]]
    else:
        sorted_indices = misclassified_indices[:n_samples]

    n_cols = 4
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]

    for i, idx in enumerate(sorted_indices[:n_samples]):
        ax = axes[i]
        ax.imshow(X[idx], cmap='gray')

        title = f"Vrai: {y_true[idx]} ’ Pred: {y_pred[idx]}"
        if y_probs is not None:
            prob = y_probs[idx, y_pred[idx]] * 100
            title += f"\nConfiance: {prob:.1f}%"

        ax.set_title(title, color='red', fontsize=9, fontweight='bold')
        ax.axis('off')

    # Cacher les axes vides
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Exemples Mal Classifiés ({len(misclassified_indices)} erreurs totales)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Erreurs sauvegardées: {save_path}")

    plt.show()


def plot_class_distribution(y, class_names=None, figsize=(10, 6), save_path=None):
    """
    Affiche la distribution des classes

    Args:
        y: labels (n,)
        class_names: noms des classes
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    unique, counts = np.unique(y, return_counts=True)

    if class_names is None:
        class_names = [f"Classe {i}" for i in unique]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
    bars = ax.bar(class_names, counts, color=colors, edgecolor='black', linewidth=1.5)

    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Classe', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre d\'exemples', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des Classes', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Distribution sauvegardée: {save_path}")

    plt.show()


def plot_learning_rate_comparison(histories, lr_values, figsize=(14, 5), save_path=None):
    """
    Compare différentes valeurs de learning rate

    Args:
        histories: liste de dictionnaires history
        lr_values: liste des learning rates testés
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(histories)))

    for i, (history, lr) in enumerate(zip(histories, lr_values)):
        epochs = range(1, len(history['loss']) + 1)

        # Loss
        ax1.plot(epochs, history['loss'], '-o', color=colors[i],
                linewidth=2, markersize=3, label=f'LR = {lr}')

        # Val Accuracy
        ax2.plot(epochs, history['val_acc'], '-o', color=colors[i],
                linewidth=2, markersize=3, label=f'LR = {lr}')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Comparaison Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Comparaison Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Comparaison sauvegardée: {save_path}")

    plt.show()


def plot_architecture_comparison(results, figsize=(12, 8), save_path=None):
    """
    Compare différentes architectures de réseau

    Args:
        results: liste de dictionnaires avec 'name', 'train_acc', 'val_acc', 'time'
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    names = [r['name'] for r in results]
    train_accs = [r['train_acc'] for r in results]
    val_accs = [r['val_acc'] for r in results]
    times = [r['time'] for r in results]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig)

    # Accuracy comparison
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, train_accs, width, label='Train Accuracy',
                    color='skyblue', edgecolor='black')
    bars2 = ax1.bar(x + width/2, val_accs, width, label='Val Accuracy',
                    color='lightcoral', edgecolor='black')

    ax1.set_xlabel('Architecture', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Comparaison des Architectures', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([min(min(train_accs), min(val_accs)) - 0.02, 1.0])

    # Ajouter valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)

    # Training time
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(names, times, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Architecture', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temps (secondes)', fontsize=12, fontweight='bold')
    ax2.set_title('Temps d\'Entraînement', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)

    # Overfitting indicator (train - val)
    ax3 = fig.add_subplot(gs[1, 1])
    overfit = [train - val for train, val in zip(train_accs, val_accs)]
    colors_overfit = ['red' if o > 0.05 else 'green' for o in overfit]
    bars = ax3.bar(names, overfit, color=colors_overfit, edgecolor='black', alpha=0.7)
    ax3.axhline(y=0.05, color='orange', linestyle='--', label='Seuil surapprentissage')
    ax3.set_xlabel('Architecture', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Train Acc - Val Acc', fontsize=12, fontweight='bold')
    ax3.set_title('Indicateur de Surapprentissage', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Comparaison architectures sauvegardée: {save_path}")

    plt.show()


def plot_activation_outputs(activations, layer_names, n_samples=3, n_neurons=16, figsize=(15, 10), save_path=None):
    """
    Visualise les sorties d'activation de différentes couches

    Args:
        activations: dictionnaire {layer_name: activation_output}
        layer_names: liste des noms de couches à visualiser
        n_samples: nombre d'échantillons
        n_neurons: nombre de neurones à afficher par couche
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_samples, n_layers, figsize=figsize)

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for col, layer_name in enumerate(layer_names):
        layer_output = activations[layer_name]

        for row in range(n_samples):
            ax = axes[row, col]

            # Prendre les n premiers neurones
            sample_activations = layer_output[row, :n_neurons]

            ax.bar(range(len(sample_activations)), sample_activations,
                   color='steelblue', edgecolor='black')
            ax.set_ylim([0, max(sample_activations) * 1.1])

            if row == 0:
                ax.set_title(f'{layer_name}\n({layer_output.shape[1]} neurones)',
                           fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Échantillon {row+1}', fontsize=10)

            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Activations des Couches Cachées', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Activations sauvegardées: {save_path}")

    plt.show()


def plot_probability_distribution(y_probs, y_true, n_samples=10, figsize=(12, 8), save_path=None):
    """
    Affiche la distribution de probabilités pour des échantillons

    Args:
        y_probs: probabilités prédites (n_samples, n_classes)
        y_true: vraies classes
        n_samples: nombre d'échantillons à afficher
        figsize: taille de la figure
        save_path: chemin de sauvegarde optionnel
    """
    n_samples = min(n_samples, len(y_probs))
    n_classes = y_probs.shape[1]

    fig, axes = plt.subplots(2, 5, figsize=figsize) if n_samples == 10 else \
                plt.subplots(n_samples, 1, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]

    for i in range(n_samples):
        ax = axes[i]
        probs = y_probs[i]
        pred_class = np.argmax(probs)
        true_class = y_true[i]

        colors = ['green' if j == true_class else 'lightblue' for j in range(n_classes)]
        colors[pred_class] = 'red' if pred_class != true_class else 'darkgreen'

        bars = ax.bar(range(n_classes), probs, color=colors, edgecolor='black')
        ax.set_ylim([0, 1])
        ax.set_xticks(range(n_classes))
        ax.set_title(f'Vrai: {true_class} | Pred: {pred_class} ({probs[pred_class]*100:.1f}%)',
                    fontsize=9, fontweight='bold',
                    color='green' if pred_class == true_class else 'red')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Distribution de Probabilités', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n Probabilités sauvegardées: {save_path}")

    plt.show()
