"""
Script d'Entraînement CLI pour Réseaux de Neurones

Usage:
    # Entraînement simple avec paramètres par défaut
    python train.py

    # Configuration personnalisée
    python train.py --epochs 20 --batch-size 64 --lr 0.001 --hidden-layers 256 128 64

    # Avec visualisations et sauvegarde
    python train.py --epochs 15 --optimizer adam --save models/my_model.pkl --visualize

    # Mode verbose
    python train.py --epochs 10 --verbose
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

from src.network import NeuralNetwork
from src.utils import load_mnist_data
from src import visualize
from src.metrics import confusion_matrix, print_classification_report


def parse_args():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Entraîne un réseau de neurones sur MNIST',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Simple
  python train.py

  # Personnalisé
  python train.py --epochs 20 --lr 0.001 --hidden-layers 256 128

  # Avec sauvegarde et visualisations
  python train.py --save models/best.pkl --visualize

  # Optimiseur Adam avec learning rate adapté
  python train.py --optimizer adam --lr 0.01 --epochs 15
        """
    )

    # Hyperparamètres du réseau
    parser.add_argument(
        '--hidden-layers',
        type=int,
        nargs='+',
        default=[256, 128],
        help='Tailles des couches cachées (ex: 256 128 64). Default: 256 128'
    )

    # Hyperparamètres d'entraînement
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Nombre d\'époques. Default: 10'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Taille des mini-batches. Default: 128'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=0.01,
        dest='learning_rate',
        help='Learning rate. Default: 0.01'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['sgd', 'momentum', 'adam'],
        default='adam',
        help='Optimiseur à utiliser. Default: adam'
    )

    # Options de visualisation et sauvegarde
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Chemin pour sauvegarder le modèle (ex: models/my_model.pkl)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Créer des visualisations (courbes, matrice de confusion)'
    )
    parser.add_argument(
        '--save-plots',
        type=str,
        default=None,
        help='Dossier où sauvegarder les plots. Default: models/'
    )

    # Options d'affichage
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mode verbeux (affiche plus de détails)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Mode silencieux (affiche le minimum)'
    )

    # Seed pour reproductibilité
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed pour reproductibilité. Default: 42'
    )

    return parser.parse_args()


def print_config(args, layer_dims):
    """Affiche la configuration de l'entraînement"""
    print("\n" + "="*80)
    print("™  CONFIGURATION DE L'ENTRAÎNEMENT")
    print("="*80)
    print(f"\n>à Architecture du Réseau:")
    print(f"   {' ’ '.join(map(str, layer_dims))}")
    print(f"   Total: {len(layer_dims)} couches ({len(layer_dims)-2} cachées)")

    print(f"\n=Ê Hyperparamètres:")
    print(f"   Epochs:        {args.epochs}")
    print(f"   Batch Size:    {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Optimizer:     {args.optimizer}")

    if args.save:
        print(f"\n=¾ Sauvegarde:")
        print(f"   Modèle: {args.save}")

    if args.visualize:
        plot_dir = args.save_plots or 'models'
        print(f"\n=È Visualisations:")
        print(f"   Dossier: {plot_dir}/")

    print("\n" + "="*80 + "\n")


def main():
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Configuration de verbosité
    verbose = not args.quiet
    show_training = args.verbose

    if not args.quiet:
        print("\n" + "="*80)
        print("=€ ENTRAÎNEMENT D'UN RÉSEAU DE NEURONES")
        print("="*80)

    # Charger les données
    if verbose:
        print("\n=Â Chargement des données MNIST...")

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()

    if verbose:
        print(f"    Train: {X_train.shape[0]:,} exemples")
        print(f"    Val:   {X_val.shape[0]:,} exemples")
        print(f"    Test:  {X_test.shape[0]:,} exemples")

    # Construire l'architecture
    layer_dims = [784] + args.hidden_layers + [10]

    # Afficher la configuration
    if verbose and not args.quiet:
        print_config(args, layer_dims)

    # Créer le modèle
    if verbose:
        print("<×  Création du modèle...")

    model = NeuralNetwork(
        layer_dims=layer_dims,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer
    )

    if verbose:
        n_params = sum(p.size for p in model.parameters.values())
        print(f"    Modèle créé avec {n_params:,} paramètres\n")

    # Entraîner
    if verbose:
        print("=% Début de l'entraînement...\n")

    start_time = time.time()

    model.train(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=show_training or verbose
    )

    training_time = time.time() - start_time

    # Évaluation
    if verbose:
        print("\n=Ê Évaluation du modèle...\n")

    train_acc = model.accuracy(X_train, y_train)
    val_acc = model.accuracy(X_val, y_val)
    test_acc = model.accuracy(X_test, y_test)

    # Afficher les résultats
    print("\n" + "="*80)
    print(" ENTRAÎNEMENT TERMINÉ !")
    print("="*80)
    print(f"\nñ  Temps d'entraînement: {training_time:.1f}s ({training_time/60:.1f} min)")

    print(f"\n<¯ Performances Finales:")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Val Accuracy:   {val_acc:.4f}")
    print(f"   Test Accuracy:  {test_acc:.4f}")

    overfit = train_acc - val_acc
    if overfit > 0.05:
        print(f"\n   Surapprentissage détecté (gap: {overfit:.4f})")
    else:
        print(f"\n Pas de surapprentissage (gap: {overfit:.4f})")

    # Rapport de classification détaillé si verbose
    if args.verbose:
        print("\n" + "="*80)
        y_pred = model.predict(X_test)
        print_classification_report(
            y_test, y_pred,
            class_names=[str(i) for i in range(10)]
        )

    # Sauvegarder le modèle
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)

        if verbose:
            print(f"\n=¾ Modèle sauvegardé: {save_path}")

    # Créer les visualisations
    if args.visualize:
        plot_dir = Path(args.save_plots or 'models')
        plot_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"\n=È Création des visualisations...")

        # Courbes d'apprentissage
        visualize.plot_training_history(
            model.history,
            save_path=str(plot_dir / 'training_history.png')
        )

        # Matrice de confusion
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, num_classes=10)
        visualize.plot_confusion_matrix(
            cm,
            class_names=[str(i) for i in range(10)],
            save_path=str(plot_dir / 'confusion_matrix.png')
        )

        # Prédictions
        y_probs, _ = model.forward(X_test)
        visualize.plot_sample_predictions(
            X_test, y_test, y_pred, y_probs,
            n_samples=25,
            save_path=str(plot_dir / 'sample_predictions.png')
        )

        # Poids de la première couche
        visualize.plot_weights_visualization(
            model.parameters['W1'],
            n_neurons=64,
            save_path=str(plot_dir / 'weights_layer1.png')
        )

        if verbose:
            print(f"    Visualisations sauvegardées dans: {plot_dir}/")

    print("\n" + "="*80)
    print("<‰ Terminé avec succès !")
    print("="*80 + "\n")

    # Retourner le test accuracy pour les scripts qui appellent train.py
    return test_acc


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n   Entraînement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\nL Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
