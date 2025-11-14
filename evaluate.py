"""
Script d'Évaluation pour Modèles Entraînés

Évalue un modèle sauvegardé sur le test set et génère des visualisations et rapports.

Usage:
    # Évaluation simple
    python evaluate.py --model models/best_model.pkl

    # Avec visualisations
    python evaluate.py --model models/best_model.pkl --visualize

    # Rapport détaillé
    python evaluate.py --model models/best_model.pkl --detailed

    # Tout
    python evaluate.py --model models/best_model.pkl --visualize --detailed --save-plots results/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import time

from src.network import NeuralNetwork
from src.utils import load_mnist_data
from src import visualize
from src.metrics import (
    confusion_matrix,
    print_classification_report,
    top_k_accuracy
)


def parse_args():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Évalue un modèle entraîné sur le test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Simple
  python evaluate.py --model models/best_model.pkl

  # Avec toutes les visualisations
  python evaluate.py --model models/best_model.pkl --visualize --detailed

  # Sauvegarder dans un dossier spécifique
  python evaluate.py --model models/best_model.pkl --save-plots results/my_eval/
        """
    )

    parser.add_argument(
        '--model', '--model-path',
        type=str,
        required=True,
        dest='model_path',
        help='Chemin vers le modèle sauvegardé (.pkl)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Créer des visualisations (matrice de confusion, prédictions, etc.)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Afficher un rapport détaillé (métriques par classe)'
    )
    parser.add_argument(
        '--save-plots',
        type=str,
        default=None,
        help='Dossier où sauvegarder les plots. Default: models/'
    )
    parser.add_argument(
        '--show-errors',
        type=int,
        default=0,
        metavar='N',
        help='Afficher N exemples mal classifiés'
    )
    parser.add_argument(
        '--show-correct',
        type=int,
        default=0,
        metavar='N',
        help='Afficher N exemples bien classifiés'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Calculer la top-k accuracy (ex: --top-k 5 pour top-5)'
    )

    return parser.parse_args()


def print_model_info(model, model_path):
    """Affiche les informations sur le modèle"""
    print("\n" + "="*80)
    print("=Ë INFORMATIONS SUR LE MODÈLE")
    print("="*80)
    print(f"\n=Â Fichier: {model_path}")
    print(f"\n>à Architecture:")
    print(f"   {' ’ '.join(map(str, model.layer_dims))}")
    print(f"   Couches: {len(model.layer_dims)} ({len(model.layer_dims)-2} cachées)")

    n_params = sum(p.size for p in model.parameters.values())
    print(f"\n=Ê Paramètres:")
    print(f"   Total: {n_params:,}")
    print(f"   Learning Rate: {model.learning_rate}")
    print(f"   Optimizer: {model.optimizer_name}")

    if hasattr(model, 'history') and model.history['loss']:
        print(f"\n=È Historique d'entraînement:")
        print(f"   Époques: {len(model.history['loss'])}")
        print(f"   Loss finale: {model.history['loss'][-1]:.4f}")
        if model.history['train_acc']:
            print(f"   Train Acc finale: {model.history['train_acc'][-1]:.4f}")
        if model.history['val_acc']:
            print(f"   Val Acc finale: {model.history['val_acc'][-1]:.4f}")

    print("\n" + "="*80 + "\n")


def evaluate_model(model, X_test, y_test, args):
    """Évalue le modèle et affiche les résultats"""
    print("= Évaluation en cours...\n")

    # Temps d'inférence
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # Probabilités
    y_probs, _ = model.forward(X_test)

    # Accuracy
    test_acc = np.mean(y_pred == y_test)

    # Afficher les résultats principaux
    print("="*80)
    print(" RÉSULTATS DE L'ÉVALUATION")
    print("="*80)
    print(f"\n<¯ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"ñ  Temps d'inférence: {inference_time:.3f}s")
    print(f"   Samples/sec: {len(X_test)/inference_time:,.0f}")
    print(f"   ms/sample: {inference_time*1000/len(X_test):.2f}ms")

    # Nombre d'erreurs
    n_errors = np.sum(y_pred != y_test)
    print(f"\nL Erreurs: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.2f}%)")

    # Top-k accuracy si demandée
    if args.top_k:
        topk_acc = top_k_accuracy(y_test, y_probs, k=args.top_k)
        print(f"\n=Ê Top-{args.top_k} Accuracy: {topk_acc:.4f} ({topk_acc*100:.2f}%)")

    print("\n" + "="*80)

    return y_pred, y_probs


def show_examples(X, y_true, y_pred, y_probs, n_examples, show_correct=True):
    """Affiche des exemples de prédictions"""
    import matplotlib.pyplot as plt

    if show_correct:
        indices = np.where(y_true == y_pred)[0]
        title = "Exemples Bien Classifiés"
        color = 'green'
    else:
        indices = np.where(y_true != y_pred)[0]
        title = "Exemples Mal Classifiés"
        color = 'red'

    if len(indices) == 0:
        print(f"\n{'' if show_correct else 'L'} Aucun exemple {'correct' if show_correct else 'incorrect'} trouvé!")
        return

    # Sélectionner aléatoirement
    n_examples = min(n_examples, len(indices))
    selected = np.random.choice(indices, n_examples, replace=False)

    # Afficher
    n_cols = min(5, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    if n_examples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, idx in enumerate(selected):
        ax = axes[i]
        image = X[idx].reshape(28, 28)

        ax.imshow(image, cmap='gray')

        true_label = y_true[idx]
        pred_label = y_pred[idx]
        prob = y_probs[idx, pred_label] * 100

        ax.set_title(f"Vrai: {true_label}\nPred: {pred_label}\n({prob:.1f}%)",
                    color=color, fontsize=9, fontweight='bold')
        ax.axis('off')

    # Cacher les axes vides
    for i in range(n_examples, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    print("\n" + "="*80)
    print("=Ê ÉVALUATION DE MODÈLE")
    print("="*80)

    # Vérifier que le modèle existe
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\nL Erreur: Le modèle {model_path} n'existe pas!")
        sys.exit(1)

    # Charger le modèle
    print(f"\n=Â Chargement du modèle: {model_path}...")
    model = NeuralNetwork.load(model_path)
    print("    Modèle chargé")

    # Afficher les infos
    print_model_info(model, model_path)

    # Charger les données
    print("=Â Chargement des données de test...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()
    print(f"    Test set: {X_test.shape[0]:,} exemples\n")

    # Évaluer
    y_pred, y_probs = evaluate_model(model, X_test, y_test, args)

    # Rapport détaillé
    if args.detailed:
        print("\n" + "="*80)
        print("=Ë RAPPORT DÉTAILLÉ PAR CLASSE")
        print("="*80)
        print_classification_report(
            y_test, y_pred,
            class_names=[str(i) for i in range(10)]
        )

    # Visualisations
    if args.visualize:
        plot_dir = Path(args.save_plots or 'models')
        plot_dir.mkdir(parents=True, exist_ok=True)

        print("\n=È Création des visualisations...\n")

        # Matrice de confusion
        print("   - Matrice de confusion...")
        cm = confusion_matrix(y_test, y_pred, num_classes=10)
        visualize.plot_confusion_matrix(
            cm,
            class_names=[str(i) for i in range(10)],
            save_path=str(plot_dir / 'eval_confusion_matrix.png')
        )

        # Prédictions échantillons
        print("   - Échantillons de prédictions...")
        visualize.plot_sample_predictions(
            X_test, y_test, y_pred, y_probs,
            n_samples=25,
            save_path=str(plot_dir / 'eval_sample_predictions.png')
        )

        # Exemples mal classifiés
        print("   - Exemples mal classifiés...")
        visualize.plot_misclassified_examples(
            X_test, y_test, y_pred, y_probs,
            n_samples=16,
            save_path=str(plot_dir / 'eval_misclassified.png')
        )

        # Distribution de probabilités
        print("   - Distribution des probabilités...")
        visualize.plot_probability_distribution(
            y_probs, y_test,
            n_samples=10,
            save_path=str(plot_dir / 'eval_probability_dist.png')
        )

        print(f"\n    Visualisations sauvegardées dans: {plot_dir}/")

    # Montrer des exemples si demandé
    if args.show_errors > 0:
        print(f"\n= Affichage de {args.show_errors} exemples mal classifiés...")
        show_examples(X_test, y_test, y_pred, y_probs,
                     args.show_errors, show_correct=False)

    if args.show_correct > 0:
        print(f"\n Affichage de {args.show_correct} exemples bien classifiés...")
        show_examples(X_test, y_test, y_pred, y_probs,
                     args.show_correct, show_correct=True)

    print("\n" + "="*80)
    print(" Évaluation terminée !")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n   Évaluation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\nL Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
