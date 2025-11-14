"""
Script de Benchmark - Comparaison Automatique d'Architectures

Teste plusieurs configurations de réseaux de neurones et compare leurs performances.

Usage:
    python benchmark.py [--quick]
    python benchmark.py --quick  # Version rapide pour tests
"""

import numpy as np
import time
import json
from pathlib import Path
import argparse
from datetime import datetime

from src.network import NeuralNetwork
from src.utils import load_mnist_data
from src import visualize

# Configurations à tester
QUICK_CONFIGS = [
    {
        'name': 'Small',
        'layers': [784, 128, 10],
        'lr': 0.01,
        'optimizer': 'sgd',
        'epochs': 5
    },
    {
        'name': 'Medium',
        'layers': [784, 256, 128, 10],
        'lr': 0.01,
        'optimizer': 'adam',
        'epochs': 5
    },
]

FULL_CONFIGS = [
    {
        'name': 'Tiny',
        'layers': [784, 64, 10],
        'lr': 0.01,
        'optimizer': 'sgd',
        'epochs': 10
    },
    {
        'name': 'Small',
        'layers': [784, 128, 10],
        'lr': 0.01,
        'optimizer': 'adam',
        'epochs': 15
    },
    {
        'name': 'Medium',
        'layers': [784, 256, 128, 10],
        'lr': 0.01,
        'optimizer': 'adam',
        'epochs': 15
    },
    {
        'name': 'Large',
        'layers': [784, 512, 256, 10],
        'lr': 0.01,
        'optimizer': 'adam',
        'epochs': 15
    },
    {
        'name': 'Deep',
        'layers': [784, 256, 128, 64, 10],
        'lr': 0.01,
        'optimizer': 'adam',
        'epochs': 20
    },
    {
        'name': 'Very Deep',
        'layers': [784, 256, 128, 64, 32, 10],
        'lr': 0.005,
        'optimizer': 'adam',
        'epochs': 20
    },
    {
        'name': 'Wide',
        'layers': [784, 1024, 10],
        'lr': 0.005,
        'optimizer': 'adam',
        'epochs': 15
    },
    {
        'name': 'Momentum',
        'layers': [784, 256, 128, 10],
        'lr': 0.01,
        'optimizer': 'momentum',
        'epochs': 15
    },
]


def format_time(seconds):
    """Formate un temps en secondes de manière lisible"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def run_benchmark(configs, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Exécute le benchmark pour toutes les configurations

    Returns:
        Liste de dictionnaires avec les résultats
    """
    results = []

    print("\n" + "="*80)
    print("🚀 DÉBUT DU BENCHMARK")
    print("="*80)
    print(f"\nNombre de configurations: {len(configs)}")
    print(f"Dataset: {X_train.shape[0]:,} train, {X_val.shape[0]:,} val, {X_test.shape[0]:,} test\n")

    for i, config in enumerate(configs, 1):
        print("\n" + "="*80)
        print(f"Configuration {i}/{len(configs)}: {config['name']}")
        print("="*80)
        print(f"  Architecture: {' → '.join(map(str, config['layers']))}")
        print(f"  Learning Rate: {config['lr']}")
        print(f"  Optimizer: {config['optimizer']}")
        print(f"  Epochs: {config['epochs']}")
        print()

        # Créer le modèle
        model = NeuralNetwork(
            layer_dims=config['layers'],
            learning_rate=config['lr'],
            optimizer=config['optimizer']
        )

        # Entraîner
        start_time = time.time()

        model.train(
            X_train, y_train, X_val, y_val,
            epochs=config['epochs'],
            batch_size=128,
            verbose=True
        )

        training_time = time.time() - start_time

        # Évaluer
        train_acc = model.accuracy(X_train, y_train)
        val_acc = model.accuracy(X_val, y_val)
        test_acc = model.accuracy(X_test, y_test)

        # Calculer le nombre de paramètres
        n_params = sum(p.size for p in model.parameters.values())

        # Sauvegarder les résultats
        result = {
            'name': config['name'],
            'architecture': config['layers'],
            'learning_rate': config['lr'],
            'optimizer': config['optimizer'],
            'epochs': config['epochs'],
            'train_acc': float(train_acc),
            'val_acc': float(val_acc),
            'test_acc': float(test_acc),
            'training_time': float(training_time),
            'n_parameters': int(n_params),
            'final_loss': float(model.history['loss'][-1]),
            'history': {
                'loss': [float(x) for x in model.history['loss']],
                'train_acc': [float(x) for x in model.history['train_acc']],
                'val_acc': [float(x) for x in model.history['val_acc']]
            }
        }

        results.append(result)

        print(f"\n{'─'*80}")
        print("📊 RÉSULTATS:")
        print(f"  Train Accuracy:    {train_acc:.4f}")
        print(f"  Val Accuracy:      {val_acc:.4f}")
        print(f"  Test Accuracy:     {test_acc:.4f}")
        print(f"  Training Time:     {format_time(training_time)}")
        print(f"  Parameters:        {n_params:,}")
        print(f"  Params/Sec:        {n_params/training_time:,.0f}")
        print(f"  Overfitting Gap:   {train_acc - val_acc:.4f}")
        print(f"{'─'*80}")

    return results


def print_summary(results):
    """Affiche un résumé des résultats"""
    print("\n\n" + "="*80)
    print("📊 RÉSUMÉ FINAL DU BENCHMARK")
    print("="*80 + "\n")

    # Trier par test accuracy
    sorted_results = sorted(results, key=lambda x: x['test_acc'], reverse=True)

    print(f"{'Rang':<5} {'Nom':<15} {'Test Acc':<12} {'Val Acc':<12} {'Temps':<12} {'Params':<12} {'Overfit'}")
    print("─" * 90)

    for rank, result in enumerate(sorted_results, 1):
        overfit = result['train_acc'] - result['val_acc']
        overfit_symbol = "⚠️" if overfit > 0.05 else "✓"

        print(f"{rank:<5} {result['name']:<15} {result['test_acc']:<12.4f} "
              f"{result['val_acc']:<12.4f} {format_time(result['training_time']):<12} "
              f"{result['n_parameters']:>10,}  {overfit_symbol} {overfit:.4f}")

    print("\n" + "="*80)

    # Meilleur modèle
    best = sorted_results[0]
    print(f"\n🏆 MEILLEUR MODÈLE: {best['name']}")
    print(f"   Test Accuracy: {best['test_acc']:.4f}")
    print(f"   Architecture: {' → '.join(map(str, best['architecture']))}")
    print(f"   Optimizer: {best['optimizer']}")
    print(f"   Training Time: {format_time(best['training_time'])}")

    # Modèle le plus rapide
    fastest = min(results, key=lambda x: x['training_time'])
    print(f"\n⚡ MODÈLE LE PLUS RAPIDE: {fastest['name']}")
    print(f"   Training Time: {format_time(fastest['training_time'])}")
    print(f"   Test Accuracy: {fastest['test_acc']:.4f}")

    # Meilleur compromis (accuracy / time)
    best_efficiency = max(results, key=lambda x: x['test_acc'] / (x['training_time'] / 60))
    print(f"\n⚖️ MEILLEUR COMPROMIS: {best_efficiency['name']}")
    print(f"   Test Accuracy: {best_efficiency['test_acc']:.4f}")
    print(f"   Training Time: {format_time(best_efficiency['training_time'])}")
    print(f"   Efficiency: {best_efficiency['test_acc'] / (best_efficiency['training_time'] / 60):.4f} acc/min")

    # Statistiques
    avg_acc = np.mean([r['test_acc'] for r in results])
    std_acc = np.std([r['test_acc'] for r in results])

    print(f"\n📈 STATISTIQUES:")
    print(f"   Moyenne Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"   Min: {min(r['test_acc'] for r in results):.4f}")
    print(f"   Max: {max(r['test_acc'] for r in results):.4f}")
    print(f"   Range: {max(r['test_acc'] for r in results) - min(r['test_acc'] for r in results):.4f}")

    print("\n" + "="*80 + "\n")


def save_results(results, output_dir='benchmark_results'):
    """Sauvegarde les résultats"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp pour le fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sauvegarder JSON
    json_path = output_dir / f'benchmark_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Résultats sauvegardés: {json_path}")

    # Créer visualisations
    try:
        import matplotlib.pyplot as plt

        # Préparer les données pour visualize
        viz_results = []
        for r in results:
            viz_results.append({
                'name': r['name'],
                'train_acc': r['train_acc'],
                'val_acc': r['val_acc'],
                'time': r['training_time']
            })

        # Sauvegarder le graphique
        viz_path = output_dir / f'benchmark_{timestamp}.png'
        visualize.plot_architecture_comparison(
            viz_results,
            figsize=(14, 10),
            save_path=str(viz_path)
        )

        print(f"✓ Graphique sauvegardé: {viz_path}")

    except Exception as e:
        print(f"⚠️ Erreur lors de la création du graphique: {e}")

    return json_path


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark de différentes architectures de réseaux de neurones'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Mode rapide (teste seulement 2 configurations pour debug)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Dossier de sortie pour les résultats'
    )

    args = parser.parse_args()

    # Choisir les configurations
    configs = QUICK_CONFIGS if args.quick else FULL_CONFIGS

    print("\n" + "="*80)
    print("🔬 SCRIPT DE BENCHMARK - RÉSEAUX DE NEURONES")
    print("="*80)
    print(f"\nMode: {'RAPIDE (debug)' if args.quick else 'COMPLET'}")
    print(f"Configurations à tester: {len(configs)}")

    # Charger les données
    print("\n📂 Chargement des données MNIST...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()

    print(f"✓ Données chargées:")
    print(f"  Train: {X_train.shape[0]:,} exemples")
    print(f"  Val:   {X_val.shape[0]:,} exemples")
    print(f"  Test:  {X_test.shape[0]:,} exemples")

    # Lancer le benchmark
    start_time = time.time()
    results = run_benchmark(configs, X_train, y_train, X_val, y_val, X_test, y_test)
    total_time = time.time() - start_time

    # Afficher le résumé
    print_summary(results)

    # Sauvegarder
    output_path = save_results(results, args.output_dir)

    # Temps total
    print(f"⏱️  Temps total du benchmark: {format_time(total_time)}")
    print(f"📁 Résultats sauvegardés dans: {args.output_dir}/")

    print("\n" + "="*80)
    print("✅ BENCHMARK TERMINÉ !")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
