# CLAUDE.md - Documentation du Projet de Reconnaissance de Chiffres

## 🎯 Objectif du Projet

Ce projet est un exercice d'apprentissage pour comprendre en profondeur le fonctionnement des réseaux de neurones en implémentant un système de reconnaissance de chiffres manuscrits **from scratch** (sans utiliser PyTorch, TensorFlow ou Keras).

L'objectif est de comprendre :
- Comment fonctionne la propagation avant (forward propagation)
- Comment fonctionne la rétropropagation (backpropagation)
- Comment les réseaux de neurones apprennent à partir de données
- Comment optimiser et améliorer les performances

## 📁 Structure du Projet

```
nombre/
├── CLAUDE.md              # Ce fichier - documentation détaillée
├── README.md              # Vue d'ensemble du projet
├── ARCHITECTURE.md        # Explications mathématiques détaillées
├── requirements.txt       # Dépendances Python
│
├── data/                  # Dataset MNIST
│   ├── mnist_train.csv
│   └── mnist_test.csv
│
├── src/                   # Code source principal
│   ├── __init__.py
│   ├── network.py         # Classe principale du réseau de neurones
│   ├── layers.py          # Implémentation des couches (Dense, etc.)
│   ├── activations.py     # Fonctions d'activation (sigmoid, ReLU, softmax)
│   ├── losses.py          # Fonctions de coût (MSE, cross-entropy)
│   ├── optimizers.py      # Algorithmes d'optimisation (SGD, Adam)
│   ├── utils.py           # Utilitaires (chargement données, normalisation)
│   ├── visualize.py       # Fonctions de visualisation
│   └── metrics.py         # Calcul des métriques (accuracy, confusion matrix)
│
├── notebooks/             # Notebooks Jupyter pour expérimentation
│   ├── 01_exploration.ipynb      # Exploration du dataset MNIST
│   ├── 02_simple_network.ipynb   # Réseau simple
│   └── 03_improvements.ipynb     # Améliorations et optimisations
│
├── models/                # Modèles entraînés sauvegardés
│   └── .gitkeep
│
├── tests/                 # Tests unitaires
│   ├── test_activations.py
│   ├── test_layers.py
│   └── test_network.py
│
├── train.py               # Script d'entraînement principal
├── evaluate.py            # Script d'évaluation
└── draw_interface.py      # Interface pour dessiner et tester
```

## 🧠 Concepts Implémentés

### 1. Architecture du Réseau de Neurones

#### Phase 1 : Réseau Simple (MLP - Multi-Layer Perceptron)
```
Input Layer (784 neurones)
    ↓
Hidden Layer 1 (128 neurones) + ReLU
    ↓
Hidden Layer 2 (64 neurones) + ReLU
    ↓
Output Layer (10 neurones) + Softmax
```

#### Phase 2 : Améliorations
- Plus de couches cachées
- Dropout pour éviter le surapprentissage
- Batch normalization
- Différentes fonctions d'activation

### 2. Fonctions d'Activation

| Fonction | Équation | Usage |
|----------|----------|-------|
| **Sigmoid** | σ(x) = 1/(1+e^(-x)) | Couches cachées (historique) |
| **ReLU** | f(x) = max(0, x) | Couches cachées (moderne) |
| **Softmax** | f(x)ᵢ = e^(xᵢ) / Σe^(xⱼ) | Couche de sortie (classification) |
| **Tanh** | f(x) = (e^x - e^(-x))/(e^x + e^(-x)) | Alternative à sigmoid |

### 3. Fonctions de Coût

- **Cross-Entropy** : Pour la classification multi-classes
  ```
  L = -Σ yᵢ log(ŷᵢ)
  ```
- **MSE (Mean Squared Error)** : Alternative plus simple
  ```
  L = (1/n) Σ(y - ŷ)²
  ```

### 4. Algorithmes d'Optimisation

- **SGD (Stochastic Gradient Descent)** : Basique
- **SGD avec Momentum** : Accélération de la convergence
- **Adam** : Adaptatif et performant

## 🚀 Fonctionnalités

### ✅ Fonctionnalités Principales

1. **Entraînement du modèle**
   - Chargement automatique de MNIST
   - Entraînement avec différentes configurations
   - Sauvegarde des poids entraînés

2. **Évaluation et métriques**
   - Accuracy globale et par classe
   - Matrice de confusion
   - Courbes d'apprentissage (loss/accuracy)

3. **Interface de test interactive**
   - Dessiner des chiffres à la main
   - Prédiction en temps réel
   - Affichage des probabilités

### ✅ Fonctionnalités Avancées

4. **Visualisation**
   - Visualisation des poids de la première couche
   - Exemples mal classifiés
   - Évolution des métriques pendant l'entraînement

5. **Comparaison de modèles**
   - Tester différentes architectures
   - Comparer les performances
   - Sauvegarder les résultats

6. **Data Augmentation**
   - Rotation légère
   - Translation
   - Zoom

7. **Tests unitaires**
   - Validation des calculs de gradient
   - Tests des fonctions d'activation
   - Tests de backpropagation

8. **Documentation explicative**
   - Notebooks avec explications pas à pas
   - Commentaires détaillés dans le code
   - Explications mathématiques

## 📊 Dataset MNIST

- **Taille** : 60,000 images d'entraînement + 10,000 images de test
- **Format** : Images en niveaux de gris 28×28 pixels
- **Classes** : Chiffres de 0 à 9
- **Prétraitement** : Normalisation des pixels [0, 255] → [0, 1]

## 🔧 Utilisation

### Installation

```bash
# Cloner le dépôt
git clone <repo-url>
cd nombre

# Installer les dépendances
pip install -r requirements.txt
```

### Entraînement

```bash
# Entraînement avec configuration par défaut
python train.py

# Entraînement avec paramètres personnalisés
python train.py --epochs 20 --batch-size 64 --learning-rate 0.001
```

### Évaluation

```bash
# Évaluer un modèle entraîné
python evaluate.py --model-path models/best_model.pkl
```

### Interface de dessin

```bash
# Lancer l'interface graphique
python draw_interface.py
```

## 🎓 Apprentissage Progressif

### Étape 1 : Comprendre les bases
- Lire `ARCHITECTURE.md` pour comprendre les mathématiques
- Explorer le notebook `01_exploration.ipynb`
- Comprendre le dataset MNIST

### Étape 2 : Implémenter le réseau simple
- Implémenter les fonctions d'activation
- Implémenter la propagation avant
- Implémenter la rétropropagation
- Tester avec `02_simple_network.ipynb`

### Étape 3 : Entraîner et évaluer
- Entraîner un premier modèle
- Analyser les résultats
- Identifier les faiblesses

### Étape 4 : Améliorer
- Ajouter des couches
- Tester différentes fonctions d'activation
- Optimiser les hyperparamètres
- Implémenter des techniques avancées

### Étape 5 : Expérimenter
- Data augmentation
- Visualisation des poids
- Analyse des erreurs

## 🐛 Debugging et Validation

### Vérification du Gradient

Pour s'assurer que la backpropagation est correctement implémentée :

```python
# Gradient checking (comparaison numérique vs analytique)
python -m tests.test_network
```

### Validation de l'Apprentissage

Signes d'un apprentissage correct :
- ✅ Loss qui diminue progressivement
- ✅ Accuracy qui augmente sur le train et le test
- ✅ Pas de divergence (loss qui explose)

Signes de problèmes :
- ❌ Loss qui stagne immédiatement → Learning rate trop faible
- ❌ Loss qui explose → Learning rate trop élevé
- ❌ Train accuracy élevée mais test accuracy faible → Surapprentissage

## 📈 Résultats Attendus

### Réseau Simple (MLP)
- **Accuracy attendue** : ~95-97% sur le test set
- **Temps d'entraînement** : 5-10 minutes sur CPU

### Réseau Amélioré
- **Accuracy attendue** : ~98-99% sur le test set
- **Temps d'entraînement** : 10-20 minutes sur CPU

## 🔬 Expérimentations Suggérées

1. **Impact du learning rate** : Tester 0.001, 0.01, 0.1
2. **Nombre de couches** : 1 vs 2 vs 3 couches cachées
3. **Taille des couches** : 32, 64, 128, 256 neurones
4. **Fonctions d'activation** : ReLU vs Sigmoid vs Tanh
5. **Batch size** : 16, 32, 64, 128
6. **Optimisateurs** : SGD vs Momentum vs Adam

## 📚 Ressources Complémentaires

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Détails mathématiques
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Michael Nielsen - Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Andrew Ng - Machine Learning Course](https://www.coursera.org/learn/machine-learning)

## 🤝 Contribution

Ce projet est un projet d'apprentissage personnel. Suggestions d'améliorations bienvenues !

## 📝 Notes de Développement

### Version 1.0 - Réseau Simple
- [ ] Implémentation basique du MLP
- [ ] Fonctions d'activation (sigmoid, ReLU, softmax)
- [ ] Backpropagation
- [ ] Entraînement sur MNIST
- [ ] Évaluation basique

### Version 2.0 - Améliorations
- [ ] Interface de dessin
- [ ] Visualisations avancées
- [ ] Sauvegarde/chargement des modèles
- [ ] Comparaison de configurations
- [ ] Data augmentation

### Version 3.0 - Optimisations
- [ ] Optimisateurs avancés (Adam)
- [ ] Batch normalization
- [ ] Dropout
- [ ] Tests unitaires complets

---

**Date de création** : 2025-11-07
**Auteur** : Projet d'apprentissage personnel
**Langage** : Python 3.8+
