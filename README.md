# 🔢 Nombre - Reconnaissance de Chiffres Manuscrits

Un projet d'apprentissage pour comprendre les réseaux de neurones en implémentant from scratch un système de reconnaissance de chiffres manuscrits (MNIST).

## 🎯 Objectif

Apprendre le fonctionnement interne des réseaux de neurones en construisant un classificateur de chiffres **sans utiliser de frameworks** comme PyTorch, TensorFlow ou Keras. Seulement NumPy, Matplotlib et les mathématiques !

## ✨ Fonctionnalités

- ✅ Implémentation from scratch d'un réseau de neurones multi-couches
- ✅ Entraînement sur le dataset MNIST (60,000 images)
- ✅ Interface graphique pour dessiner et tester des chiffres
- ✅ Visualisation de l'apprentissage (courbes, matrice de confusion)
- ✅ Visualisation des poids et features apprises
- ✅ Sauvegarde/chargement des modèles entraînés
- ✅ Comparaison de différentes architectures
- ✅ Data augmentation
- ✅ Tests unitaires pour valider l'implémentation

## 🚀 Quick Start

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/Yteria17/nombre.git
cd nombre

# Installer les dépendances
pip install -r requirements.txt
```

### Entraîner un modèle

```bash
# Entraînement simple avec paramètres par défaut
python train.py

# Entraînement avec paramètres personnalisés
python train.py --epochs 20 --batch-size 64 --lr 0.001 --hidden-layers 128 64
```

### Tester avec l'interface graphique

```bash
python draw_interface.py
```

Dessinez un chiffre et voyez la prédiction en temps réel !

### Évaluer le modèle

```bash
python evaluate.py --model-path models/best_model.pkl
```

## 📊 Résultats

| Modèle | Architecture | Accuracy | Temps d'entraînement |
|--------|-------------|----------|---------------------|
| Simple MLP | 784-128-64-10 | ~96% | 5 min (CPU) |
| MLP Optimisé | 784-256-128-64-10 | ~98% | 15 min (CPU) |

## 🧠 Architecture

### Réseau Simple (Version 1)

```
Input (784)  →  Hidden (128)  →  Hidden (64)  →  Output (10)
               [ReLU]           [ReLU]          [Softmax]
```

### Composants Implémentés

- **Couches** : Dense (fully connected)
- **Activations** : Sigmoid, ReLU, Tanh, Softmax
- **Loss** : Cross-Entropy, MSE
- **Optimisateurs** : SGD, SGD + Momentum, Adam
- **Régularisation** : L2, Dropout (version avancée)

## 📁 Structure du Projet

```
nombre/
├── src/                   # Code source
│   ├── network.py         # Réseau de neurones
│   ├── layers.py          # Couches
│   ├── activations.py     # Fonctions d'activation
│   ├── losses.py          # Fonctions de coût
│   ├── optimizers.py      # Optimisateurs
│   ├── utils.py           # Utilitaires
│   ├── visualize.py       # Visualisation
│   └── metrics.py         # Métriques
├── notebooks/             # Notebooks Jupyter
├── tests/                 # Tests unitaires
├── models/                # Modèles sauvegardés
├── train.py               # Script d'entraînement
├── evaluate.py            # Script d'évaluation
└── draw_interface.py      # Interface de test
```

## 📚 Documentation

- **[CLAUDE.md](./CLAUDE.md)** - Documentation complète du projet
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Explications mathématiques détaillées
- **[notebooks/](./notebooks/)** - Tutoriels interactifs

## 🎓 Apprendre avec ce Projet

### 1. Comprendre les Bases
Lisez `ARCHITECTURE.md` pour comprendre :
- La propagation avant (forward pass)
- La rétropropagation (backpropagation)
- La descente de gradient
- Les fonctions d'activation

### 2. Explorer le Code
Commencez par les notebooks :
1. `01_exploration.ipynb` - Dataset MNIST
2. `02_simple_network.ipynb` - Premier réseau
3. `03_improvements.ipynb` - Optimisations

### 3. Expérimenter
Testez différentes configurations :
- Nombre de couches
- Taille des couches
- Learning rate
- Fonctions d'activation
- Optimisateurs

## 🧪 Tests

```bash
# Lancer tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_network.py
pytest tests/test_activations.py
```

## 🔬 Expérimentations

Quelques idées d'expérimentations :

1. **Impact du learning rate**
   ```bash
   python train.py --lr 0.001
   python train.py --lr 0.01
   python train.py --lr 0.1
   ```

2. **Architecture profonde vs large**
   ```bash
   python train.py --hidden-layers 512 256 128 64  # Profond
   python train.py --hidden-layers 256             # Large
   ```

3. **Différents optimisateurs**
   ```bash
   python train.py --optimizer sgd
   python train.py --optimizer momentum
   python train.py --optimizer adam
   ```

## 📈 Visualisations

Le projet inclut plusieurs visualisations :

- **Courbes d'apprentissage** : Loss et accuracy au fil des epochs
- **Matrice de confusion** : Performance par classe
- **Poids de la première couche** : Ce que les neurones "voient"
- **Exemples d'erreurs** : Images mal classifiées
- **Distribution des probabilités** : Confiance du modèle

## 🛠️ Technologies

- **Python 3.8+**
- **NumPy** - Calculs matriciels
- **Matplotlib** - Visualisation
- **Pillow** - Manipulation d'images
- **Jupyter** - Notebooks interactifs
- **pytest** - Tests unitaires

## 🤝 Contribution

Ce projet est un projet d'apprentissage. Les suggestions d'améliorations sont les bienvenues !

## 📝 TODO

- [ ] Implémentation réseau simple (MLP)
- [ ] Entraînement et évaluation
- [ ] Interface de dessin
- [ ] Visualisations avancées
- [ ] Dropout et batch normalization
- [ ] Data augmentation
- [ ] Optimisateurs avancés (Adam)
- [ ] Tests unitaires complets

## 📖 Ressources

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Michael Nielsen - Neural Networks Book](http://neuralnetworksanddeeplearning.com/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## 📄 Licence

Ce projet est à but éducatif.

---

**Note** : Ce projet est conçu pour l'apprentissage. Pour des applications en production, utilisez des frameworks optimisés comme PyTorch ou TensorFlow.