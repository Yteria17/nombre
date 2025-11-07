# 🧠 Architecture et Concepts Mathématiques

Ce document explique en détail les concepts mathématiques et l'architecture du réseau de neurones implémenté dans ce projet.

## Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Le Neurone Artificiel](#le-neurone-artificiel)
3. [Propagation Avant (Forward Pass)](#propagation-avant-forward-pass)
4. [Fonctions d'Activation](#fonctions-dactivation)
5. [Fonction de Coût](#fonction-de-coût)
6. [Rétropropagation (Backpropagation)](#rétropropagation-backpropagation)
7. [Optimisation](#optimisation)
8. [Architecture du Réseau](#architecture-du-réseau)

---

## Vue d'ensemble

Un réseau de neurones est un modèle mathématique inspiré du cerveau humain, composé de **couches de neurones artificiels** connectés entre eux.

### Principe général

```
Input → [Couche 1] → [Couche 2] → ... → [Couche N] → Output
```

Le réseau apprend en ajustant les **poids** (weights) et **biais** (biases) de chaque connexion pour minimiser l'erreur de prédiction.

---

## Le Neurone Artificiel

### Structure d'un Neurone

Un neurone reçoit plusieurs entrées, les combine, et produit une sortie.

```
Entrées: x₁, x₂, ..., xₙ
Poids:   w₁, w₂, ..., wₙ
Biais:   b

Sortie:  y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

### Formule Mathématique

Pour un neurone `j` :

```
z_j = Σᵢ (wᵢⱼ · xᵢ) + bⱼ
a_j = f(z_j)
```

Où :
- `xᵢ` : entrées
- `wᵢⱼ` : poids de la connexion de `i` vers `j`
- `bⱼ` : biais du neurone `j`
- `z_j` : somme pondérée (activation pré-activation)
- `f` : fonction d'activation
- `a_j` : activation du neurone (sortie)

### Notation Matricielle

Pour une couche entière :

```
Z = W · X + b
A = f(Z)
```

Où :
- `X` : vecteur/matrice d'entrées
- `W` : matrice des poids
- `b` : vecteur des biais
- `Z` : activations pré-activation
- `A` : activations (sorties)

---

## Propagation Avant (Forward Pass)

La propagation avant consiste à calculer les sorties du réseau en propageant les données à travers les couches.

### Pour une couche Dense

```python
# Pseudo-code
Z = np.dot(W, X) + b    # Combinaison linéaire
A = activation(Z)        # Application fonction d'activation
```

### Pour un réseau complet (3 couches)

```
Couche 1:
Z¹ = W¹ · X + b¹
A¹ = f¹(Z¹)

Couche 2:
Z² = W² · A¹ + b²
A² = f²(Z²)

Couche 3 (sortie):
Z³ = W³ · A² + b³
A³ = f³(Z³)
```

### Exemple MNIST

Pour une image 28×28 pixels :

```
Input: X = [784 valeurs] (image aplatie)
       ↓
Layer 1: W¹[128×784], b¹[128]
       Z¹ = W¹ · X + b¹         [128 valeurs]
       A¹ = ReLU(Z¹)            [128 valeurs]
       ↓
Layer 2: W²[64×128], b²[64]
       Z² = W² · A¹ + b²        [64 valeurs]
       A² = ReLU(Z²)            [64 valeurs]
       ↓
Layer 3: W³[10×64], b³[10]
       Z³ = W³ · A² + b³        [10 valeurs]
       A³ = Softmax(Z³)         [10 probabilités]
```

---

## Fonctions d'Activation

Les fonctions d'activation introduisent de la **non-linéarité** dans le réseau, permettant d'apprendre des relations complexes.

### 1. Sigmoid

**Formule** :
```
σ(x) = 1 / (1 + e⁻ˣ)
```

**Dérivée** :
```
σ'(x) = σ(x) · (1 - σ(x))
```

**Propriétés** :
- Sortie entre 0 et 1
- Utilisée historiquement
- Problème : gradient vanishing pour valeurs extrêmes

**Graphique** :
```
  1 |     ┌─────
    |    /
0.5 |   /
    |  /
  0 |──
    └────────────
    -5  0   5
```

### 2. ReLU (Rectified Linear Unit)

**Formule** :
```
ReLU(x) = max(0, x) = {
    x  si x > 0
    0  si x ≤ 0
}
```

**Dérivée** :
```
ReLU'(x) = {
    1  si x > 0
    0  si x ≤ 0
}
```

**Propriétés** :
- Très utilisée dans les couches cachées
- Calcul rapide
- Résout le gradient vanishing
- Problème : "dying ReLU" (neurones morts si x < 0 toujours)

**Graphique** :
```
    |    ╱
    |   ╱
    |  ╱
    | ╱
────┼────
    |
```

### 3. Tanh (Tangente Hyperbolique)

**Formule** :
```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

**Dérivée** :
```
tanh'(x) = 1 - tanh²(x)
```

**Propriétés** :
- Sortie entre -1 et 1
- Centrée autour de 0 (mieux que sigmoid)
- Gradient plus fort que sigmoid

### 4. Softmax

**Formule** (pour un vecteur de sortie) :
```
softmax(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)
```

**Propriétés** :
- Transforme les valeurs en probabilités
- Σ softmax(xᵢ) = 1
- Utilisée pour la couche de sortie (classification multi-classes)

**Exemple** :
```
Input:  [2.0, 1.0, 0.1]
Softmax: [0.659, 0.242, 0.099]  (somme = 1.0)
```

---

## Fonction de Coût

La fonction de coût (loss) mesure l'erreur entre les prédictions et les vraies valeurs.

### 1. Cross-Entropy (Entropie Croisée)

Pour la classification multi-classes (utilisée avec Softmax).

**Formule** :
```
L = -Σᵢ yᵢ · log(ŷᵢ)
```

Où :
- `yᵢ` : vraie valeur (one-hot encoded)
- `ŷᵢ` : prédiction (probabilité)

**Exemple** :
```
Vraie classe: 3
y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  (one-hot)
ŷ = [0.05, 0.05, 0.10, 0.60, 0.10, 0.05, 0.02, 0.01, 0.01, 0.01]

L = -(0·log(0.05) + ... + 1·log(0.60) + ...)
  = -log(0.60)
  ≈ 0.51
```

**Propriétés** :
- Pénalise fortement les mauvaises prédictions confiantes
- Gradient bien défini avec softmax

### 2. MSE (Mean Squared Error)

**Formule** :
```
L = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

**Propriétés** :
- Plus simple à comprendre
- Moins adaptée pour la classification
- Souvent utilisée pour la régression

---

## Rétropropagation (Backpropagation)

La rétropropagation calcule les gradients de la fonction de coût par rapport à chaque poids et biais.

### Principe

On utilise la **règle de la chaîne** (chain rule) pour calculer les dérivées en remontant du dernier vers le premier layer.

### Notation

- `L` : Loss (fonction de coût)
- `∂L/∂W` : Gradient de la loss par rapport aux poids W
- `∂L/∂b` : Gradient de la loss par rapport aux biais b

### Algorithme

Pour la **dernière couche** (couche de sortie) :

```
∂L/∂Z³ = A³ - Y  (si softmax + cross-entropy)

∂L/∂W³ = (∂L/∂Z³) · A²ᵀ
∂L/∂b³ = ∂L/∂Z³
```

Pour les **couches cachées** (backprop de la couche l) :

```
∂L/∂Aˡ = W^(l+1)ᵀ · ∂L/∂Z^(l+1)

∂L/∂Zˡ = (∂L/∂Aˡ) ⊙ f'(Zˡ)
         où ⊙ est le produit élément par élément (Hadamard)

∂L/∂Wˡ = (∂L/∂Zˡ) · A^(l-1)ᵀ
∂L/∂bˡ = ∂L/∂Zˡ
```

### Exemple Concret (Réseau 3 couches)

#### Forward Pass
```
X → Z¹ = W¹·X + b¹ → A¹ = ReLU(Z¹)
  → Z² = W²·A¹ + b² → A² = ReLU(Z²)
  → Z³ = W³·A² + b³ → A³ = Softmax(Z³)
  → L = CrossEntropy(A³, Y)
```

#### Backward Pass
```
∂L/∂Z³ = A³ - Y

∂L/∂W³ = ∂L/∂Z³ · A²ᵀ
∂L/∂b³ = ∂L/∂Z³

∂L/∂A² = W³ᵀ · ∂L/∂Z³
∂L/∂Z² = ∂L/∂A² ⊙ ReLU'(Z²)
∂L/∂W² = ∂L/∂Z² · A¹ᵀ
∂L/∂b² = ∂L/∂Z²

∂L/∂A¹ = W²ᵀ · ∂L/∂Z²
∂L/∂Z¹ = ∂L/∂A¹ ⊙ ReLU'(Z¹)
∂L/∂W¹ = ∂L/∂Z¹ · Xᵀ
∂L/∂b¹ = ∂L/∂Z¹
```

### Dérivées des Fonctions d'Activation

#### ReLU
```python
def relu_derivative(Z):
    return (Z > 0).astype(float)
```

#### Sigmoid
```python
def sigmoid_derivative(A):
    return A * (1 - A)
```

#### Tanh
```python
def tanh_derivative(A):
    return 1 - A**2
```

#### Softmax + Cross-Entropy
```python
# Dérivée combinée simplifiée
dZ = A - Y  # Très simple !
```

---

## Optimisation

### 1. Gradient Descent (Descente de Gradient)

On met à jour les paramètres dans la direction opposée au gradient.

**Formule** :
```
W := W - α · ∂L/∂W
b := b - α · ∂L/∂b
```

Où `α` est le **learning rate** (taux d'apprentissage).

### 2. Stochastic Gradient Descent (SGD)

Au lieu de calculer le gradient sur tout le dataset, on utilise des **mini-batches**.

**Algorithme** :
```
Pour chaque epoch:
    Mélanger les données
    Pour chaque mini-batch:
        1. Forward pass sur le batch
        2. Calcul de la loss
        3. Backward pass (calcul gradients)
        4. Mise à jour des paramètres
```

### 3. SGD avec Momentum

Accélère la convergence en accumulant les gradients précédents.

**Formule** :
```
v := β · v + (1 - β) · ∂L/∂W
W := W - α · v
```

Où `β` (momentum) est généralement 0.9.

### 4. Adam (Adaptive Moment Estimation)

Combine momentum et adaptation du learning rate.

**Formule** :
```
m := β₁ · m + (1 - β₁) · ∂L/∂W         (moment 1)
v := β₂ · v + (1 - β₂) · (∂L/∂W)²      (moment 2)

m_corrected := m / (1 - β₁ᵗ)
v_corrected := v / (1 - β₂ᵗ)

W := W - α · m_corrected / (√v_corrected + ε)
```

**Paramètres typiques** :
- α = 0.001
- β₁ = 0.9
- β₂ = 0.999
- ε = 10⁻⁸

---

## Architecture du Réseau

### Réseau Simple pour MNIST

```
┌─────────────────────────────────────────┐
│         INPUT LAYER (784)               │
│   Image 28×28 aplatie en vecteur        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      HIDDEN LAYER 1 (128 neurones)      │
│         Z¹ = W¹ · X + b¹                │
│         A¹ = ReLU(Z¹)                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      HIDDEN LAYER 2 (64 neurones)       │
│         Z² = W² · A¹ + b²               │
│         A² = ReLU(Z²)                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      OUTPUT LAYER (10 neurones)         │
│         Z³ = W³ · A² + b³               │
│         A³ = Softmax(Z³)                │
│    [P(0), P(1), ..., P(9)]              │
└─────────────────────────────────────────┘
```

### Dimensions des Matrices

Pour un batch de taille `B` :

| Couche | Poids W | Biais b | Input | Output |
|--------|---------|---------|-------|--------|
| Layer 1 | [128, 784] | [128, 1] | [784, B] | [128, B] |
| Layer 2 | [64, 128] | [64, 1] | [128, B] | [64, B] |
| Layer 3 | [10, 64] | [10, 1] | [64, B] | [10, B] |

### Nombre de Paramètres

```
Layer 1: 784 × 128 + 128 = 100,480 paramètres
Layer 2: 128 × 64 + 64   = 8,256 paramètres
Layer 3: 64 × 10 + 10    = 650 paramètres

TOTAL: 109,386 paramètres
```

---

## Processus d'Entraînement Complet

### Pseudo-code

```python
# Initialisation
W1, b1 = initialize_weights(128, 784)
W2, b2 = initialize_weights(64, 128)
W3, b3 = initialize_weights(10, 64)

learning_rate = 0.01
epochs = 20
batch_size = 64

for epoch in range(epochs):
    # Mélanger les données
    shuffle(train_data)

    for batch in get_batches(train_data, batch_size):
        X, Y = batch

        # === FORWARD PASS ===
        Z1 = W1 @ X + b1
        A1 = relu(Z1)

        Z2 = W2 @ A1 + b2
        A2 = relu(Z2)

        Z3 = W3 @ A2 + b3
        A3 = softmax(Z3)

        # Calcul de la loss
        loss = cross_entropy(A3, Y)

        # === BACKWARD PASS ===
        dZ3 = A3 - Y
        dW3 = dZ3 @ A2.T / batch_size
        db3 = np.sum(dZ3, axis=1, keepdims=True) / batch_size

        dA2 = W3.T @ dZ3
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = dZ2 @ A1.T / batch_size
        db2 = np.sum(dZ2, axis=1, keepdims=True) / batch_size

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = dZ1 @ X.T / batch_size
        db1 = np.sum(dZ1, axis=1, keepdims=True) / batch_size

        # === UPDATE ===
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Évaluation
    accuracy = evaluate(W1, b1, W2, b2, W3, b3, test_data)
    print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
```

---

## Techniques Avancées (À Implémenter)

### 1. Dropout

Désactive aléatoirement des neurones pendant l'entraînement pour éviter le surapprentissage.

```python
def dropout(A, keep_prob=0.8):
    mask = np.random.rand(*A.shape) < keep_prob
    return A * mask / keep_prob
```

### 2. Batch Normalization

Normalise les activations pour stabiliser l'apprentissage.

```python
def batch_norm(Z):
    mean = np.mean(Z, axis=0)
    std = np.std(Z, axis=0)
    return (Z - mean) / (std + 1e-8)
```

### 3. Weight Initialization (Xavier/He)

Initialisation intelligente pour éviter les gradients qui explosent ou disparaissent.

```python
# Xavier (pour sigmoid, tanh)
W = np.random.randn(n_out, n_in) * np.sqrt(1 / n_in)

# He (pour ReLU)
W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
```

### 4. Learning Rate Decay

Réduire progressivement le learning rate.

```python
learning_rate = initial_lr / (1 + decay_rate * epoch)
```

---

## Débogage et Validation

### Gradient Checking

Vérifier que la backpropagation est correcte en comparant avec le gradient numérique.

```python
# Gradient numérique (approximation)
epsilon = 1e-7
grad_numeric = (loss(W + epsilon) - loss(W - epsilon)) / (2 * epsilon)

# Gradient analytique (backprop)
grad_analytic = backprop(W)

# Vérification
difference = abs(grad_numeric - grad_analytic)
assert difference < 1e-7, "Gradient incorrect !"
```

### Signes d'un Bon Apprentissage

✅ **Bon** :
- Loss qui diminue progressivement
- Accuracy qui augmente sur train ET test
- Convergence stable

❌ **Problèmes** :
- Loss qui explose → Learning rate trop élevé
- Loss qui stagne → Learning rate trop faible, ou modèle trop simple
- Train accuracy élevée, test accuracy faible → Surapprentissage (overfitting)
- Loss = NaN → Gradient qui explose, mauvaise initialisation

---

## Références

- **Livre** : [Neural Networks and Deep Learning - Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- **Vidéos** : [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- **Cours** : [CS231n - Stanford](http://cs231n.stanford.edu/)
- **Paper** : [Backpropagation - Rumelhart et al. 1986](https://www.nature.com/articles/323533a0)

---

**Note** : Cette documentation est destinée à l'apprentissage. Pour approfondir, consultez les ressources ci-dessus et expérimentez avec le code !
