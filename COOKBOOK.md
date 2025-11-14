# 📚 COOKBOOK - Guide de Dépannage et Recettes

Guide pratique de troubleshooting et recettes pour résoudre les problèmes courants avec les réseaux de neurones.

---

## 🎯 Table des Matières

1. [Problèmes d'Apprentissage](#problèmes-dapprentissage)
2. [Problèmes de Performance](#problèmes-de-performance)
3. [Recettes Communes](#recettes-communes)
4. [Optimisation des Hyperparamètres](#optimisation-des-hyperparamètres)
5. [Checklist de Debugging](#checklist-de-debugging)

---

## 🔥 Problèmes d'Apprentissage

### ❌ Problème: La Loss ne Diminue Pas

**Symptômes:**
```
Epoch 1 - Loss: 2.3025
Epoch 2 - Loss: 2.3024
Epoch 3 - Loss: 2.3023
...
```

**Causes possibles et solutions:**

#### 1. **Learning Rate trop faible**

```python
# ❌ Mauvais
model = NeuralNetwork([784, 128, 10], learning_rate=0.00001)

# ✅ Bon
model = NeuralNetwork([784, 128, 10], learning_rate=0.01)
```

**Action:** Augmenter le learning rate (essayer 0.001, 0.01, 0.1)

#### 2. **Poids mal initialisés**

```python
# Vérifier l'initialisation
W = model.parameters['W1']
print(f"Mean: {W.mean():.6f}, Std: {W.std():.6f}")

# Devrait être proche de:
# Mean: ~0, Std: ~sqrt(2/n_input)
```

**Action:** Le code utilise déjà He initialization, mais vérifier que c'est bien appliqué.

#### 3. **Données non normalisées**

```python
# Vérifier les données
print(f"X_train - Min: {X_train.min()}, Max: {X_train.max()}")

# ✅ Devrait être [0, 1]
```

**Action:** Normaliser les données: `X = X / 255.0`

---

### ❌ Problème: La Loss Explose (NaN)

**Symptômes:**
```
Epoch 1 - Loss: 2.305
Epoch 2 - Loss: 156.4
Epoch 3 - Loss: nan
```

**Causes possibles et solutions:**

#### 1. **Learning Rate trop élevé**

```python
# ❌ Mauvais
model = NeuralNetwork([784, 128, 10], learning_rate=1.0)

# ✅ Bon
model = NeuralNetwork([784, 128, 10], learning_rate=0.01)
```

**Action:** Réduire le learning rate d'un facteur 10

#### 2. **Gradient Exploding**

```python
# Ajouter gradient clipping (si implémenté)
max_grad_norm = 5.0
for key in gradients:
    grad_norm = np.linalg.norm(gradients[key])
    if grad_norm > max_grad_norm:
        gradients[key] *= max_grad_norm / grad_norm
```

**Action:** Réduire le learning rate ou ajouter gradient clipping

---

### ❌ Problème: Surapprentissage (Overfitting)

**Symptômes:**
```
Train Accuracy: 0.995
Val Accuracy:   0.920
Test Accuracy:  0.918
```

**Causes et solutions:**

#### 1. **Réseau trop grand pour le dataset**

```python
# ❌ Trop de paramètres
model = NeuralNetwork([784, 1024, 512, 256, 128, 10])

# ✅ Plus raisonnable
model = NeuralNetwork([784, 256, 128, 10])
```

#### 2. **Pas assez de données d'entraînement**

**Solutions:**
- Utiliser data augmentation
- Augmenter le dataset
- Réduire la complexité du modèle

#### 3. **Trop d'époques**

```python
# Surveiller val_acc et arrêter quand elle stagne
# ou utiliser early stopping
```

---

### ❌ Problème: Sous-apprentissage (Underfitting)

**Symptômes:**
```
Train Accuracy: 0.850
Val Accuracy:   0.845
Test Accuracy:  0.843
```

**Solutions:**

#### 1. **Augmenter la capacité du réseau**

```python
# ❌ Trop petit
model = NeuralNetwork([784, 32, 10])

# ✅ Plus de capacité
model = NeuralNetwork([784, 256, 128, 64, 10])
```

#### 2. **Entraîner plus longtemps**

```python
# Augmenter le nombre d'époques
model.train(X_train, y_train, X_val, y_val, epochs=30)
```

#### 3. **Changer l'optimiseur**

```python
# ❌ SGD basique peut être trop lent
model = NeuralNetwork([784, 256, 128, 10], optimizer='sgd')

# ✅ Adam est souvent plus efficace
model = NeuralNetwork([784, 256, 128, 10], optimizer='adam')
```

---

## ⚡ Problèmes de Performance

### ❌ Problème: Entraînement Trop Lent

**Solutions:**

#### 1. **Augmenter la taille des batches**

```python
# ❌ Petits batches = beaucoup d'itérations
model.train(X_train, y_train, X_val, y_val, batch_size=16)

# ✅ Plus rapide
model.train(X_train, y_train, X_val, y_val, batch_size=128)
```

**Note:** Batch size trop grand peut réduire la généralisation.

#### 2. **Réduire la complexité du modèle**

```python
# Si le modèle est trop complexe pour vos besoins
# Commencer simple et augmenter si nécessaire
```

#### 3. **Réduire le nombre de features**

```python
# Pour MNIST, on utilise déjà tous les pixels (784)
# Pour d'autres datasets, considérer PCA ou feature selection
```

---

### ❌ Problème: Accuracy Plafonne à ~10%

**Cause:** Le modèle prédit toujours la même classe (ou prédit au hasard)

**Solutions:**

#### 1. **Vérifier l'architecture**

```python
# Vérifier qu'il y a bien 10 neurones en sortie pour MNIST
print(model.layer_dims)  # Devrait terminer par 10
```

#### 2. **Vérifier la fonction de loss**

```python
# S'assurer d'utiliser cross-entropy pour classification
```

#### 3. **Vérifier les labels**

```python
# Les labels doivent être 0-9, pas 1-10
print(f"Labels uniques: {np.unique(y_train)}")
```

---

## 🍳 Recettes Communes

### 📖 Recette 1: Entraîner un Modèle de Base

```python
from src.network import NeuralNetwork
from src.utils import load_mnist_data

# Charger les données
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_data()

# Créer le modèle
model = NeuralNetwork(
    layer_dims=[784, 256, 128, 10],
    learning_rate=0.01,
    optimizer='adam'
)

# Entraîner
model.train(X_train, y_train, X_val, y_val,
            epochs=15, batch_size=128, verbose=True)

# Évaluer
test_acc = model.accuracy(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Sauvegarder
model.save('models/my_model.pkl')
```

**Résultat attendu:** ~96-97% accuracy

---

### 📖 Recette 2: Optimiser les Hyperparamètres

```python
# Tester différentes configurations
configs = [
    {'layers': [784, 128, 10], 'lr': 0.01, 'opt': 'adam'},
    {'layers': [784, 256, 128, 10], 'lr': 0.01, 'opt': 'adam'},
    {'layers': [784, 512, 256, 10], 'lr': 0.005, 'opt': 'adam'},
]

results = []

for config in configs:
    model = NeuralNetwork(
        layer_dims=config['layers'],
        learning_rate=config['lr'],
        optimizer=config['opt']
    )

    model.train(X_train, y_train, X_val, y_val, epochs=10)

    val_acc = model.accuracy(X_val, y_val)
    results.append((config, val_acc))

# Trier par accuracy
results.sort(key=lambda x: x[1], reverse=True)
best_config, best_acc = results[0]
print(f"Meilleure config: {best_config} - Acc: {best_acc:.4f}")
```

---

### 📖 Recette 3: Créer un Ensemble de Modèles

```python
# Entraîner plusieurs modèles
models = []

for i in range(5):
    model = NeuralNetwork([784, 256, 128, 10], learning_rate=0.01, optimizer='adam')
    model.train(X_train, y_train, X_val, y_val, epochs=10, verbose=False)
    models.append(model)
    print(f"Model {i+1} - Val Acc: {model.accuracy(X_val, y_val):.4f}")

# Prédictions par vote
def ensemble_predict(models, X):
    all_preds = [model.predict(X) for model in models]
    # Vote majoritaire
    ensemble_preds = []
    for i in range(X.shape[0]):
        votes = [preds[i] for preds in all_preds]
        ensemble_preds.append(max(set(votes), key=votes.count))
    return np.array(ensemble_preds)

# Évaluer l'ensemble
y_pred_ensemble = ensemble_predict(models, X_test)
ensemble_acc = np.mean(y_pred_ensemble == y_test)
print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")
```

**Résultat attendu:** +1-2% vs modèle individuel

---

### 📖 Recette 4: Déboguer un Modèle qui Ne Converge Pas

```python
# 1. Vérifier les données
print("="*50)
print("VÉRIFICATION DES DONNÉES")
print("="*50)
print(f"X_train shape: {X_train.shape}")
print(f"X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
print(f"y_train shape: {y_train.shape}")
print(f"y_train unique: {np.unique(y_train)}")

# 2. Vérifier l'initialisation
model = NeuralNetwork([784, 128, 10], learning_rate=0.01)
print("\n" + "="*50)
print("VÉRIFICATION DE L'INITIALISATION")
print("="*50)
for key, param in model.parameters.items():
    print(f"{key}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")

# 3. Test sur un petit batch
print("\n" + "="*50)
print("TEST SUR PETIT BATCH")
print("="*50)
X_batch = X_train[:32]
y_batch = y_train[:32]

for epoch in range(10):
    # Forward
    A, cache = model.forward(X_batch)

    # Loss
    Y_batch = model.one_hot_encode(y_batch)
    loss = model.compute_loss(Y_batch, A)

    # Backward
    grads = model.backward(Y_batch, cache)

    # Update
    model.update_parameters(grads)

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

print("\n✓ Si la loss diminue sur ce petit batch, le modèle fonctionne !")
```

---

### 📖 Recette 5: Visualiser l'Apprentissage

```python
from src import visualize
from src.metrics import confusion_matrix

# Après entraînement
# 1. Courbes d'apprentissage
visualize.plot_training_history(model.history, save_path='training.png')

# 2. Matrice de confusion
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, num_classes=10)
visualize.plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)],
                                save_path='confusion.png')

# 3. Poids de la première couche
visualize.plot_weights_visualization(model.parameters['W1'], n_neurons=64,
                                    save_path='weights.png')

# 4. Exemples de prédictions
y_probs, _ = model.forward(X_test)
visualize.plot_sample_predictions(X_test, y_test, y_pred, y_probs,
                                  n_samples=25, save_path='predictions.png')

print("✓ Visualisations sauvegardées!")
```

---

## 🎛️ Optimisation des Hyperparamètres

### Learning Rate

| Valeur | Effet | Quand utiliser |
|--------|-------|----------------|
| 0.0001 | Très lent, stable | Réseaux très profonds |
| 0.001  | Lent, stable | Architecture complexe |
| 0.01   | **Bon défaut** | La plupart des cas |
| 0.1    | Rapide, instable | Réseaux simples, expérimentation |
| 1.0    | Trop rapide, diverge | ❌ Éviter |

**Recette:** Commencer à 0.01, diviser par 10 si ça diverge, multiplier par 10 si trop lent.

---

### Batch Size

| Taille | Avantages | Inconvénients |
|--------|-----------|---------------|
| 16-32  | Bonne généralisation | Très lent |
| 64-128 | **Bon compromis** | - |
| 256-512 | Très rapide | Moins bonne généralisation |
| 1024+  | Maximum de vitesse | Mauvaise généralisation |

**Règle d'or:** 128 est un bon défaut pour MNIST.

---

### Nombre de Couches

| Architecture | Capacité | Quand utiliser |
|-------------|----------|----------------|
| [784, 128, 10] | Faible | Baseline rapide |
| [784, 256, 128, 10] | **Moyenne** | **Recommandé** |
| [784, 512, 256, 128, 10] | Élevée | Dataset complexe |
| [784, 256, 128, 64, 32, 10] | Très élevée | Risque d'overfitting |

**Recette:** Commencer simple, augmenter si underfitting.

---

### Optimiseurs

| Optimiseur | Vitesse | Stabilité | Quand utiliser |
|-----------|---------|-----------|----------------|
| SGD | Lent | Stable | Baseline, compréhension |
| Momentum | Moyen | Stable | Alternative à Adam |
| **Adam** | **Rapide** | **Très stable** | **Par défaut** |

**Recommandation:** Toujours commencer avec Adam.

---

## ✅ Checklist de Debugging

### Avant l'Entraînement

- [ ] **Données normalisées** : X entre [0, 1]
- [ ] **Labels corrects** : y entre [0, 9]
- [ ] **Architecture valide** : Input=784, Output=10
- [ ] **Batch size raisonnable** : 64-128
- [ ] **Learning rate approprié** : 0.001-0.01
- [ ] **Train/val/test splits** : Pas de fuite de données

### Pendant l'Entraînement

- [ ] **Loss diminue** : Doit descendre progressivement
- [ ] **Accuracy augmente** : Sur train ET val
- [ ] **Pas de NaN** : Vérifier les valeurs
- [ ] **Val acc suit train acc** : Gap < 5%
- [ ] **Logs clairs** : Afficher les métriques

### Après l'Entraînement

- [ ] **Test accuracy raisonnable** : ~96-98% pour MNIST
- [ ] **Matrice de confusion** : Erreurs logiques ?
- [ ] **Pas de surapprentissage** : Train-Val gap < 5%
- [ ] **Reproductibilité** : Fixer le random seed
- [ ] **Modèle sauvegardé** : .pkl existe et charge correctement

---

## 🚨 Erreurs Communes et Solutions

### Erreur: "IndexError: index out of bounds"

**Cause:** Mauvaise dimension des données

```python
# Vérifier
print(f"X shape: {X.shape}")  # Devrait être (n, 784)
print(f"y shape: {y.shape}")  # Devrait être (n,)
```

---

### Erreur: "ValueError: operands could not be broadcast"

**Cause:** Incompatibilité de dimensions dans les calculs matriciels

```python
# Vérifier les dimensions
print(f"W shape: {W.shape}")
print(f"X shape: {X.shape}")

# S'assurer que les multiplications sont cohérentes
# Z = X @ W (n, d) @ (d, h) → (n, h)
```

---

### Erreur: "RuntimeWarning: overflow in exp"

**Cause:** Valeurs trop grandes dans softmax/sigmoid

**Solution:** Déjà géré par la soustraction du max dans softmax, mais vérifier le learning rate.

---

### Accuracy Reste à 10%

**Cause:** Le modèle prédit toujours la même classe

**Solution:**
1. Vérifier que les poids ne sont pas tous à zéro
2. Vérifier que le learning rate n'est pas trop faible
3. S'assurer que les labels sont corrects

---

## 💡 Astuces et Best Practices

### 1. Toujours Commencer Simple

```python
# ✅ Bon workflow
# 1. Baseline simple
model = NeuralNetwork([784, 128, 10], learning_rate=0.01, optimizer='adam')
model.train(X_train, y_train, X_val, y_val, epochs=5)

# 2. Si ça marche, complexifier
model = NeuralNetwork([784, 256, 128, 64, 10], learning_rate=0.01, optimizer='adam')
model.train(X_train, y_train, X_val, y_val, epochs=15)
```

### 2. Monitorer Pendant l'Entraînement

```python
# Afficher verbose=True au début
model.train(X_train, y_train, X_val, y_val, epochs=10, verbose=True)

# Observer:
# - Loss descend régulièrement ?
# - Val acc augmente avec train acc ?
# - Pas de plateau prématuré ?
```

### 3. Sauvegarder Souvent

```python
# Sauvegarder le meilleur modèle
best_val_acc = 0
for epoch in range(50):
    # ... entraînement ...
    val_acc = model.accuracy(X_val, y_val)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save('models/best_model.pkl')
        print(f"✓ Nouveau meilleur: {val_acc:.4f}")
```

### 4. Comparer Plusieurs Configurations

```python
# Utiliser le script benchmark
python benchmark.py
```

### 5. Visualiser pour Comprendre

```python
# Toujours créer des visualisations
python train.py --visualize
python evaluate.py --model models/best_model.pkl --visualize
```

---

## 📞 Support

Si vous rencontrez un problème non couvert ici :

1. Vérifier les notebooks (`notebooks/07_debugging_gradient_checking.ipynb`)
2. Consulter `CLAUDE.md` pour la documentation complète
3. Examiner les tests (`tests/`) pour des exemples
4. Utiliser le mode verbose pour plus de détails

---

**Bonne chance avec vos réseaux de neurones ! 🚀**
