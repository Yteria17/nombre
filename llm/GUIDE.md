# 🚀 Guide de Démarrage - LLM from Scratch

## 🎯 Bienvenue !

Ce guide te permettra de **construire ton propre Large Language Model (comme GPT) from scratch** et de comprendre en profondeur comment fonctionnent ChatGPT, Claude, et autres LLMs modernes.

---

## 📚 Parcours d'Apprentissage Complet

### 🌟 Phase 1 : Fondamentaux (4-6 heures)

Avant de construire le Transformer, il faut comprendre les briques de base.

#### **Notebook 00 - Introduction aux LLMs** ✅ (30 min)
**Fichier** : `notebooks/00_introduction_llms.ipynb`

**Tu vas apprendre** :
- Qu'est-ce qu'un LLM ?
- L'histoire : RNN → LSTM → Transformer (2017)
- Pourquoi l'attention est révolutionnaire
- GPT vs BERT vs T5
- Notre objectif : mini-GPT (~10M paramètres)

**Concepts clés** :
- Transformer architecture
- Autoregressive generation
- Attention mechanism (aperçu)

---

#### **Notebook 01 - Tokenization** ✅ (1h)
**Fichier** : `notebooks/01_tokenization.ipynb`

**Tu vas apprendre** :
- Pourquoi tokenizer ? (réseaux = nombres seulement)
- 3 approches : Character, Word, **Subword (BPE)**
- Implémentation complète d'un tokenizer BPE
- Encode/Decode

**Implémentation** :
```python
class SimpleBPETokenizer:
    def train(corpus)  # Apprendre le vocabulaire
    def encode(text)   # Texte → IDs
    def decode(ids)    # IDs → Texte
```

**Résultat** :
```python
"Bonjour le monde" → [145, 298, 1023]
```

---

#### **Notebook 02 - Embeddings** ✅ (1h)
**Fichier** : `notebooks/02_embeddings.ipynb`

**Tu vas apprendre** :
- Le problème des IDs bruts (pas de sémantique)
- Embeddings = vecteurs denses qui capturent le sens
- Similarité cosinus
- Word2Vec (Skip-gram, CBOW)

**Implémentation** :
```python
class EmbeddingLayer:
    def forward(token_ids)   # IDs → Vecteurs
    def backward(gradients)  # Backprop
```

**Résultat** :
```python
"chat"  → [0.2, -0.5, 0.8, ...]  # 256D
"chien" → [0.3, -0.4, 0.7, ...]  # Proche !
```

---

#### **Notebook 03 - Attention Mechanism** 🔄 (1h30)
**Fichier** : `notebooks/03_attention_mechanism.ipynb` (À venir)

**Tu vas apprendre** :
- Le cœur du Transformer : **Attention**
- Queries, Keys, Values (Q, K, V)
- Scaled Dot-Product Attention
- Implémentation from scratch

**Formule clé** :
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**Résultat** :
- Comprendre comment le modèle "regarde" le contexte
- Visualiser les scores d'attention

---

### 🏗️ Phase 2 : Architecture Transformer (6-8 heures)

Maintenant qu'on a les bases, construisons le Transformer !

#### **Notebook 04 - Multi-Head Attention** 🔄
Plusieurs "têtes" d'attention en parallèle

#### **Notebook 05 - Positional Encoding** 🔄
Intégrer l'information de position (ordre des mots)

#### **Notebook 06 - Transformer Block** 🔄
Assembler : Attention + Feed-Forward + LayerNorm + Residual

#### **Notebook 07 - Architecture GPT Complète** 🔄
Le modèle final : Embedding → N×TransformerBlock → Output

---

### 🚀 Phase 3 : Entraînement et Génération (6-8 heures)

#### **Notebook 08 - Dataset et Preprocessing** 🔄
Préparer Tiny Shakespeare pour l'entraînement

#### **Notebook 09 - Training Loop** 🔄
Entraîner le modèle (forward, backward, update)

#### **Notebook 10 - Text Generation** 🔄
Générer du texte : Greedy, Top-k, Top-p, Temperature

#### **Notebook 11 - Fine-tuning** 🔄
Adapter le modèle à des tâches spécifiques

---

### 🎓 Phase 4 : Projet Final (3-4 heures)

#### **Notebook 12 - Mini-ChatGPT Project** 🔄
Projet complet end-to-end avec interface de chat

---

## 🛠️ Installation et Setup

### Prérequis

- Python 3.8+
- Jupyter Notebook
- Connaissances de base en Python et NumPy

### Installation

```bash
# Cloner le repo
cd llm/

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook
```

### Ouvrir le premier notebook

```bash
jupyter notebook notebooks/00_introduction_llms.ipynb
```

---

## 📖 Comment Utiliser ce Cours

### 1️⃣ **Approche Linéaire** (Recommandée pour débutants)

Suis les notebooks dans l'ordre :

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12
```

**Temps total** : ~20 heures

**Rythme suggéré** :
- Semaine 1 : Notebooks 00-03 (Fondamentaux)
- Semaine 2 : Notebooks 04-07 (Architecture)
- Semaine 3 : Notebooks 08-11 (Training)
- Semaine 4 : Notebook 12 (Projet final)

### 2️⃣ **Approche par Objectifs**

Tu veux juste comprendre un concept spécifique ?

- **Comprendre l'attention** → Notebooks 00, 03, 04
- **Tokenization** → Notebooks 01
- **Entraîner un modèle** → Notebooks 08, 09
- **Générer du texte** → Notebooks 10

### 3️⃣ **Approche Pratique** (Pour les expérimentés)

Tu connais déjà la théorie et veux coder ?

1. Lis rapidement les notebooks 00-02
2. Code avec les notebooks 03-07 (architecture)
3. Projet final (notebook 12)

---

## 💡 Conseils d'Apprentissage

### ✅ À Faire

1. **Exécute TOUT le code** : Ne te contente pas de lire
2. **Expérimente** : Modifie les paramètres, observe les résultats
3. **Prends des notes** : Écris ce que tu comprends
4. **Fais les exercices** : Ils renforcent la compréhension
5. **Visualise** : Les graphiques aident énormément

### ❌ À Éviter

1. ❌ Ne saute pas les notebooks : Ils sont progressifs
2. ❌ Ne te décourage pas : Les LLMs sont complexes, c'est normal
3. ❌ Ne copie pas aveuglément : Comprends chaque ligne
4. ❌ Ne t'arrête pas aux erreurs : Debug et apprends

### 🎯 Objectifs d'Apprentissage

À la fin de ce cours, tu sauras :

✅ Comment fonctionne l'attention (le cœur des LLMs)
✅ Pourquoi les Transformers sont révolutionnaires
✅ Comment tokenizer du texte (BPE)
✅ Comment les embeddings capturent la sémantique
✅ Comment construire un Transformer from scratch
✅ Comment entraîner un modèle de langage
✅ Comment générer du texte de qualité
✅ Les différences entre GPT, BERT, T5

---

## 🧪 Compétences Acquises

### Niveau Débutant (Notebooks 00-03)

- Comprendre ce qu'est un LLM
- Tokenizer du texte
- Créer des embeddings
- Le mécanisme d'attention de base

### Niveau Intermédiaire (Notebooks 04-07)

- Multi-head attention
- Positional encoding
- Construire un Transformer block
- Assembler un modèle GPT complet

### Niveau Avancé (Notebooks 08-12)

- Préparer un dataset
- Entraîner un LLM
- Générer du texte avec stratégies variées
- Fine-tuner pour des tâches spécifiques
- Projet complet end-to-end

---

## 📊 Comparaison : Ce que tu vas construire

| Paramètre | Notre Mini-GPT | GPT-2 | GPT-3 |
|-----------|----------------|-------|-------|
| **Vocabulaire** | ~5,000 | 50,257 | 50,257 |
| **Embedding dim** | 256 | 768 | 12,288 |
| **Layers** | 4-6 | 12-48 | 96 |
| **Attention heads** | 8 | 12-16 | 96 |
| **Paramètres** | **~10M** | 117M-1.5B | **175B** |
| **Dataset** | Tiny Shakespeare | WebText | 45TB texte |
| **Entraînement** | Minutes (CPU) | Heures (GPU) | Semaines (cluster) |
| **Performance** | Style Shakespeare | Texte cohérent | ChatGPT level |

**Notre objectif** : Comprendre les concepts, pas rivaliser avec ChatGPT ! 🎓

---

## 🗂️ Structure du Projet

```
llm/
├── README.md                      # Vue d'ensemble
├── GUIDE.md                       # Ce fichier - Guide de démarrage
├── requirements.txt               # Dépendances
│
├── notebooks/                     # Notebooks d'apprentissage
│   ├── ✅ 00_introduction_llms.ipynb
│   ├── ✅ 01_tokenization.ipynb
│   ├── ✅ 02_embeddings.ipynb
│   ├── 🔄 03_attention_mechanism.ipynb      (À venir)
│   ├── 🔄 04_multi_head_attention.ipynb     (À venir)
│   ├── 🔄 05_positional_encoding.ipynb      (À venir)
│   ├── 🔄 06_transformer_block.ipynb        (À venir)
│   ├── 🔄 07_gpt_architecture.ipynb         (À venir)
│   ├── 🔄 08_dataset_preprocessing.ipynb    (À venir)
│   ├── 🔄 09_training_loop.ipynb            (À venir)
│   ├── 🔄 10_text_generation.ipynb          (À venir)
│   ├── 🔄 11_fine_tuning.ipynb              (À venir)
│   └── 🔄 12_mini_chatgpt_project.ipynb     (À venir)
│
├── src/                           # Code source (À implémenter)
│   ├── __init__.py
│   ├── tokenizer.py              # BPE tokenizer
│   ├── embeddings.py             # Embedding layer
│   ├── attention.py              # Attention mechanisms
│   ├── transformer.py            # Transformer blocks
│   ├── model.py                  # Full GPT model
│   ├── training.py               # Training utilities
│   ├── generation.py             # Text generation
│   └── utils.py                  # Helper functions
│
├── data/                          # Datasets
│   └── tiny_shakespeare.txt      # Dataset d'exemple
│
├── models/                        # Modèles entraînés
│   └── checkpoints/              # Sauvegardes
│
└── tests/                         # Tests unitaires
    └── test_*.py
```

---

## 📚 Ressources Complémentaires

### Papers Fondamentaux

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - LE paper du Transformer
   - https://arxiv.org/abs/1706.03762

2. **"Language Models are Unsupervised Multitask Learners"** (GPT-2)
   - https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

3. **"Language Models are Few-Shot Learners"** (GPT-3)
   - https://arxiv.org/abs/2005.14165

### Tutoriels Recommandés

- **The Illustrated Transformer** (Jay Alammar)
  - http://jalammar.github.io/illustrated-transformer/
  - Visualisations excellentes

- **The Annotated Transformer** (Harvard NLP)
  - http://nlp.seas.harvard.edu/annotated-transformer/
  - Code ligne par ligne avec PyTorch

- **Andrej Karpathy - "Let's build GPT"**
  - YouTube : Construction step-by-step
  - https://www.youtube.com/watch?v=kCc8FmEb1nY

### Livres

- **"Speech and Language Processing"** (Jurafsky & Martin)
  - Gratuit en ligne
- **"Natural Language Processing with Transformers"** (Tunstall et al.)
  - Livre pratique avec Hugging Face

---

## 🆘 Aide et Support

### Tu es bloqué ?

1. **Relis le notebook** : La réponse est souvent là
2. **Vérifie les erreurs** : Lis les messages d'erreur
3. **Debug** : Utilise `print()` pour comprendre
4. **Compare** : Vérifie avec le code du notebook
5. **Expérimente** : Teste sur un exemple simple

### Ressources de Debug

```python
# Vérifier les shapes
print(f"Shape: {tensor.shape}")

# Vérifier les valeurs
print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")

# Vérifier les NaN
print(f"Has NaN: {np.isnan(tensor).any()}")
```

---

## 🎯 Prochaines Étapes

### Tu as terminé les 3 premiers notebooks ?

✅ **Félicitations !** Tu as compris les fondamentaux :
- Tokenization (texte → nombres)
- Embeddings (nombres → vecteurs riches)

### Continue avec :

1. **Notebook 03 - Attention** ← LE concept le plus important !
2. Puis les notebooks 04-07 (architecture)
3. Puis les notebooks 08-11 (training)
4. Enfin le projet final (12)

---

## 💬 Feedback et Contributions

Ce projet est éducatif et open-source. Les suggestions d'amélioration sont bienvenues !

---

## 🎓 Citation

Si tu utilises ce cours pour apprendre ou enseigner, mentionne :

```
"LLM from Scratch - Cours éducatif complet pour comprendre
les Large Language Models (GPT) en construisant from scratch"
```

---

## ⭐ Philosophie du Cours

> "Si tu ne peux pas le construire from scratch,
> tu ne le comprends pas vraiment."

Ce cours te fait construire **chaque composant** d'un LLM pour que tu comprennes **vraiment** comment ça marche.

---

**Bon apprentissage ! 🚀**

**Commence maintenant** → Ouvre `notebooks/00_introduction_llms.ipynb`

---

## 🗓️ Changelog

- **2025-01-14** : Création du projet LLM
  - ✅ README complet
  - ✅ Notebook 00 - Introduction
  - ✅ Notebook 01 - Tokenization
  - ✅ Notebook 02 - Embeddings
  - 🔄 Notebooks 03-12 en cours de création
