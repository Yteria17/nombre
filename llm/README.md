# 🤖 LLM from Scratch - Apprendre à créer un Grand Modèle de Langage

## 🎯 Objectif

Construire un **Large Language Model** (type GPT) **from scratch** pour comprendre en profondeur comment fonctionnent ChatGPT, Claude, et autres LLMs modernes.

> ⚠️ **Note** : On va créer un **mini-LLM** éducatif (millions de paramètres), pas un modèle de production (milliards). L'objectif est la **compréhension**, pas la performance ultime.

---

## 🗺️ Parcours d'Apprentissage

### 📘 Phase 1 : Fondamentaux (Notebooks 00-03)

| # | Notebook | Concepts Clés | Durée |
|---|----------|---------------|-------|
| **00** | Introduction aux LLMs | Architecture Transformer, GPT vs BERT, Tokens | 30min |
| **01** | Tokenization | BPE, WordPiece, Vocabulaire, Encoding/Decoding | 1h |
| **02** | Embeddings | Word2Vec concepts, Embedding layers, Similarité | 1h |
| **03** | Attention Mechanism | Queries, Keys, Values, Scaled Dot-Product | 1h30 |

### 🏗️ Phase 2 : Architecture Transformer (Notebooks 04-07)

| # | Notebook | Concepts Clés | Durée |
|---|----------|---------------|-------|
| **04** | Multi-Head Attention | Parallel attention, Concatenation, Projections | 1h30 |
| **05** | Positional Encoding | Position information, Sinusoidal encoding | 1h |
| **06** | Transformer Block | LayerNorm, Residual connections, FFN | 1h30 |
| **07** | Architecture GPT Complète | Decoder-only, Causal masking, Full model | 2h |

### 🚀 Phase 3 : Entraînement et Génération (Notebooks 08-11)

| # | Notebook | Concepts Clés | Durée |
|---|----------|---------------|-------|
| **08** | Dataset et Preprocessing | Text corpus, Batching, Data loading | 1h |
| **09** | Training Loop | Loss calculation, Gradient descent, Monitoring | 2h |
| **10** | Text Generation | Greedy, Beam search, Top-k, Top-p, Temperature | 1h30 |
| **11** | Fine-tuning | Instruction tuning, RLHF concepts, Chat format | 2h |

### 🎓 Phase 4 : Projet Final (Notebook 12)

| # | Notebook | Description | Durée |
|---|----------|-------------|-------|
| **12** | **Mini-ChatGPT** | Projet complet end-to-end | 3h |

**Temps total estimé : ~20 heures**

---

## 🧠 Concepts Couverts

### Fondamentaux Mathématiques
- ✅ Produits matriciels et tenseurs
- ✅ Softmax et probabilités
- ✅ Fonctions de perte (Cross-Entropy)
- ✅ Backpropagation à travers le temps
- ✅ Optimisation (Adam, learning rate scheduling)

### Architecture Transformer
- ✅ **Self-Attention** : Comment le modèle "regarde" le contexte
- ✅ **Multi-Head Attention** : Plusieurs perspectives simultanées
- ✅ **Positional Encoding** : Intégrer l'ordre des mots
- ✅ **Feed-Forward Networks** : Transformations non-linéaires
- ✅ **Layer Normalization** : Stabiliser l'entraînement
- ✅ **Residual Connections** : Faciliter le gradient flow

### Training et Génération
- ✅ **Tokenization** : Transformer texte → nombres
- ✅ **Causal Masking** : Empêcher de "tricher" (regarder le futur)
- ✅ **Teacher Forcing** : Technique d'entraînement
- ✅ **Sampling Strategies** : Contrôler la créativité
- ✅ **Temperature** : Ajuster la diversité des réponses
- ✅ **Top-k / Top-p** : Filtrage intelligent des tokens

---

## 📁 Structure du Projet

```
llm/
├── README.md                      # Ce fichier
├── GUIDE.md                       # Guide de démarrage rapide
├── requirements.txt               # Dépendances Python
│
├── notebooks/                     # Notebooks d'apprentissage
│   ├── 00_introduction_llms.ipynb
│   ├── 01_tokenization.ipynb
│   ├── 02_embeddings.ipynb
│   ├── 03_attention_mechanism.ipynb
│   ├── 04_multi_head_attention.ipynb
│   ├── 05_positional_encoding.ipynb
│   ├── 06_transformer_block.ipynb
│   ├── 07_gpt_architecture.ipynb
│   ├── 08_dataset_preprocessing.ipynb
│   ├── 09_training_loop.ipynb
│   ├── 10_text_generation.ipynb
│   ├── 11_fine_tuning.ipynb
│   └── 12_mini_chatgpt_project.ipynb
│
├── src/                           # Code source
│   ├── __init__.py
│   ├── tokenizer.py              # BPE tokenizer
│   ├── embeddings.py             # Embedding layers
│   ├── attention.py              # Attention mechanisms
│   ├── transformer.py            # Transformer blocks
│   ├── model.py                  # Full GPT model
│   ├── training.py               # Training utilities
│   ├── generation.py             # Text generation
│   └── utils.py                  # Helper functions
│
├── data/                          # Datasets
│   ├── tiny_shakespeare.txt      # Dataset d'exemple
│   ├── vocab/                    # Vocabulaires
│   └── processed/                # Données preprocessées
│
├── models/                        # Modèles entraînés
│   └── checkpoints/              # Sauvegardes
│
├── tests/                         # Tests unitaires
│   ├── test_tokenizer.py
│   ├── test_attention.py
│   └── test_model.py
│
├── train.py                       # Script d'entraînement CLI
├── generate.py                    # Script de génération CLI
└── chat.py                        # Interface de chat interactive
```

---

## 🎯 Objectifs d'Apprentissage

À la fin de ce parcours, tu sauras :

1. ✅ **Comment fonctionne l'attention** (le cœur des LLMs)
2. ✅ **Pourquoi les Transformers sont révolutionnaires**
3. ✅ **Comment un texte est transformé en nombres** (tokenization)
4. ✅ **Comment les LLMs "comprennent" le contexte**
5. ✅ **Comment entraîner un modèle de langage**
6. ✅ **Comment générer du texte de qualité**
7. ✅ **Les différences entre GPT, BERT, T5**
8. ✅ **Comment fine-tuner pour des tâches spécifiques**

---

## 💻 Technologies Utilisées

### Phase 1-3 (Apprentissage)
- **NumPy** : Implémentation from scratch pour comprendre
- **Matplotlib** : Visualisations

### Phase 4 (Projet pratique)
- **PyTorch** : Framework moderne pour le projet final
- **Transformers (HuggingFace)** : Comparaison avec l'état de l'art

---

## 🚀 Quick Start

```bash
# Installation
cd llm
pip install -r requirements.txt

# Lancer le premier notebook
jupyter notebook notebooks/00_introduction_llms.ipynb

# Ou suivre l'ordre recommandé :
# 00 → 01 → 02 → 03 → ... → 12
```

---

## 📊 Mini-LLM : Spécifications

Le modèle final que nous allons construire :

| Paramètre | Valeur | Note |
|-----------|--------|------|
| **Vocabulaire** | ~5,000 tokens | Petit mais fonctionnel |
| **Embedding dimension** | 256 | Représentation de chaque token |
| **Nombre de layers** | 4-6 | GPT-3 en a 96 ! |
| **Attention heads** | 8 | Perspectives multiples |
| **Context window** | 128-256 tokens | GPT-4 va jusqu'à 128k |
| **Paramètres totaux** | ~10M | GPT-3 : 175B |
| **Dataset** | Tiny Shakespeare | ~1MB de texte |

**Performance attendue** :
- ✅ Génère du texte cohérent dans le style Shakespeare
- ✅ Complète des phrases correctement
- ✅ Peut être fine-tuné pour des tâches simples
- ❌ Ne rivalise PAS avec ChatGPT (objectif pédagogique)

---

## 🎓 Prérequis

### Connaissances
- ✅ Python (numpy, classes)
- ✅ Réseaux de neurones basiques (si tu as fait le projet `nombre/`, parfait !)
- ✅ Algèbre linéaire (matrices, vecteurs)
- ⚠️ **Pas besoin d'être expert** - tout est expliqué !

### Matériel
- **CPU suffit** pour les notebooks d'apprentissage
- **GPU recommandé** pour entraîner le modèle final (ou Google Colab gratuit)

---

## 📚 Ressources Complémentaires

- **Paper original** : "Attention Is All You Need" (Vaswani et al., 2017)
- **GPT Paper** : "Language Models are Unsupervised Multitask Learners"
- **Illustrated Transformer** : http://jalammar.github.io/illustrated-transformer/
- **Andrej Karpathy** : "Let's build GPT from scratch" (YouTube)

---

## 🗓️ Plan de Progression Recommandé

### Semaine 1 : Fondamentaux
- Jour 1-2 : Notebooks 00-01 (Intro, Tokenization)
- Jour 3-4 : Notebooks 02-03 (Embeddings, Attention)

### Semaine 2 : Architecture
- Jour 1-2 : Notebooks 04-05 (Multi-head, Positional)
- Jour 3-4 : Notebooks 06-07 (Transformer Block, GPT complet)

### Semaine 3 : Training
- Jour 1-2 : Notebooks 08-09 (Dataset, Training)
- Jour 3-4 : Notebooks 10-11 (Generation, Fine-tuning)

### Semaine 4 : Projet
- Jour 1-5 : Notebook 12 (Mini-ChatGPT complet)

**Ou à ton rythme ! C'est auto-guidé.**

---

## 🎯 Différences avec le Projet `nombre/`

| Aspect | Réseaux de Neurones | LLMs |
|--------|---------------------|------|
| **Complexité** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Architecture** | MLP, CNN | Transformer |
| **Input** | Images fixes | Séquences variables |
| **Output** | Classification | Génération de texte |
| **Mécanisme clé** | Convolution | Attention |
| **Dataset** | 60k images | Millions de tokens |

**Les LLMs sont BEAUCOUP plus complexes**, mais on va y aller étape par étape !

---

## 🤝 Contribution

Ce projet est éducatif. Suggestions et améliorations bienvenues !

---

**Prêt à construire ton propre mini-GPT ? Let's go ! 🚀**

*Commence par le notebook `00_introduction_llms.ipynb`*
