"""
Métriques d'évaluation pour les modèles
"""

import numpy as np


def accuracy(Y_true, Y_pred):
    """
    Calcule l'accuracy (pourcentage de prédictions correctes)

    Args:
        Y_true: vrais labels (shape: (n,) ou (n, classes))
        Y_pred: prédictions (shape: (n,) ou (n, classes))

    Returns:
        acc: accuracy entre 0 et 1
    """
    # Si one-hot encoded, prendre argmax
    if Y_true.ndim > 1 and Y_true.shape[1] > 1:
        Y_true = np.argmax(Y_true, axis=1)
    if Y_pred.ndim > 1 and Y_pred.shape[1] > 1:
        Y_pred = np.argmax(Y_pred, axis=1)

    return np.mean(Y_true == Y_pred)


def confusion_matrix(Y_true, Y_pred, num_classes=10):
    """
    Calcule la matrice de confusion

    Args:
        Y_true: vrais labels
        Y_pred: prédictions
        num_classes: nombre de classes

    Returns:
        matrix: matrice de confusion (num_classes, num_classes)
    """
    # Convertir en labels si nécessaire
    if Y_true.ndim > 1:
        Y_true = np.argmax(Y_true, axis=1)
    if Y_pred.ndim > 1:
        Y_pred = np.argmax(Y_pred, axis=1)

    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(Y_true, Y_pred):
        matrix[true, pred] += 1

    return matrix


def precision(Y_true, Y_pred, average='macro'):
    """
    Calcule la précision

    Precision = TP / (TP + FP)
    """
    cm = confusion_matrix(Y_true, Y_pred)
    num_classes = cm.shape[0]

    precisions = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp

        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)

    if average == 'macro':
        return np.mean(precisions)
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        return np.average(precisions, weights=weights)
    else:
        return precisions


def recall(Y_true, Y_pred, average='macro'):
    """
    Calcule le recall (sensibilité)

    Recall = TP / (TP + FN)
    """
    cm = confusion_matrix(Y_true, Y_pred)
    num_classes = cm.shape[0]

    recalls = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp

        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)

    if average == 'macro':
        return np.mean(recalls)
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        return np.average(recalls, weights=weights)
    else:
        return recalls


def f1_score(Y_true, Y_pred, average='macro'):
    """
    Calcule le F1-score (moyenne harmonique de précision et recall)

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    prec = precision(Y_true, Y_pred, average=None)
    rec = recall(Y_true, Y_pred, average=None)

    f1_scores = []
    for p, r in zip(prec, rec):
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0.0)

    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'weighted':
        cm = confusion_matrix(Y_true, Y_pred)
        weights = np.sum(cm, axis=1)
        return np.average(f1_scores, weights=weights)
    else:
        return f1_scores


def top_k_accuracy(Y_true, Y_pred_probs, k=5):
    """
    Calcule la top-k accuracy

    Args:
        Y_true: vrais labels
        Y_pred_probs: probabilités prédites (n, classes)
        k: nombre de top prédictions à considérer

    Returns:
        acc: top-k accuracy
    """
    if Y_true.ndim > 1:
        Y_true = np.argmax(Y_true, axis=1)

    # Prendre les k meilleures prédictions
    top_k_preds = np.argsort(Y_pred_probs, axis=1)[:, -k:]

    # Vérifier si le vrai label est dans les top-k
    correct = 0
    for true_label, preds in zip(Y_true, top_k_preds):
        if true_label in preds:
            correct += 1

    return correct / len(Y_true)


def classification_report(Y_true, Y_pred, class_names=None):
    """
    Génère un rapport de classification complet

    Returns:
        report: dictionnaire avec toutes les métriques
    """
    if class_names is None:
        num_classes = len(np.unique(Y_true))
        class_names = [f"Class {i}" for i in range(num_classes)]

    cm = confusion_matrix(Y_true, Y_pred, len(class_names))
    prec = precision(Y_true, Y_pred, average=None)
    rec = recall(Y_true, Y_pred, average=None)
    f1 = f1_score(Y_true, Y_pred, average=None)

    report = {
        'confusion_matrix': cm,
        'overall_accuracy': accuracy(Y_true, Y_pred),
        'macro_precision': np.mean(prec),
        'macro_recall': np.mean(rec),
        'macro_f1': np.mean(f1),
        'per_class': {}
    }

    for i, name in enumerate(class_names):
        report['per_class'][name] = {
            'precision': prec[i],
            'recall': rec[i],
            'f1_score': f1[i],
            'support': np.sum(cm[i, :])
        }

    return report


def print_classification_report(Y_true, Y_pred, class_names=None):
    """
    Affiche un rapport de classification formaté
    """
    report = classification_report(Y_true, Y_pred, class_names)

    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(f"\nOverall Accuracy: {report['overall_accuracy']:.4f}")
    print(f"Macro Precision:  {report['macro_precision']:.4f}")
    print(f"Macro Recall:     {report['macro_recall']:.4f}")
    print(f"Macro F1-Score:   {report['macro_f1']:.4f}")

    print("\n" + "-"*70)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("-"*70)

    for class_name, metrics in report['per_class'].items():
        print(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics['support']}")

    print("="*70 + "\n")
