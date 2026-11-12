"""Métricas de classificação (entrega final): acurácia, precisão, revocação, F1 e matriz de confusão."""
import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


warnings.filterwarnings("ignore", module="sklearn")


def compute_metrics(y, pred, classes):
    y, pred = np.asarray(y), np.asarray(pred)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    _, _, fw, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted", zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(set(y)) > 1 else float("nan"),
        "macro_precision": float(pm), "macro_recall": float(rm), "macro_f1": float(fm),
        "weighted_f1": float(fw),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }
