import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve


def eer_and_threshold(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold, fpr, tpr


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_0_bonafide": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0])},
        "class_1_spoof": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f1[1])},
    }
