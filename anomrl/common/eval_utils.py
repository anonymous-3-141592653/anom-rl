from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from TSB_AD.evaluation.basic_metrics import generate_curve


def fpr_at_95_tpr(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    try:
        return fpr[np.where(tpr >= 0.95)[0][0]]
    except IndexError:
        return 1.0


def compute_vus(y_true, y_scores, slidingWindow=10, pred=None, version="opt", thre=500):
    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(y_true=y_true, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(y_true.astype(int), y_scores, slidingWindow, version, thre)
    return vus_roc, vus_pr


def compute_global_metrics(y_true_list, y_score_list):
    y_true_all = np.concatenate(y_true_list)
    y_score_all = np.concatenate(y_score_list)

    if len(np.unique(y_true_all)) < 2:
        # AUROC/AUPR undefined if only one class present
        return {"GLOB_AUROC": np.nan, "GLOB_AUPR": np.nan, "GLOB_FPR@95": np.nan}

    auroc = roc_auc_score(y_true_all, y_score_all)
    aupr = average_precision_score(y_true_all, y_score_all)
    fpr95 = fpr_at_95_tpr(y_true_all, y_score_all)

    return {"GLOB_AUROC": auroc, "GLOB_AUPR": aupr, "GLOB_FPR@95": fpr95}


def compute_local_metrics(y_true_list, y_score_list, vus=False):
    metrics = defaultdict(list)

    for y_true, y_score in zip(y_true_list, y_score_list):
        if len(np.unique(y_true)) < 2:
            # AUROC/AUPR undefined if only one class present
            continue
        metrics["AUROC"].append(roc_auc_score(y_true, y_score))
        metrics["AUPR"].append(average_precision_score(y_true, y_score))
        metrics["FPR95"].append(fpr_at_95_tpr(y_true, y_score))
        if vus:
            vus_roc, vus_pr = compute_vus(y_true, y_score.flatten())
            metrics["VUSROC"].append(vus_roc)
            metrics["VUSPR"].append(vus_pr)

    summary = {}
    for key, values in metrics.items():
        summary[key] = np.mean(values)
        summary[key + "_STD"] = np.std(values)

    return summary


def compute_thresholds(val_scores) -> dict:
    val_scores = np.concatenate(val_scores)
    mean = val_scores.mean()
    std = val_scores.std()
    return {
        "3sigma": mean + 3 * std,
        "95id": np.percentile(val_scores, 95),
        "max": val_scores.max(),
    }


def compute_detection_times(y_true_list, y_score_list, thresholds):
    """
    Compute detection times for each threshold.
    Args:
        y_true_list (list): List of true labels.
        y_score_list (list): List of predicted scores.
        thresholds (dict): Dictionary of thresholds.
    Returns:

    """
    times = {}
    for name, th in thresholds.items():
        det = np.array([np.argmax(x > th) if np.any(x > th) else len(x) for x in y_score_list])
        tru = np.array([np.argmax(x) if np.any(x) else len(x) for x in y_true_list])
        lag = det - tru
        times[name] = lag.tolist()
    return times
