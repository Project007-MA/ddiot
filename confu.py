from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.metrics import confusion_matrix
import numpy as np

# Metric calculation function
def metric(TP, TN, FP, FN, ln, alpha=None, beta=None, cond=False):
    # Adjust for conditional scaling if needed
    if cond:
        TN /= ln ** 1
        FP /= ln ** alpha
        FN /= ln ** beta

    # Sensitivity, Specificity, Precision, etc.
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = sensitivity  # same as sensitivity
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    e_measure = 1 - (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Rand_index = accuracy ** 0.5

    # MCC calculation
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0

    # Other metrics
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0  # Negative Predictive Value
    fdr = FP / (FP + TP) if (FP + TP) > 0 else 0  # False Discovery Rate

    # Return metrics as a dictionary
    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f_measure': f_measure,
        'accuracy': accuracy,
        #'mcc': mcc,
        'fpr': fpr,
        'fnr': fnr,
        'npv': npv,
        'fdr': fdr
    }

    # Return the values in list form
    metrics_list = [accuracy, precision, sensitivity, specificity, f_measure, npv, fpr, fnr]

    return metrics_list

# Multilabel confusion matrix function
def multi_confu_matrix(Y_test, Y_pred, *args):
    cm = mcm(Y_test, Y_pred)
    ln = len(cm)
    TN, FP, FN, TP = 0, 0, 0, 0

    # Aggregate all the confusion matrices for each label
    for i in range(ln):
        TN += cm[i][0][0]
        FP += cm[i][0][1]
        FN += cm[i][1][0]
        TP += cm[i][1][1]

    # Pass to metric function
    return metric(TP, TN, FP, FN, ln, *args)

# Binary confusion matrix function
def confu_matrix(Y_test, Y_pred, *args):
    cm = confusion_matrix(Y_test, Y_pred)
    TN, FP = cm[0][0], cm[0][1]
    FN, TP = cm[1][0], cm[1][1]
    ln = len(cm)

    # Pass to metric function
    return metric(TP, TN, FP, FN, ln, *args)
