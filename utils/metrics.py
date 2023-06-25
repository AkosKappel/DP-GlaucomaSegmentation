import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score


def calculate_metrics(tp, tn, fp, fn):
    return {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),  # ACC
        'precision': tp / (tp + fp),  # PPV
        'sensitivity': tp / (tp + fn),  # recall, hit rate, TPR
        'specificity': tn / (tn + fp),  # TNR
        'dice': 2 * tp / (2 * tp + fp + fn),  # F1 score
        'iou': tp / (tp + fp + fn)  # Jaccard index
    }


def get_performance_metrics(truth, prediction, average: str = 'macro',
                            combined_only: bool = True, num_classes: int = 3):
    # Flatten the tensors to 1D
    truth_flat = truth.flatten().numpy()
    prediction_flat = prediction.flatten().numpy()

    # Compute metrics for the all classes combined (averaged)
    metrics = {'combined': {
        'accuracy': accuracy_score(truth_flat, prediction_flat),
        'precision': precision_score(truth_flat, prediction_flat, average=average, zero_division=0),
        'sensitivity': recall_score(truth_flat, prediction_flat, average=average, zero_division=0),
        'specificity': get_specificity_score(truth_flat, prediction_flat, n_classes=num_classes),
        'dice': f1_score(truth_flat, prediction_flat, average=average, zero_division=0),
        'iou': jaccard_score(truth_flat, prediction_flat, average=average, zero_division=0),
    }}

    # Return only the combined metrics of all classes
    if combined_only:
        return metrics['combined']

    # Binary metrics for OD (including OC as part of OD)
    od_truth_flat = (truth_flat != 0).astype(np.uint8)
    od_prediction_flat = (prediction_flat != 0).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(od_truth_flat, od_prediction_flat).ravel()
    metrics['OD'] = calculate_metrics(tp, tn, fp, fn)

    # Binary metrics for OC
    oc_truth_flat = (truth_flat == 2).astype(np.uint8)
    oc_prediction_flat = (prediction_flat == 2).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(oc_truth_flat, oc_prediction_flat).ravel()
    metrics['OC'] = calculate_metrics(tp, tn, fp, fn)

    # Compute the confusion matrix
    conf_mat = confusion_matrix(truth_flat, prediction_flat)
    num_classes = conf_mat.shape[0]

    # Compute metrics for each class separately as if it were binary
    for class_id in range(num_classes):
        tp = conf_mat[class_id, class_id]
        fp = conf_mat[:, class_id].sum() - tp
        fn = conf_mat[class_id, :].sum() - tp
        tn = conf_mat.sum() - tp - fp - fn
        metrics[class_id] = calculate_metrics(tp, tn, fp, fn)

    return metrics


def get_confusion_matrix(truth, prediction, n_classes=3):
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int32)
    for cls in range(n_classes):
        pred = prediction == cls
        for c in range(n_classes):
            conf_mat[cls, c] = (pred & (truth == c)).sum()
    return conf_mat


def get_accuracy_score(truth, prediction):
    return (prediction == truth).mean()


def get_precision_score(truth, prediction, n_classes=3):
    def get_precision(truth, prediction, cls):
        pred = prediction == cls
        mask = truth == cls
        tp = (pred & mask).sum()
        fp = (pred & (~mask)).sum()
        return tp / (tp + fp)

    # Binary
    if n_classes == 2:
        return get_precision(truth, prediction, 1)

    # Multi-class
    precisions = []
    for cls in range(n_classes):
        precision = get_precision(truth, prediction, cls)
        precisions.append(precision)
    return np.mean(precisions)


def get_sensitivity_score(truth, prediction, n_classes=3):
    def get_sensitivity(truth, prediction, cls):
        pred = prediction == cls
        mask = truth == cls
        tp = (pred & mask).sum()
        fn = (~pred & mask).sum()
        return tp / (tp + fn)

    # Binary
    if n_classes == 2:
        return get_sensitivity(truth, prediction, 1)

    # Multi-class
    recalls = []
    for cls in range(n_classes):
        recall = get_sensitivity(truth, prediction, cls)
        recalls.append(recall)
    return np.mean(recalls)


def get_specificity_score(truth, prediction, n_classes=3):
    def get_specificity(truth, prediction, cls):
        pred = prediction == cls
        mask = truth == cls
        tn = (~pred & ~mask).sum()
        fp = (~pred & mask).sum()
        return tn / (tn + fp)

    # Binary
    if n_classes == 2:
        return get_specificity(truth, prediction, 1)

    # Multi-class
    specificities = []
    for cls in range(n_classes):
        specificity = get_specificity(truth, prediction, cls)
        specificities.append(specificity)
    return np.mean(specificities)


def iou_score(truth, prediction, smooth=1e-6):
    """
    Intersection over Union (IoU) / Jaccard Index

    IoU = TP / (TP + FP + FN) = GT ∩ Pred / (GT ∪ Pred)

    The parameters are expected to contain boolean True (object) and False (background) values
    for each pixel. This version of the IoU score can be used for binary segmentation problems.
    """
    intersection = (truth & prediction).sum()
    union = (truth | prediction).sum()
    return (intersection + smooth) / (union + smooth)


def mean_iou_score(truth, prediction, smooth=1e-6, n_classes=3):
    """
    Mean Intersection over Union (mIoU) / Mean Jaccard Index

    MeanIoU = IoU_1 + IoU_2 + ... + IoU_n / n

    The parameters are expected to contain integer class labels for each pixel. The labels
    are assumed to be  in the range [0, n_classes - 1]. This version of the IoU score can
    be used for multi-class segmentation problems.
    """
    ious = []
    for cls in range(n_classes):
        pred = prediction == cls
        mask = truth == cls
        iou = iou_score(mask, pred, smooth)
        ious.append(iou)
    return np.mean(ious)


def dice_score(truth, prediction, smooth=1e-6):
    """
    Dice Coefficient

    DSC = 2 * TP / (2 * TP + FP + FN) = 2 * GT ∩ Pred / (GT + Pred)

    The parameters are expected to contain boolean True (object) and False (background) values
    for each pixel. This version of the dice score can be used for binary segmentation problems.
    """
    intersection = (truth & prediction).sum()
    prediction_sum = prediction.sum()
    truth_sum = truth.sum()
    return (2 * intersection + smooth) / (prediction_sum + truth_sum + smooth)


def mean_dice_score(truth, prediction, smooth=1e-6, n_classes=3):
    """
    Mean Dice Coefficient

    MeanDSC = (DSC_1 + DSC_2 + ... + DSC_n) / n

    The parameters are expected to contain integer class labels for each pixel. The labels
    are assumed to be in the range [0, n_classes - 1]. This version of the dice score can
    be used for multi-class segmentation problems.
    """
    dices = []
    for cls in range(n_classes):
        pred = prediction == cls
        mask = truth == cls
        dice = dice_score(mask, pred, smooth)
        dices.append(dice)
    return np.mean(dices)
