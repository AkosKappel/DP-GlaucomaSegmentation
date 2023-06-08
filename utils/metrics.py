import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score


def get_performance_metrics(truth, prediction, average='macro', total_only=True):
    # Flatten the tensors to 1D
    truth_flat = truth.flatten()
    prediction_flat = prediction.flatten()

    # Compute the confusion matrix
    conf_mat = confusion_matrix(truth_flat, prediction_flat)
    num_classes = conf_mat.shape[0]

    # Compute metrics for the all classes combined
    metrics = {'total': {
        'accuracy': accuracy_score(truth_flat, prediction_flat),
        'precision': precision_score(truth_flat, prediction_flat, average=average, zero_division=0),
        'sensitivity': recall_score(truth_flat, prediction_flat, average=average),
        'specificity': get_specificity(truth, prediction, n_classes=num_classes),
        'dice': f1_score(truth_flat, prediction_flat, average=average),
        'iou': jaccard_score(truth_flat, prediction_flat, average=average),
    }}

    if total_only:
        return metrics['total']

    # Compute metrics for each class
    for class_id in range(num_classes):
        tp = conf_mat[class_id, class_id]
        fp = conf_mat[class_id, :].sum() - tp
        fn = conf_mat[:, class_id].sum() - tp
        tn = conf_mat.sum() - tp - fp - fn

        # Compute metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)  # PPV
        sensitivity = tp / (tp + fn)  # recall, hit rate, TPR
        specificity = tn / (tn + fp)  # TNR
        dice = 2 * tp / (2 * tp + fp + fn)  # F1 score
        iou = tp / (tp + fp + fn)

        # Add metrics to the dictionary
        metrics[class_id] = {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'dice': dice,
            'iou': iou,
        }

    return metrics


def get_confusion_matrix(truth, prediction, n_classes=3):
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int32)
    for cls in range(n_classes):
        pred = prediction == cls
        for c in range(n_classes):
            conf_mat[cls, c] = (pred & (truth == c)).float().sum().item()
    return conf_mat


def get_accuracy(truth, prediction):
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    The parameters are expected to be tensors of shape (batch_size, height, width)
    containing integer class labels for each pixel.
    """
    return (prediction == truth).float().mean().item()


def get_precision(truth, prediction, n_classes=3):
    """
    Precision = TP / (TP + FP)

    The parameters are expected to be tensors of shape (batch_size, height, width)
    containing integer class labels for each pixel.
    """
    precisions = []
    for cls in range(n_classes):
        pred = prediction == cls
        mask = truth == cls
        tp = (pred & mask).float().sum()
        fp = (pred & (~mask)).float().sum()
        precision = tp / (tp + fp)
        precisions.append(precision)
    return np.mean(precisions)


def get_sensitivity(truth, prediction, n_classes=3):
    """
    Recall = TP / (TP + FN)

    The parameters are expected to be tensors of shape (batch_size, height, width)
    containing integer class labels for each pixel.
    """
    recalls = []
    for cls in range(n_classes):
        pred = prediction == cls
        mask = truth == cls
        tp = (pred & mask).float().sum()
        fn = (~pred & mask).float().sum()
        recall = tp / (tp + fn)
        recalls.append(recall)
    return np.mean(recalls)


def get_specificity(truth, prediction, n_classes=3):
    """
    Specificity = TN / (TN + FP)

    The parameters are expected to be tensors of shape (batch_size, height, width)
    containing integer class labels for each pixel.
    """
    specificities = []
    for cls in range(n_classes):
        pred = prediction == cls
        mask = truth == cls
        tn = (~pred & ~mask).float().sum()
        fp = (~pred & mask).float().sum()
        specificity = tn / (tn + fp)
        specificities.append(specificity)
    return np.mean(specificities)


def iou_score(truth, prediction, smooth=1e-6):
    """
    Intersection over Union (IoU) / Jaccard Index

    IoU = TP / (TP + FP + FN) = GT ∩ Pred / (GT ∪ Pred)

    The parameters are expected to be tensors of shape (batch_size, height, width)
    containing boolean True (object) and False (background) values for each pixel.
    This version of the IoU score can be used for binary segmentation problems.
    """
    intersection = (prediction & truth).float().sum((1, 2))
    union = (prediction | truth).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)  # Add epsilon to prevent division by zero
    return iou.mean().item()


def mean_iou_score(truth, prediction, smooth=1e-6, n_classes=3):
    """
    Mean Intersection over Union (mIoU) / Mean Jaccard Index

    MeanIoU = IoU_1 + IoU_2 + ... + IoU_n / n

    The parameters are expected to be tensors of shape (batch_size, height, width)
    containing integer class labels for each pixel. The labels are assumed to be
    in the range [0, n_classes - 1]. This version of the IoU score can be used
    for multi-class segmentation problems.
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

    The parameters are expected to be tensors of shape (batch_size, height, width)
    containing boolean True (object) and False (background) values for each pixel.
    This version of the dice score can be used for binary segmentation problems.
    """
    intersection = (prediction & truth).float().sum((1, 2))
    prediction_area = prediction.float().sum((1, 2))
    truth_area = truth.float().sum((1, 2))
    dice = (2. * intersection + smooth) / (prediction_area + truth_area + smooth)
    return dice.mean().item()


def mean_dice_score(truth, prediction, smooth=1e-6, n_classes=3):
    """
    Mean Dice Coefficient

    MeanDSC = (DSC_1 + DSC_2 + ... + DSC_n) / n

    The parameters are expected to be tensors of shape (batch_size, height, width)
    containing integer class labels for each pixel. The labels are assumed to be
    in the range [0, n_classes - 1]. This version of the dice score can be used
    for multi-class segmentation problems.
    """
    dices = []
    for cls in range(n_classes):
        pred = prediction == cls
        mask = truth == cls
        dice = dice_score(mask, pred, smooth)
        dices.append(dice)
    return np.mean(dices)
