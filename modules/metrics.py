import numpy as np
import torch
from scipy.ndimage import rotate

__all__ = [
    'safe_division', 'calculate_diameter', 'calculate_vCDR', 'calculate_hCDR', 'calculate_aCDR',
    'calculate_disc_height', 'calculate_disc_width', 'calculate_disc_area',
    'calculate_cup_height', 'calculate_cup_width', 'calculate_cup_area',
    'calculate_metrics', 'get_tp_tn_fp_fn', 'get_metrics', 'update_metrics', 'get_extreme_examples',
    'get_best_OD_examples', 'get_worst_OD_examples', 'get_best_and_worst_OD_examples',
    'get_best_OC_examples', 'get_worst_OC_examples', 'get_best_and_worst_OC_examples',
]


def safe_division(a: int | float, b: int | float) -> float:
    return a / b if b != 0 else 0


def calculate_diameter(image: np.ndarray, label: int | list[int], angle: int = 0) -> int:
    image = np.array(image)
    label = np.array(label) if isinstance(label, (list, np.ndarray)) else np.array([label])

    # Rotate the image (angle = 0 is vertical diameter, angle = 90 is horizontal diameter)
    rotated_image = rotate(image, angle, reshape=True, order=0, mode='constant', cval=0)

    object_mask = np.isin(rotated_image, label)
    if not np.any(object_mask):
        return 0

    object_row_indices = np.where(np.any(object_mask, axis=1))[0]

    if object_row_indices.size == 0:
        return 0
    return object_row_indices[-1] - object_row_indices[0] + 1


def calculate_vCDR(mask: np.ndarray, disc_label: int = 1, cup_label: int = 2) -> float:
    disc_diameter = calculate_diameter(mask, [disc_label], angle=0)
    cup_diameter = calculate_diameter(mask, [cup_label], angle=0)
    return safe_division(cup_diameter, disc_diameter)


def calculate_hCDR(mask: np.ndarray, disc_label: int = 1, cup_label: int = 2) -> float:
    disc_diameter = calculate_diameter(mask, [disc_label], angle=90)
    cup_diameter = calculate_diameter(mask, [cup_label], angle=90)
    return safe_division(cup_diameter, disc_diameter)


def calculate_aCDR(mask: np.ndarray, disc_label: int = 1, cup_label: int = 2) -> float:
    cup_area = np.sum(mask == cup_label)
    disc_area = np.sum(mask == disc_label) + cup_area
    return safe_division(cup_area, disc_area)


def calculate_disc_height(mask: np.ndarray, disc_label: int = 1) -> int:
    return calculate_diameter(mask, [disc_label], angle=0)


def calculate_disc_width(mask: np.ndarray, disc_label: int = 1) -> int:
    return calculate_diameter(mask, [disc_label], angle=90)


def calculate_disc_area(mask: np.ndarray, disc_label: int = 1, cup_label: int = 2) -> int:
    return np.sum(mask == disc_label) + np.sum(mask == cup_label)


def calculate_cup_height(mask: np.ndarray, cup_label: int = 2) -> int:
    return calculate_diameter(mask, [cup_label], angle=0)


def calculate_cup_width(mask: np.ndarray, cup_label: int = 2) -> int:
    return calculate_diameter(mask, [cup_label], angle=90)


def calculate_cup_area(mask: np.ndarray, cup_label: int = 2) -> int:
    return np.sum(mask == cup_label)


# Calculate individual metrics from TP, TN, FP, FN
def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    accuracy = safe_division(tp + tn, tp + tn + fp + fn)
    precision = safe_division(tp, tp + fp)  # PPV (Positive Predictive Value)
    npv = safe_division(tn, tn + fn)  # NPV (Negative Predictive Value)
    sensitivity = safe_division(tp, tp + fn)  # TPR (True Positive Rate), Recall, Hit-rate
    specificity = safe_division(tn, tn + fp)  # TNR (True Negative Rate), Selectivity
    fpr = safe_division(fp, fp + tn)  # FPR (False Positive Rate), Fall-out
    fnr = safe_division(fn, tp + fn)  # FNR (False Negative Rate), Miss Rate
    dice = safe_division(2 * tp, 2 * tp + fp + fn)  # F1 score
    iou = safe_division(tp, tp + fp + fn)  # Jaccard index
    balanced_accuracy = safe_division(sensitivity + specificity, 2)
    # f1 = safe_division(2 * precision * sensitivity, precision + sensitivity)
    # informedness = specificity + sensitivity - 1
    # prevalence = safe_division(tp + fn, tp + tn + fp + fn)
    # fdr = safe_division(fp, tp + fp)  # False Discovery Rate
    # _for = 1 - safe_division(tn, tn + fn)  # False Omission Rate
    # lr_pos = safe_division(sensitivity, fpr)  # PLR (Positive Likelihood Ratio)
    # lr_neg = safe_division(fnr, specificity)  # NLR (Negative Likelihood Ratio)
    # dor = safe_division(lr_pos, lr_neg)  # DOR (Diagnostic Odds Ratio)
    # voe = 1 - iou  # Volume Overlap Error
    # rvd = safe_division(fp - fn, tp + fn)  # Relative Volume Difference
    return {
        'accuracy': accuracy,
        'precision': precision,
        'npv': npv,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,
        'fnr': fnr,
        'dice': dice,
        'iou': iou,
        'balanced_accuracy': balanced_accuracy,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def get_tp_tn_fp_fn(gt: np.ndarray, sr: np.ndarray, class_ids: list[int]) -> tuple[int, int, int, int]:
    # Flatten the arrays if they are not 1D
    if gt.ndim > 1:
        gt = gt.flatten()
    if sr.ndim > 1:
        sr = sr.flatten()

    # Binarize masks - since OC is always inside OD, we can use >= instead of == for extracting
    # the masks of individual classes. For example, in the masks, the binary_labels are as follows:
    # 0 = BG, 1 = OD, 2 = OC
    # but when creating the binary mask we accept:
    # OD = 1 or 2, OC = 2
    gt = np.isin(gt, class_ids)
    sr = np.isin(sr, class_ids)

    # GT = Ground Truth, SR = Segmentation Region
    # |GT| = tp + fn, |SR| = tp + fp, |GT ∩ SR| = tp, |GT ∪ SR| = tp + fp + fn
    tp = (gt & sr).sum()
    tn = (~gt & ~sr).sum()
    fp = (~gt & sr).sum()
    fn = (gt & ~sr).sum()

    # True Positives, True Negatives, False Positives, False Negatives
    return tp, tn, fp, fn


def get_metrics(true: torch.Tensor | np.ndarray, pred: torch.Tensor | np.ndarray,
                labels: list = None) -> dict[str, float]:
    if labels is None:
        labels = [[1, 2], [2]]

    # Flatten the tensors to 1D shape
    true_flat = true.flatten()
    pred_flat = pred.flatten()

    # Convert to numpy arrays if they are tensors
    if isinstance(true, torch.Tensor):
        true_flat = true_flat.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred_flat = pred_flat.detach().cpu().numpy()

    def _get_metrics(gt: np.ndarray, sr: np.ndarray, class_ids: list[int]) -> dict[str, float]:
        tp, tn, fp, fn = get_tp_tn_fp_fn(gt, sr, class_ids)
        return calculate_metrics(tp, tn, fp, fn)

    # Get metrics separately for OD and OC, treating it as binary segmentation
    metrics_bg = _get_metrics(true_flat, pred_flat, [0]) if [0] in labels else {}
    metrics_nrr = _get_metrics(true_flat, pred_flat, [1]) if [1] in labels else {}
    metrics_od = _get_metrics(true_flat, pred_flat, [1, 2]) if [1, 2] in labels else {}
    metrics_oc = _get_metrics(true_flat, pred_flat, [2]) if [2] in labels else {}

    # Combine the metrics and add suffix to the keys
    return {
        # background
        **{k + '_BG': v for k, v in metrics_bg.items()},
        # only the neuro-retinal rim is used for evaluation (excluding the overlap with optic cup)
        **{k + '_NRR': v for k, v in metrics_nrr.items()},
        # entire optic disc is used for evaluation (including the inner overlap with optic cup)
        **{k + '_OD': v for k, v in metrics_od.items()},
        # optic cup
        **{k + '_OC': v for k, v in metrics_oc.items()},
    }


def update_metrics(true: torch.Tensor, pred: torch.Tensor, old: dict[str, list[float]],
                   labels: list, prefix: str = '', postfix: str = '') -> dict[str, float]:
    new = get_metrics(true, pred, labels)

    # Insert the new metrics at the end of the list in the old dictionary
    for key, value in new.items():
        k = prefix + key + postfix
        if k not in old:
            old[k] = []
        old[k].append(value)

    return new


def get_best_OD_examples(*args, **kwargs):
    class_ids = [1, 2] if 'binary_labels' not in kwargs else kwargs['binary_labels']
    return get_extreme_examples(*args, **kwargs, best=True, worst=False, class_ids=class_ids)


def get_worst_OD_examples(*args, **kwargs):
    class_ids = [1, 2] if 'binary_labels' not in kwargs else kwargs['binary_labels']
    return get_extreme_examples(*args, **kwargs, best=False, worst=True, class_ids=class_ids)


def get_best_and_worst_OD_examples(*args, **kwargs):
    class_ids = [1, 2] if 'binary_labels' not in kwargs else kwargs['binary_labels']
    return get_extreme_examples(*args, **kwargs, best=True, worst=True, class_ids=class_ids)


def get_best_OC_examples(*args, **kwargs):
    class_ids = [2] if 'binary_labels' not in kwargs else kwargs['binary_labels']
    return get_extreme_examples(*args, **kwargs, best=True, worst=False, class_ids=class_ids)


def get_worst_OC_examples(*args, **kwargs):
    class_ids = [2] if 'binary_labels' not in kwargs else kwargs['binary_labels']
    return get_extreme_examples(*args, **kwargs, best=False, worst=True, class_ids=class_ids)


def get_best_and_worst_OC_examples(*args, **kwargs):
    class_ids = [2] if 'binary_labels' not in kwargs else kwargs['binary_labels']
    return get_extreme_examples(*args, **kwargs, best=True, worst=True, class_ids=class_ids)


def get_extreme_examples(model, loader, n, best: bool = True, worst: bool = True, class_ids: list = None,
                         device: torch.device = None, metric: str = 'iou', thresh: float = 0.5, softmax: bool = False,
                         out_idx: int = None, first_model=None):
    """
    Returns the best/worst segmentation examples of a model from a given data loader based on a specified metric.
    The examples are returned as a list of (image, mask, prediction, score) tuples. The metric for determining the
    correctness of the segmentation can be one of the following: 'iou', 'dice', 'accuracy', 'precision',
    'sensitivity', 'specificity'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if metric not in ('iou', 'dice', 'accuracy', 'precision', 'sensitivity', 'specificity'):
        raise ValueError(f'Invalid metric: {metric}')

    if class_ids is None:
        class_ids = [1, 2]

    model = model.to(device)
    model.eval()  # Don't forget to set the model to evaluation mode (e.g. disable dropout)

    examples = []
    with torch.no_grad():  # Disable gradient calculation
        for images, masks in loader:
            images = images.float().to(device)
            masks = masks.long().to(device)

            # Apply the first model to the images when cascading
            if first_model is not None:
                first_model.eval()
                outputs = first_model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > thresh).long()
                images = images * preds

            # Forward pass
            outputs = model(images)

            # Select specific output from models with dual outputs
            if out_idx is not None:
                outputs = outputs[out_idx]

            if outputs.shape[1] == 1:
                # Binary segmentation
                probs = torch.sigmoid(outputs)
                preds = (probs > thresh).squeeze(1).long()

                # Convert the predictions to correct binary labels in ground truth format
                if class_ids == [1, 2]:
                    masks[masks == 2] = 1  # turn OC labels to OD binary labels
                elif class_ids == [1]:
                    masks[masks == 2] = 0  # hide OC binary labels
                elif class_ids == [2]:
                    preds[preds == 1] = 2  # change predicted positive binary labels to OC labels
                    masks[masks == 1] = 0  # hide OD from ground truth
            elif softmax:
                # Multi-class segmentation
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            else:
                # Multi-label segmentation
                probs = torch.sigmoid(outputs)
                preds = torch.zeros_like(probs[:, 0])
                for i in range(1, probs.shape[1]):
                    preds += (probs[:, i] > thresh).long()

            # Convert to numpy arrays and transpose the images to (H, W, C) format
            images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            masks = masks.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            for i, _ in enumerate(images):
                tp, tn, fp, fn = get_tp_tn_fp_fn(masks[i], preds[i], class_ids)
                score = calculate_metrics(tp, tn, fp, fn)[metric]
                examples.append((images[i], masks[i], preds[i], score))

        # Sort the examples based on their scores
        examples.sort(key=lambda x: x[-1])

        # Keep only the top n best/worst examples
        best_examples = examples[-n:] if best else []
        worst_examples = examples[:n] if worst else []

    # Return the best/worst examples
    if best and worst:
        return best_examples, worst_examples
    elif best:
        return best_examples
    elif worst:
        return worst_examples
