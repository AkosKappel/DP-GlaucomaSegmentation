import numpy as np
import torch

__all__ = [
    'calculate_metrics', 'get_metrics', 'update_metrics', 'get_mean_and_standard_deviation', 'get_extreme_examples',
    'get_best_OD_examples', 'get_worst_OD_examples', 'get_best_and_worst_OD_examples',
    'get_best_OC_examples', 'get_worst_OC_examples', 'get_best_and_worst_OC_examples',
]


def calculate_metrics(true: np.ndarray, pred: np.ndarray, class_ids: list[int]) -> dict[str, float]:
    # Binarize masks - since OC is always inside OD, we can use >= instead of == for extracting
    # the masks of individual classes. For example, in the masks, the labels are as follows:
    # 0 = BG, 1 = OD, 2 = OC
    # but when creating the binary mask we accept:
    # OD = 1 or 2, OC = 2
    true = np.isin(true, class_ids)
    pred = np.isin(pred, class_ids)

    # True Positives, True Negatives, False Positives, False Negatives
    tp = (true & pred).sum()
    tn = (~true & ~pred).sum()
    fp = (~true & pred).sum()
    fn = (true & ~pred).sum()

    # GT = Ground Truth, SR = Segmentation Region
    # |GT| = tp + fn, |SR| = tp + fp, |GT ∩ SR| = tp, |GT ∪ SR| = tp + fp + fn

    def safe_division(a, b):
        return a / b if b != 0 else 0

    # Calculate individual metrics
    accuracy = safe_division(tp + tn, tp + tn + fp + fn)
    precision = safe_division(tp, tp + fp)
    sensitivity = safe_division(tp, tp + fn)
    specificity = safe_division(tn, tn + fp)
    dice = safe_division(2 * tp, 2 * tp + fp + fn)
    iou = safe_division(tp, tp + fp + fn)
    balance_accuracy = safe_division(sensitivity + specificity, 2)
    # informedness = specificity + sensitivity - 1
    # prevalence = safe_division(tp + fn, tp + tn + fp + fn)
    # f1 = safe_division(2 * precision * sensitivity, precision + sensitivity)
    # npv = safe_division(tn, tn + fn)  # Negative Predictive Value
    # fpr = safe_division(fp, fp + tn)  # False Positive Rate
    # fnr = safe_division(fn, tp + fn)  # False Negative Rate
    # fdr = safe_division(fp, tp + fp)  # False Discovery Rate
    # _for = 1 - safe_division(tn, tn + fn)  # False Omission Rate
    # lr_pos = safe_division(sensitivity, fpr)  # Positive Likelihood Ratio
    # lr_neg = safe_division(fnr, specificity)  # Negative Likelihood Ratio
    # dor = safe_division(lr_pos, lr_neg)  # Diagnostic Odds Ratio
    # voe = 1 - iou  # Volume Overlap Error
    # rvd = safe_division(fp - fn, tp + fn)  # Relative Volume Difference

    return {
        'accuracy': accuracy,
        'precision': precision,  # PPV (Positive Predictive Value)
        'sensitivity': sensitivity,  # Recall, Hit-rate, TPR (True Positive Rate)
        'specificity': specificity,  # Selectivity, TNR (True Negative Rate)
        'dice': dice,  # F1 score
        'iou': iou,  # Jaccard index
        'balance_accuracy': balance_accuracy,
    }


def get_metrics(true: torch.Tensor | np.ndarray, pred: torch.Tensor | np.ndarray, labels: list) -> dict[str, float]:
    # Flatten the tensors to 1D
    true_flat = true.flatten()
    pred_flat = pred.flatten()

    # Convert to numpy arrays if they are tensors
    if isinstance(true, torch.Tensor):
        true_flat = true_flat.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred_flat = pred_flat.detach().cpu().numpy()

    # Get metrics separately for OD and OC, treating it as binary segmentation
    metrics_bg = calculate_metrics(true_flat, pred_flat, [0]) if [0] in labels else {}
    metrics_od_ring = calculate_metrics(true_flat, pred_flat, [1]) if [1] in labels else {}
    metrics_od = calculate_metrics(true_flat, pred_flat, [1, 2]) if [1, 2] in labels else {}
    metrics_oc = calculate_metrics(true_flat, pred_flat, [2]) if [2] in labels else {}

    # Combine the metrics and add suffix to the keys
    return {
        # background
        **{k + '_BG': v for k, v in metrics_bg.items()},
        # only the outer ring of optic disc is considered (not the overlap with optic cup)
        **{k + '_OD_ring': v for k, v in metrics_od_ring.items()},
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


def get_mean_and_standard_deviation(loader):
    """
    Calculate the mean and standard deviation of a dataset. The values are calculated per channel
    across all images. The images should be just from the training set, not the entire dataset to
    avoid data leakage.
    """
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for images, _ in loader:
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def get_best_OD_examples(*args, **kwargs):
    class_ids = [1, 2] if 'class_ids' not in kwargs else kwargs['class_ids']
    return get_extreme_examples(*args, **kwargs, best=True, worst=False, class_ids=class_ids)


def get_worst_OD_examples(*args, **kwargs):
    class_ids = [1, 2] if 'class_ids' not in kwargs else kwargs['class_ids']
    return get_extreme_examples(*args, **kwargs, best=False, worst=True, class_ids=class_ids)


def get_best_and_worst_OD_examples(*args, **kwargs):
    class_ids = [1, 2] if 'class_ids' not in kwargs else kwargs['class_ids']
    return get_extreme_examples(*args, **kwargs, best=True, worst=True, class_ids=class_ids)


def get_best_OC_examples(*args, **kwargs):
    class_ids = [2] if 'class_ids' not in kwargs else kwargs['class_ids']
    return get_extreme_examples(*args, **kwargs, best=True, worst=False, class_ids=class_ids)


def get_worst_OC_examples(*args, **kwargs):
    class_ids = [2] if 'class_ids' not in kwargs else kwargs['class_ids']
    return get_extreme_examples(*args, **kwargs, best=False, worst=True, class_ids=class_ids)


def get_best_and_worst_OC_examples(*args, **kwargs):
    class_ids = [2] if 'class_ids' not in kwargs else kwargs['class_ids']
    return get_extreme_examples(*args, **kwargs, best=True, worst=True, class_ids=class_ids)


def get_extreme_examples(model, loader, n, best: bool = True, worst: bool = True, class_ids: list = None,
                         device: str = 'cuda', metric: str = 'iou', thresh: float = 0.5, softmax: bool = False,
                         out_idx: int = None, first_model=None):
    """
    Returns the best/worst segmentation examples of a model from a given data loader based on a specified metric.
    The examples are returned as a list of (image, mask, prediction, score) tuples. The metric for determining the
    correctness of the segmentation can be one of the following: 'iou', 'dice', 'accuracy', 'precision',
    'sensitivity', 'specificity'.
    """
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

                # Convert the predictions to correct labels in ground truth format
                if class_ids == [1, 2]:
                    masks[masks == 2] = 1  # turn OC labels to OD labels
                elif class_ids == [1]:
                    masks[masks == 2] = 0  # hide OC labels
                elif class_ids == [2]:
                    preds[preds == 1] = 2  # change predicted positive labels to OC labels
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
                score = calculate_metrics(masks[i], preds[i], class_ids)[metric]
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
