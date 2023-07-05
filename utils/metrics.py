import numpy as np
import torch

__all__ = [
    'calculate_metrics', 'get_metrics', 'update_metrics', 'get_mean_and_standard_deviation', 'get_extreme_examples',
    'get_best_OD_examples', 'get_worst_OD_examples', 'get_best_OC_examples', 'get_worst_OC_examples',
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

    # Calculate individual metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    dice = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)

    return {
        'accuracy': accuracy,
        'precision': precision,  # PPV (Positive Predictive Value)
        'sensitivity': sensitivity,  # Recall, Hit-rate, TPR (True Positive Rate)
        'specificity': specificity,  # TNR (True Negative Rate)
        'dice': dice,  # F1 score
        'iou': iou,  # Jaccard index
    }


def get_metrics(true: torch.Tensor, pred: torch.Tensor, types: list) -> dict[str, float]:
    # Flatten the tensors to 1D
    true_flat = true.flatten().cpu().numpy()
    pred_flat = pred.flatten().cpu().numpy()

    # Get metrics separately for OD and OC, treating it as binary segmentation
    metrics_bg = calculate_metrics(true_flat, pred_flat, [0]) if [0] in types else {}
    metrics_od_ring = calculate_metrics(true_flat, pred_flat, [1]) if [1] in types else {}
    metrics_od = calculate_metrics(true_flat, pred_flat, [1, 2]) if [1, 2] in types else {}
    metrics_oc = calculate_metrics(true_flat, pred_flat, [2]) if [2] in types else {}

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
                   types: list, prefix: str = '', postfix: str = '') -> dict[str, float]:
    new = get_metrics(true, pred, types)

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
    return get_extreme_examples(*args, **kwargs, best=True, class_ids=[1, 2])


def get_worst_OD_examples(*args, **kwargs):
    return get_extreme_examples(*args, **kwargs, best=False, class_ids=[1, 2])


def get_best_OC_examples(*args, **kwargs):
    return get_extreme_examples(*args, **kwargs, best=True, class_ids=[2])


def get_worst_OC_examples(*args, **kwargs):
    return get_extreme_examples(*args, **kwargs, best=False, class_ids=[2])


def get_extreme_examples(model, loader, n: int = 5, best: bool = True, class_ids: list = None,
                         device: str = 'cuda', metric: str = 'iou'):
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

            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Convert to numpy arrays and transpose the images to (H, W, C) format
            images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
            preds = preds.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            for i, _ in enumerate(images):
                score = calculate_metrics(masks[i], preds[i], class_ids)[metric]
                examples.append((images[i], masks[i], preds[i], score))

            # Sort the examples (ascending or descending) based on their scores
            examples.sort(key=lambda x: x[-1], reverse=best)
            # Keep only the top n examples
            examples = examples[:n]

    return examples
