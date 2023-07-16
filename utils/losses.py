import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DiceLoss', 'GeneralizedDice', 'IoULoss', 'FocalLoss', 'TverskyLoss', 'FocalTverskyLoss',
           'BoundaryLoss', 'HausdorffLoss', 'CrossEntropyLoss', 'SensitivitySpecificityLoss', 'EdgeLoss', 'ComboLoss']


def logits_to_probs(logits: torch.Tensor, num_classes: int, dim: int = 1) -> torch.Tensor:
    """Convert logits from a model to probabilities by applying softmax or sigmoid activation function."""
    # logits.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    return F.softmax(logits, dim=dim) if num_classes > 1 else F.sigmoid(logits)


def probs_to_labels(probs: torch.Tensor, thresh: int = None, dim: int = 1) -> torch.Tensor:
    """Convert class probabilities to labels by choosing the class with the highest probability."""
    # probs.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, height, width)
    return torch.argmax(probs, dim=dim) if thresh is None else (probs > thresh).squeeze(dim).long()


def labels_to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer class label to one-hot encoding."""
    # labels.shape = (batch_size, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    # One-hot encode targets (e.g. 1 -> [0, 1, 0], 2 -> [0, 0, 1]) and move channel dimension to second position or
    # add channel dimension if the data is binary (e.g. (batch_size, height, width) -> (batch_size, 1, height, width))
    return F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float() \
        if num_classes > 1 else labels.unsqueeze(1).float()


def onehot_to_labels(onehot: torch.Tensor, num_classes: int, dim: int = 1) -> torch.Tensor:
    """Convert one-hot encoding to a class label."""
    # onehot.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, height, width)
    return torch.argmax(onehot, dim=dim) if num_classes > 1 else onehot.squeeze().long()


def probs_to_onehot(probs: torch.Tensor, num_classes: int, thresh: int = None, dim: int = 1) -> torch.Tensor:
    """Convert probabilities from a model to one-hot encoding."""
    # probs.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    return labels_to_onehot(probs_to_labels(probs, thresh=thresh, dim=dim), num_classes=num_classes)


# Soft Dice Loss for binary or multi-class segmentation
# soft means that we use probabilities instead of 0/1 predictions for getting the intersection and union
class DiceLoss(nn.Module):

    def __init__(self, num_classes: int, class_weights: list = None, smooth: float = 1e-7):
        super(DiceLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes  # 1 for binary segmentation, > 1 for multiclass segmentation
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch_size, num_classes, height, width)
        # targets.shape = (batch_size, height, width)

        probabilities = logits_to_probs(logits, self.num_classes)
        targets = labels_to_onehot(targets, self.num_classes)

        assert probabilities.shape == targets.shape, \
            f'Probabilities shape {probabilities.shape} does not match targets shape {targets.shape}'

        # Calculate intersection and union between predicted probabilities and targets
        intersection = (probabilities * targets).sum(dim=(2, 3))
        union = (probabilities + targets).sum(dim=(2, 3))

        # Calculate Dice coefficients for each class per each image in the batch
        dice_coeffs = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average across the batch dimension
        dice_coeffs = dice_coeffs.mean(dim=0)

        # Calculate Dice loss for each class (1 - Dice coefficient)
        dice_losses = 1 - dice_coeffs

        # Apply class weights if provided
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            dice_losses *= self.class_weights

        # Average loss across all classes
        return dice_losses.mean()


class GeneralizedDice(nn.Module):

    def __init__(self, num_classes: int, class_weights: list = None, smooth: float = 1e-7):
        super(GeneralizedDice, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes  # 1 for binary segmentation, > 1 for multiclass segmentation
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch_size, num_classes, height, width)
        # targets.shape = (batch_size, height, width)

        probabilities = logits_to_probs(logits, self.num_classes)
        targets = labels_to_onehot(targets, self.num_classes)

        assert probabilities.shape == targets.shape, \
            f'Probabilities shape {probabilities.shape} does not match targets shape {targets.shape}'

        # Calculate weights, intersection and union between predicted probabilities and targets
        weights = 1 / ((targets.sum(dim=(2, 3)) + self.smooth) ** 2)
        intersection = weights * (probabilities * targets).sum(dim=(2, 3))
        union = weights * (probabilities + targets).sum(dim=(2, 3))

        # Calculate Generalized Dice coefficients for each class per each image in the batch
        dice_coeffs = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average across the batch dimension
        dice_coeffs = dice_coeffs.mean(dim=0)

        # Calculate Dice loss for each class (1 - Dice coefficient)
        dice_losses = 1 - dice_coeffs

        # Apply class weights if provided
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            dice_losses *= self.class_weights

        # Average loss across all classes
        return dice_losses.mean()


class IoULoss(nn.Module):

    def __init__(self, num_classes: int, class_weights: list = None, smooth: float = 1e-7):
        super(IoULoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits.shape = (batch_size, num_classes, height, width)
        # targets.shape = (batch_size, height, width)

        probabilities = logits_to_probs(logits, self.num_classes)
        targets = labels_to_onehot(targets, self.num_classes)

        # Compute intersection and union
        intersection = (probabilities * targets).sum(dim=(2, 3))
        union = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection

        # Compute IoU score
        iou_scores = (intersection + self.smooth) / (union + self.smooth)

        # Compute mean across all batches
        iou_scores = iou_scores.mean(dim=0)

        # Compute loss for each class (1 - IoU score)
        iou_losses = 1 - iou_scores

        # Apply class weights if provided
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            iou_losses *= self.class_weights

        # Average across all classes
        return iou_losses.mean()


class FocalLoss(nn.Module):

    def __init__(self, num_classes: int, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes  # 1 for binary classification
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: = (batch_size, height, width)

        probabilities = logits_to_probs(logits, self.num_classes)
        targets_onehot = labels_to_onehot(targets, self.num_classes)

        if self.num_classes > 1:
            # Compute categorical cross entropy loss
            ce_loss = F.cross_entropy(logits, targets, reduction='none')

            # Calculate Focal loss
            pt = (probabilities * targets_onehot).sum(dim=1)
            focal_weights = torch.pow((1 - pt), self.gamma)
            focal_loss = self.alpha * focal_weights * ce_loss
        else:
            # Compute binary cross entropy loss
            ce_loss = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction='none')

            # Calculate Focal loss
            pt = probabilities * targets_onehot + (1 - probabilities) * (1 - targets_onehot)
            focal_weights = (self.alpha * targets_onehot + (1 - self.alpha) * (1 - targets_onehot))
            focal_loss = focal_weights * torch.pow((1 - pt), self.gamma) * ce_loss

        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):

    def __init__(self, num_classes: int, alpha: float = 0.5, beta: float = 0.5,
                 class_weights: list = None, smooth: float = 1e-7):
        super(TverskyLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: (batch_size, height, width)

        # Apply softmax or sigmoid to convert logits to probabilities
        targets_onehot = labels_to_onehot(targets, self.num_classes)
        probabilities = logits_to_probs(logits, self.num_classes)

        # Calculate True Positives, False Positives and False Negatives
        tp = (probabilities * targets_onehot).sum(dim=(2, 3))
        fp = (probabilities * (1 - targets_onehot)).sum(dim=(2, 3))
        fn = ((1 - probabilities) * targets_onehot).sum(dim=(2, 3))

        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Average for each class
        tversky = tversky.mean(dim=0)

        # Calculate Tversky loss
        tversky_loss = 1 - tversky

        # Apply class weights if provided
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            tversky_loss *= self.class_weights

        # Average across all batches
        return tversky_loss.mean()


class FocalTverskyLoss(nn.Module):

    def __init__(self, num_classes: int, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0,
                 class_weights: list = None, smooth: float = 1e-7):
        super(FocalTverskyLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: (batch_size, height, width)

        # Apply softmax or sigmoid to convert logits to probabilities
        targets_onehot = labels_to_onehot(targets, self.num_classes)
        probabilities = logits_to_probs(logits, self.num_classes)

        # Calculate True Positives, False Positives and False Negatives
        tp = (probabilities * targets_onehot).sum(dim=(2, 3))
        fp = (probabilities * (1 - targets_onehot)).sum(dim=(2, 3))
        fn = ((1 - probabilities) * targets_onehot).sum(dim=(2, 3))

        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Calculate Focal Tversky loss
        focal_tversky = torch.pow((1 - tversky), self.gamma)

        # Average for each class
        focal_tversky = focal_tversky.mean(dim=0)

        # Apply class weights if provided
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            focal_tversky *= self.class_weights

        # Average across all batches
        return focal_tversky.mean()


# see: https://github.com/LIVIAETS/boundary-loss/blob/master/losses.py
class BoundaryLoss(nn.Module):

    def __init__(self, num_classes: int, class_weights: list[float] = None, normalize: bool = True):
        super(BoundaryLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.normalize = normalize

    def forward(self, logits, targets):
        probabilities = logits_to_probs(logits, self.num_classes).to(logits.device)
        distance_maps = labels_to_sdf(targets, self.num_classes, self.normalize).to(logits.device)

        # Element-wise multiplication of probabilities and distances
        multiplied = torch.einsum('bchw,bchw->bchw', probabilities, distance_maps)
        # Get average for each class
        boundary_loss = multiplied.mean(dim=(0, 2, 3))

        # Apply class weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            boundary_loss *= self.class_weights

        # Average across classes
        return boundary_loss.mean()


# Signed Distance Field is a distance map with negative values inside the object and positive values outside
def labels_to_sdf(labels: torch.Tensor, num_classes: int, normalize: bool) -> torch.Tensor:
    """Convert a label map to onehot and then to a set of distance maps."""
    # labels.shape = (batch_size, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    onehot = labels_to_onehot(labels, num_classes).detach().cpu().numpy()
    dist_maps = np.zeros_like(onehot)

    batch_size = len(onehot)
    for b in range(batch_size):
        dist_maps[b] = onehot_to_sdf(onehot[b], normalize)

    return torch.from_numpy(dist_maps).float()


def onehot_to_sdf(onehot: np.ndarray, normalize: bool) -> np.ndarray:
    """Convert a one-hot encoding to a distance map."""
    # onehot.shape = (num_classes, height, width)
    # returns.shape = (num_classes, height, width)
    num_classes = len(onehot)
    dist_maps = np.zeros_like(onehot)

    for c in range(num_classes):
        # Get the binary mask for the current class
        fg_mask = onehot[c].astype(np.uint8)

        # If the class is not present in the image, skip it
        if fg_mask.any():
            # Get the binary mask for everything outside the current class
            bg_mask = 1 - fg_mask

            # Calculate distance maps for both masks using Euclidean Distance Transform
            fg_dist = edt(fg_mask)
            bg_dist = edt(bg_mask)

            # Normalize distance maps
            if normalize:
                fg_dist = fg_dist / np.max(fg_dist)
                bg_dist = bg_dist / np.max(bg_dist)

            # Combine maps into signed distance field (negative inside, positive outside)
            dist_maps[c] = bg_dist - fg_dist

    return dist_maps


# see: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/hausdorff.py
class HausdorffLoss(nn.Module):

    def __init__(self, num_classes: int = None, class_weights: list[float] = None,
                 alpha: float = 2.0, normalize: bool = True):
        super(HausdorffLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, logits, targets):
        probs = logits_to_probs(logits, self.num_classes)

        # Convert ground truth and prediction to one-hot
        targets = labels_to_onehot(targets, num_classes=self.num_classes)
        predictions = probs_to_onehot(probs, num_classes=self.num_classes)

        # Convert one-hot encodings to distance maps
        targets_dist_maps = onehot_to_dist_map(targets, normalize=self.normalize)
        predictions_dist_maps = onehot_to_dist_map(predictions, normalize=self.normalize)

        # Calculate the Hausdorff distance
        prediction_error = (probs - targets) ** 2
        distance = targets_dist_maps ** self.alpha + predictions_dist_maps ** self.alpha

        # Element-wise multiplication of predication error and hausdorff distance
        multiplied = torch.einsum('bchw,bchw->bchw', prediction_error, distance)
        # Average for each class
        hausdorff_loss = multiplied.mean(dim=(0, 2, 3))

        # Apply class weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            hausdorff_loss *= self.class_weights

        # Average across classes
        return hausdorff_loss.mean()


def onehot_to_dist_map(onehot_batch: torch.Tensor, normalize: bool) -> torch.Tensor:
    # onehot_batch.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    onehot_batch_np = onehot_batch.detach().cpu().numpy()
    distance_maps = np.zeros_like(onehot_batch_np)

    num_items, num_classes, height, width = onehot_batch_np.shape

    # Repeat for each item in the batch
    for b in range(num_items):
        # Repeat the same procedure for each one-hot encoded class map
        for c in range(num_classes):
            # Get the binary mask for the current class
            fg_mask = onehot_batch_np[b, c].astype(np.uint8)

            # If the class is not present in the image, skip it
            if fg_mask.any():
                # Get the binary mask for everything outside the current class
                bg_mask = 1 - fg_mask

                # Compute Euclidean Distance Transform for both masks
                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                # Normalize distance maps
                if normalize:
                    fg_dist = fg_dist / np.max(fg_dist)
                    bg_dist = bg_dist / np.max(bg_dist)

                # in the original code they say: "The idea is to leave blank the negative classes since
                # this is one-hot encoded, another class will supervise that pixel.", but the negative
                # distance map can be added anyway, and it does not change the outcome largely
                distance_maps[b, c] = fg_dist  # + bg_dist

    return torch.tensor(distance_maps, device=onehot_batch.device, dtype=torch.float32)


class CrossEntropyLoss(nn.Module):

    def __init__(self, num_classes: int = None, smooth: float = 1e-7):
        super(CrossEntropyLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'

        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = logits_to_probs(logits, self.num_classes)
        targets = labels_to_onehot(targets, num_classes=self.num_classes)

        log_p = (probs + self.smooth).log()

        cross_entropy = - torch.einsum('bcwh,bcwh->', targets, log_p)
        cross_entropy /= targets.sum() + self.smooth

        return cross_entropy


class SensitivitySpecificityLoss(nn.Module):

    def __init__(self, num_classes: int):
        super(SensitivitySpecificityLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'

        self.num_classes = num_classes

    def forward(self, logits, target):
        probs = logits_to_probs(logits, self.num_classes)
        target = labels_to_onehot(target, self.num_classes)

        true_positives = (probs * target).sum(dim=(2, 3))
        true_negatives = ((1 - probs) * (1 - target)).sum(dim=(2, 3))
        false_positives = (probs * (1 - target)).sum(dim=(2, 3))
        false_negatives = ((1 - probs) * target).sum(dim=(2, 3))

        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)

        return (1 - sensitivity).mean() + (1 - specificity).mean()


# custom loss inspired by boundary loss and hausdorff loss
class EdgeLoss(nn.Module):

    def __init__(self, num_classes: int, class_weights: list[float] = None, alpha: float = 2.0):
        super(EdgeLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.alpha = alpha

    def forward(self, logits, targets):
        probs = logits_to_probs(logits, self.num_classes)
        targets = labels_to_onehot(targets, self.num_classes)

        # Convert one-hot ground truth labels to inverse distance maps
        target_idms = onehot_to_inverse_dist_map(targets)

        # Calculate the Hausdorff distance
        pred_error = (probs - targets) ** 2
        penalty = target_idms ** self.alpha

        # Element-wise multiplication of predication error and hausdorff distance
        multiplied = torch.einsum('bchw,bchw->bchw', pred_error, penalty)
        # Average for each class
        hausdorff_loss = multiplied.mean(dim=(0, 2, 3))

        # Apply class weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            hausdorff_loss *= self.class_weights

        # Average across classes
        return hausdorff_loss.mean()


def onehot_to_inverse_dist_map(onehot_batch: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    # onehot_batch.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    onehot_batch_np = onehot_batch.detach().cpu().numpy()
    inverse_distance_maps = np.zeros_like(onehot_batch_np)

    num_items, num_classes, height, width = onehot_batch_np.shape

    for b in range(num_items):  # Repeat for each item in the batch
        for c in range(num_classes):  # Repeat for each class from onehot encoding
            fg_mask = onehot_batch_np[b, c].astype(np.uint8)

            if fg_mask.any():
                # Get mask of everything that is not the current class
                bg_mask = 1 - fg_mask

                # Calculate distance maps
                fg_dist_map = edt(fg_mask)
                bg_dist_map = edt(bg_mask)

                # Normalize
                if normalize:
                    fg_dist_map = fg_dist_map / np.max(fg_dist_map)
                    bg_dist_map = bg_dist_map / np.max(bg_dist_map)

                # Invert distance maps
                fg_dist_map = np.max(fg_dist_map) - fg_dist_map
                bg_dist_map = np.max(bg_dist_map) - bg_dist_map

                # Combine foreground and background maps
                inverse_distance_maps[b, c] = fg_dist_map + bg_dist_map

    return torch.tensor(inverse_distance_maps, device=onehot_batch.device, dtype=torch.float32)


# Dice + Cross Entropy loss
class ComboLoss(nn.Module):

    def __init__(self, num_classes: int, class_weights: list[float] = None, alpha: float = 0.5, smooth: float = 1e-7):
        super(ComboLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'
        assert 0 <= alpha <= 1, 'Alpha must be between 0 and 1'

        self.num_classes = num_classes
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = logits_to_probs(logits, self.num_classes)
        targets_onehot = labels_to_onehot(targets, self.num_classes)

        # Calculate Cross Entropy loss
        if self.num_classes == 1:
            cross_entropy_loss = F.binary_cross_entropy_with_logits(logits, targets_onehot)
        else:
            cross_entropy_loss = F.cross_entropy(logits, targets, weight=self.class_weights)

        # Calculate Dice loss
        intersection = torch.einsum('bchw,bchw->bc', probs, targets_onehot)
        targets_sum = torch.einsum('bchw->bc', targets_onehot)
        probs_sum = torch.einsum('bchw->bc', probs)

        dice_coeffs = (2 * intersection + self.smooth) / (targets_sum + probs_sum + self.smooth)
        dice_loss = 1 - dice_coeffs.mean(dim=0)

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            dice_loss *= self.class_weights

        dice_loss = dice_loss.mean()

        # Combine losses with alpha parameter
        combo_loss = self.alpha * dice_loss + (1 - self.alpha) * cross_entropy_loss

        return combo_loss
