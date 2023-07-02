import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import torch
import torch.nn as nn
import torch.nn.functional as F


def logits_to_probs(logits: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Convert logits from a model to probabilities by applying softmax."""
    # logits.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    return F.softmax(logits, dim=dim)


def probs_to_labels(probs: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Convert probabilities from a model to labels by choosing the class with the highest probability."""
    # probs.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, height, width)
    return torch.argmax(probs, dim=dim)


def labels_to_onehot(labels: torch.Tensor | np.ndarray, num_classes: int) -> torch.Tensor | np.ndarray:
    """Convert a label (integers) to one-hot encoding."""
    # labels.shape = (batch_size, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    if num_classes > 1:
        return F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    else:
        return labels.unsqueeze(1).float()


def onehot_to_labels(onehot: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Convert one-hot encoding to a class label."""
    # onehot.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, height, width)
    return torch.argmax(onehot, dim=dim)


def probs_to_onehot(probs: torch.Tensor, num_classes: int, dim: int = 1) -> torch.Tensor:
    """Convert probabilities from a model to one-hot encoding."""
    # probs.shape = (batch_size, num_classes, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    return labels_to_onehot(probs_to_labels(probs, dim=dim), num_classes=num_classes)


def onehot_to_dist_maps(onehot: np.ndarray) -> np.ndarray:
    """Convert a one-hot encoding to a distance map."""
    # onehot.shape = (num_classes, height, width)
    # returns.shape = (num_classes, height, width)
    num_classes = len(onehot)
    dist_maps = np.zeros_like(onehot)

    for i in range(num_classes):
        pos_mask = onehot[i].astype(np.uint8)

        if pos_mask.any():
            neg_mask = 1 - pos_mask
            pos_dist = edt(pos_mask)
            neg_dist = edt(neg_mask)
            dist_maps[i] = neg_dist - pos_dist

    return dist_maps


def labels_to_dist_maps(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert a label map to onehot and then to a set of distance maps."""
    # labels.shape = (batch_size, height, width)
    # returns.shape = (batch_size, num_classes, height, width)
    onehot = labels_to_onehot(labels, num_classes).cpu().numpy()

    dist_maps = np.zeros_like(onehot)
    for i in range(len(onehot)):
        dist_maps[i] = onehot_to_dist_maps(onehot[i])

    return torch.from_numpy(dist_maps).float()


def onehot_to_hd_maps(onehot: np.ndarray) -> np.ndarray:
    # onehot.shape = (num_classes, height, width)
    # returns.shape = (num_classes, height, width)
    num_classes = len(onehot)

    hd_maps = np.zeros_like(onehot)
    for i in range(num_classes):
        pos_mask = onehot[i].astype(np.uint8)

        if pos_mask.any():
            neg_mask = 1 - pos_mask
            pos_dist = edt(pos_mask)
            neg_dist = edt(neg_mask)
            hd_maps[i] = pos_dist + neg_dist

    return hd_maps


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

        if self.num_classes > 1:
            # Calculate probabilities using softmax
            probabilities = logits_to_probs(logits)
            # One-hot encode targets (e.g. 1 -> [0, 1, 0], 2 -> [0, 0, 1])
            targets = labels_to_onehot(targets, self.num_classes)
        else:
            # Calculate probabilities using sigmoid
            probabilities = torch.sigmoid(logits)
            # Add channel dimension to targets (e.g. (batch_size, height, width) -> (batch_size, 1, height, width))
            targets = targets.unsqueeze(1).float()

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
        self.idc = torch.arange(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits.shape = (batch_size, num_classes, height, width)
        # targets.shape = (batch_size, height, width)

        if self.num_classes > 1:
            # Calculate probabilities using softmax
            probabilities = logits_to_probs(logits)
            # One-hot encode targets (e.g. 1 -> [0, 1, 0], 2 -> [0, 0, 1])
            targets = labels_to_onehot(targets, self.num_classes)
        else:
            # Calculate probabilities using sigmoid
            probabilities = torch.sigmoid(logits)
            # Add channel dimension to targets (e.g. (batch_size, height, width) -> (batch_size, 1, height, width))
            targets = targets.unsqueeze(1).float()

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
    def __init__(self, num_classes: int = 3, class_weights: list = None, smooth: float = 1e-7):
        super(IoULoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None

    def forward(self, logits, targets):
        # logits.shape = (batch_size, num_classes, height, width)
        # targets.shape = (batch_size, height, width)

        # Apply softmax or sigmoid to convert logits to probabilities
        if self.num_classes > 1:
            probabilities = logits_to_probs(logits)
            targets = labels_to_onehot(targets, self.num_classes)
        else:
            probabilities = torch.sigmoid(logits)
            targets = targets.unsqueeze(1).float()

        # Compute intersection and union
        intersection = (probabilities * targets).sum(dim=(2, 3))
        union = probabilities.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection

        # Compute IoU score
        iou_scores = (intersection + self.smooth) / (union + self.smooth)
        # Compute mean across all batches
        iou_scores = iou_scores.mean(dim=0)

        # Compute loss for each class
        iou_losses = 1 - iou_scores

        # Apply class weights if provided
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            iou_losses *= self.class_weights

        # Average across all classes
        return iou_losses.mean()


class FocalLoss(nn.Module):
    def __init__(self, num_classes: int = 3, alpha: float = 0.25, gamma: float = 2, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes  # 1 for binary classification
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: = (batch_size, height, width)

        if self.num_classes > 1:
            # Apply softmax to convert logits to probabilities and one-hot encode targets
            probabilities = logits_to_probs(logits)
            targets_one_hot = labels_to_onehot(targets, self.num_classes)

            # Compute categorical cross entropy loss
            ce_loss = F.cross_entropy(logits, targets, reduction='none')

            # Calculate Focal loss
            pt = (probabilities * targets_one_hot).sum(dim=1)
            focal_weights = torch.pow((1 - pt), self.gamma)
            focal_loss = self.alpha * focal_weights * ce_loss
        else:
            # Apply sigmoid to convert logits to probabilities and remove class dimension from targets
            probabilities = torch.sigmoid(logits)
            targets_one_hot = targets.unsqueeze(1).float()

            # Compute binary cross entropy loss
            ce_loss = F.binary_cross_entropy_with_logits(logits, targets_one_hot, reduction='none')

            # Calculate Focal loss
            pt = probabilities * targets_one_hot + (1 - probabilities) * (1 - targets_one_hot)
            focal_weights = (self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot))
            focal_loss = focal_weights * torch.pow((1 - pt), self.gamma) * ce_loss

        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):

    def __init__(self, num_classes: int = 3, alpha: float = 0.5, beta: float = 0.5,
                 class_weights: list = None, smooth: float = 1e-7):
        super(TverskyLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: (batch_size, height, width)

        # Apply softmax or sigmoid to convert logits to probabilities
        if self.num_classes > 1:
            targets_one_hot = labels_to_onehot(targets, self.num_classes)
            probabilities = logits_to_probs(logits)
        else:
            targets_one_hot = targets.unsqueeze(1).float()
            probabilities = torch.sigmoid(logits)

        # Calculate True Positives, False Positives and False Negatives
        tp = (probabilities * targets_one_hot).sum(dim=(2, 3))
        fp = (probabilities * (1 - targets_one_hot)).sum(dim=(2, 3))
        fn = ((1 - probabilities) * targets_one_hot).sum(dim=(2, 3))

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

    def __init__(self, num_classes: int = 3, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0,
                 class_weights: list = None, smooth: float = 1e-7):
        super(FocalTverskyLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: (batch_size, height, width)

        # Apply softmax or sigmoid to convert logits to probabilities
        if self.num_classes > 1:
            targets_one_hot = labels_to_onehot(targets, self.num_classes)
            probabilities = logits_to_probs(logits)
        else:
            targets_one_hot = targets.unsqueeze(1).float()
            probabilities = torch.sigmoid(logits)

        # Calculate True Positives, False Positives and False Negatives
        tp = (probabilities * targets_one_hot).sum(dim=(2, 3))
        fp = (probabilities * (1 - targets_one_hot)).sum(dim=(2, 3))
        fn = ((1 - probabilities) * targets_one_hot).sum(dim=(2, 3))

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

    def __init__(self, idc: list[int] = None, class_weights: list[float] = None):
        super(BoundaryLoss, self).__init__()
        self.idc = idc if idc is not None else [0, 1, 2]
        self.num_classes = len(idc)
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None

    def forward(self, logits, targets):
        if self.num_classes > 1:
            probs = logits_to_probs(logits)
        else:
            probs = torch.sigmoid(logits)
        dist_maps = labels_to_dist_maps(targets, self.num_classes)

        probs_class = probs[:, self.idc, ...].to(logits.device)
        dists_class = dist_maps[:, self.idc, ...].to(logits.device)

        multiplied = torch.einsum('bchw,bchw->bchw', probs_class, dists_class)
        boundary_loss = multiplied.mean(dim=(0, 2, 3))

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            boundary_loss *= self.class_weights

        return multiplied.mean()


# see: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/hausdorff.py
class HausdorffLoss(nn.Module):

    def __init__(self, idc: list[int] = None, class_weights: list[float] = None, alpha: float = 2.0):
        super(HausdorffLoss, self).__init__()
        self.idc = idc if idc is not None else [0, 1, 2]
        self.num_classes = len(idc)
        self.class_weights = torch.tensor(class_weights) if class_weights is not None else None
        self.alpha = alpha

    def forward(self, logits, targets):
        if self.num_classes > 1:
            probs = logits_to_probs(logits)
        else:
            probs = torch.sigmoid(logits)
        batch_size = probs.shape[0]

        targets = labels_to_onehot(targets, num_classes=self.num_classes)
        preds = probs_to_onehot(probs, num_classes=self.num_classes)

        probs_class = probs[:, self.idc, ...].float()
        targets_class = targets[:, self.idc, ...].float()
        preds_class = preds[:, self.idc, ...].float()

        target_dist_maps_np = np.stack(
            [onehot_to_hd_maps(targets_class[b].cpu().detach().numpy()) for b in range(batch_size)], axis=0)
        targets_dm = torch.tensor(target_dist_maps_np, device=probs.device, dtype=torch.float32)

        preds_dm_np = np.stack(
            [onehot_to_hd_maps(preds_class[b].cpu().detach().numpy()) for b in range(batch_size)], axis=0)
        preds_dm = torch.tensor(preds_dm_np, device=probs.device, dtype=torch.float32)

        pred_error = (probs_class - targets_class) ** 2
        distance = targets_dm ** self.alpha + preds_dm ** self.alpha

        multiplied = torch.einsum('bchw,bchw->bchw', pred_error, distance)
        hausdorff_loss = multiplied.mean(dim=(0, 2, 3))

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
            hausdorff_loss *= self.class_weights

        return hausdorff_loss.mean()
