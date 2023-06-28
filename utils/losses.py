import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-7, class_weights: list = None, device: str = 'cuda'):

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = torch.tensor(class_weights, device=device) if class_weights is not None else None

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: (batch_size, height, width)

        # Reshape targets to one-hot encoding and calculate probabilities using softmax or sigmoid
        if self.num_classes > 1:
            targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
            probabilities = F.softmax(logits, dim=1)
        else:
            targets_one_hot = targets.unsqueeze(1).float()
            probabilities = torch.sigmoid(logits)

        # Calculate intersection and union
        intersection = torch.sum(probabilities * targets_one_hot, dim=(2, 3))
        union = torch.sum(probabilities + targets_one_hot, dim=(2, 3))

        # Calculate dice coefficients for each class
        dice_coeffs = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average across the batch dimension
        dice_coeffs = dice_coeffs.mean(dim=0)

        # Calculate dice loss for each class
        dice_losses = 1 - dice_coeffs

        # Apply class weights
        if self.class_weights is not None:
            dice_losses *= self.class_weights

        # Average loss across all classes
        return dice_losses.mean()


class IoULoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1e-5, class_weights: list = None, device: str = 'cuda'):
        super(IoULoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.smooth = smooth
        self.class_weights = torch.tensor(class_weights, device=device) if class_weights is not None else None

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: (batch_size, height, width)

        # Apply softmax or sigmoid to convert logits to probabilities
        if self.num_classes > 1:
            targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
            probabilities = F.softmax(logits, dim=1)
        else:
            targets_one_hot = targets.unsqueeze(1).float()
            probabilities = torch.sigmoid(logits)

        # Compute intersection and union
        intersection = (probabilities * targets_one_hot).sum(dim=(2, 3))
        union = probabilities.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection

        # Compute IoU score
        iou_scores = (intersection + self.smooth) / (union + self.smooth)
        # Compute mean across all batches
        iou_scores = iou_scores.mean(dim=0)

        # Compute loss for each class
        iou_losses = 1 - iou_scores

        # Apply class weights if provided
        if self.class_weights is not None:
            iou_losses *= self.class_weights

        # Average across all classes
        return iou_losses.mean()

# TODO:
#  - Focal Loss
#  - Tversky Loss
#  - Focal Tversky Loss
#  - Generalized Dice Loss
#  - Lovasz Hinge Loss
#  - Lovasz Softmax Loss
#  - Combo Loss
#  - Boundary Loss
#  - Hausdorff Loss / Hausdorff Distance

