import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: list = None, device: str = 'cuda', smooth: float = 1e-7):

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

        # Calculate Dice coefficients for each class
        dice_coeffs = (2 * intersection + self.smooth) / (union + self.smooth)

        # Average across the batch dimension
        dice_coeffs = dice_coeffs.mean(dim=0)

        # Calculate Dice loss for each class
        dice_losses = 1 - dice_coeffs

        # Apply class weights
        if self.class_weights is not None:
            dice_losses *= self.class_weights

        # Average loss across all classes
        return dice_losses.mean()


class IoULoss(nn.Module):
    def __init__(self, num_classes: int = 3, class_weights: list = None, device: str = 'cuda', smooth: float = 1e-7):
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


class FocalLoss(nn.Module):
    def __init__(self, num_classes=3, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes, height, width)
        # targets: (batch_size, height, width)

        if self.num_classes > 1:
            # Apply softmax to convert logits to probabilities and one-hot encode targets
            probabilities = F.softmax(logits, dim=1)
            targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

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

        print(f'{focal_loss.shape = }, {focal_loss = }')

        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):

    def __init__(self, num_classes: int = 3, alpha: float = 0.5, beta: float = 0.5,
                 class_weights: list = None, device: str = 'cuda', smooth: float = 1e-7):
        super(TverskyLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
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
            tversky_loss *= self.class_weights

        # Average across all batches
        return tversky_loss.mean()


class FocalTverskyLoss(nn.Module):

    def __init__(self, num_classes: int = 3, alpha: float = 0.5, beta: float = 0.5, gamma: float = 1.0,
                 class_weights: list = None, device: str = 'cuda', smooth: float = 1e-7):
        super(FocalTverskyLoss, self).__init__()

        assert num_classes > 0, 'Number of classes must be greater than zero'
        assert class_weights is None or len(class_weights) == num_classes, \
            'Number of class weights must be equal to number of classes'

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
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
            focal_tversky *= self.class_weights

        # Average across all batches
        return focal_tversky.mean()

# TODO:
#  - Generalized Dice Loss
#  - Lovasz Hinge Loss
#  - Lovasz Softmax Loss
#  - Combo Loss
#  - Boundary Loss
#  - Hausdorff Loss / Hausdorff Distance
