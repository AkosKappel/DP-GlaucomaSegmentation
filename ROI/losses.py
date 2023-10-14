import torch


def focal_loss(pred, gt):
    pred = pred.unsqueeze(1).float()
    gt = gt.unsqueeze(1).float()

    pos_indices = gt.eq(1).float()
    neg_indices = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_indices
    neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 3) * neg_weights * neg_indices

    num_pos = pos_indices.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def center_loss(prediction, mask, regr, mask_weight: float = 1.0, regr_weight: float = 1.0, size_average: bool = True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = focal_loss(pred_mask, mask)

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Weighted total loss
    loss = mask_weight * mask_loss + regr_weight * regr_loss
    if not size_average:
        loss *= prediction.shape[0]

    return loss, mask_loss, regr_loss
