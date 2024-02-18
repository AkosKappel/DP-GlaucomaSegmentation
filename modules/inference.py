import torch
import torch.nn.functional as F

__all__ = [
    'predict', 'predict_multiclass', 'predict_multilabel', 'predict_binary', 'predict_cascade', 'predict_dual',
]


def predict(mode: str, model, images, masks=None, device=None,
            thresh: float = 0.5, od_thresh: float = None, oc_thresh: float = None,
            criterion=None, binary_labels=None, base_model=None):
    assert mode in ('binary', 'multiclass', 'multilabel', 'cascade', 'dual')

    if mode == 'multiclass':  # Multi-class segmentation
        return predict_multiclass(model, images, masks, device, criterion)

    if mode == 'multilabel':  # Multi-label segmentation
        return predict_multilabel(model, images, masks, device, criterion, thresh)

    if mode == 'binary':  # Binary segmentation
        assert binary_labels is not None, 'Binary class binary_labels must be provided'
        return predict_binary(model, images, masks, device, criterion, thresh, binary_labels)

    if mode == 'cascade':  # Cascade architecture
        assert base_model is not None, 'Cascade model needs a base model'
        return predict_cascade(base_model, model, images, masks, device, criterion, thresh, od_thresh, oc_thresh)

    if mode == 'dual':  # Dual architecture
        return predict_dual(model, images, masks, device, criterion, thresh, od_thresh, oc_thresh)


def predict_multiclass(model, images, masks=None, device=None, criterion=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    images = images.to(device)
    if masks is not None:
        masks = masks.to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    images = images.float()
    masks = masks.long()

    model.eval()
    with torch.no_grad():
        logits = model(images)

    loss = None
    if criterion is not None and masks is not None:
        loss = criterion(logits, masks)

    probabilities = F.softmax(logits, dim=1)  # (N, C, H, W)
    predictions = torch.argmax(probabilities, dim=1)  # (N, H, W)

    return predictions, probabilities, loss


def predict_multilabel(model, images, masks=None, device=None, criterion=None, threshold: float = 0.5):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    images = images.to(device)
    if masks is not None:
        masks = masks.to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    images = images.float()
    masks = masks.long()

    model.eval()
    with torch.no_grad():
        logits = model(images)

    loss = None
    if criterion is not None and masks is not None:
        # Create binary masks for each class
        masks = (
            (masks == 0).long(),  # background
            (masks == 1).long() + (masks == 2).long(),  # optic disc
            (masks == 2).long(),  # optic cup
        )
        loss = 0
        for i in range(logits.shape[1]):
            loss += criterion(logits[:, i:i + 1, :, :], masks[i])

    probabilities = torch.sigmoid(logits)  # (N, C, H, W)
    predictions = torch.zeros_like(probabilities[:, 0, :, :])  # (N, H, W)
    for i in range(1, probabilities.shape[1]):
        predictions += (probabilities[:, i] > threshold).long()

    return predictions, probabilities, loss


def predict_binary(model, images, masks=None, device=None, criterion=None,
                   threshold: float = 0.5, binary_labels: list[int] = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    images = images.to(device)
    if masks is not None:
        masks = masks.to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    images = images.float()
    masks = masks.long()

    model.eval()
    with torch.no_grad():
        logits = model(images)  # (N, 1, H, W)

    if binary_labels is None:
        binary_labels = torch.tensor([1, 2], device=images.device)
    elif isinstance(binary_labels, torch.Tensor):
        binary_labels = binary_labels.to(images.device)
    else:
        binary_labels = torch.tensor(binary_labels, device=images.device)

    loss = None
    if criterion is not None and masks is not None:
        masks = torch.where(torch.isin(masks, binary_labels), 1, 0)
        loss = criterion(logits, masks)

    probabilities = torch.sigmoid(logits).squeeze(1)  # (N, H, W)
    predictions = (probabilities > threshold).long()  # (N, H, W)

    return predictions, probabilities, loss


def predict_cascade(base_model, model, images, masks=None, device=None, criterion=None,
                    threshold: float = 0.5, od_threshold: float = None, oc_threshold: float = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    base_model = base_model.to(device)
    images = images.to(device)
    if masks is not None:
        masks = masks.to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    images = images.float()
    masks = masks.long()

    base_model.eval()
    with torch.no_grad():
        od_logits = base_model(images)
    od_probabilities = torch.sigmoid(od_logits)  # (N, 1, H, W)
    od_predictions = (od_probabilities > (od_threshold or threshold)).long()  # (N, 1, H, W)

    # Cascading effect: crop everything that is not inside the optic disc
    cropped_images = images * od_predictions  # (N, C, H, W)

    model.eval()
    with torch.no_grad():
        oc_logits = model(cropped_images)
    oc_probabilities = torch.sigmoid(oc_logits)  # (N, 1, H, W)
    oc_predictions = (oc_probabilities > (oc_threshold or threshold)).long()  # (N, 1, H, W)

    loss = None
    if criterion is not None and masks is not None:
        oc_masks = (masks == 2).long()
        loss = criterion(oc_logits, oc_masks)  # total loss is only from the second model

    # Join probabilities from both models along the channel dimension
    probabilities = torch.cat([od_probabilities, oc_probabilities], dim=1)  # (N, 2, H, W)

    # Join predictions from both models
    predictions = torch.zeros_like(oc_predictions)  # (N, 1, H, W)
    predictions[od_predictions == 1] = 1
    predictions[oc_predictions == 1] = 2
    predictions = predictions.squeeze(1)  # (N, H, W)

    return predictions, probabilities, loss


def predict_dual(model, images, masks=None, device=None, criterion=None,
                 threshold: float = 0.5, od_threshold: float = None, oc_threshold: float = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    images = images.to(device)
    if masks is not None:
        masks = masks.to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    images = images.float()
    masks = masks.long()

    model.eval()
    with torch.no_grad():
        od_logits, oc_logits = model(images)
    od_probabilities = torch.sigmoid(od_logits)  # (N, 1, H, W)
    oc_probabilities = torch.sigmoid(oc_logits)  # (N, 1, H, W)
    od_predictions = (od_probabilities > (od_threshold or threshold)).long()  # (N, 1, H, W)
    oc_predictions = (oc_probabilities > (oc_threshold or threshold)).long()  # (N, 1, H, W)

    loss = None
    if criterion is not None and masks is not None:
        od_masks = (masks == 1).long() + (masks == 2).long()
        oc_masks = (masks == 2).long()

        od_loss = criterion(od_logits, od_masks)
        oc_loss = criterion(oc_logits, oc_masks)
        loss = od_loss + oc_loss

    # Join probabilities from both models along the channel dimension
    probabilities = torch.cat([od_probabilities, oc_probabilities], dim=1)  # (N, 2, H, W)

    # Join predictions from both models
    predictions = torch.zeros_like(oc_predictions)  # (N, 1, H, W)
    predictions[od_predictions == 1] = 1
    predictions[oc_predictions == 1] = 2
    predictions = predictions.squeeze(1)  # (N, H, W)

    return predictions, probabilities, loss
