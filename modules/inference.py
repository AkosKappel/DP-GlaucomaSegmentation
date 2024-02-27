import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from modules.metrics import update_metrics
from modules.postprocessing import polar_to_cartesian

__all__ = [
    'evaluate', 'predict', 'predict_multiclass', 'predict_multilabel', 'predict_binary',
    'predict_cascade', 'predict_dual', 'd4_transform', 'd4_inverse_transform',
    'd4_transform_cartesian', 'd4_inverse_transform_cartesian',
    'd4_transform_polar', 'd4_inverse_transform_polar', 'apply_morphological_operation',
]


def evaluate(mode: str, model, loader, device=None, criterion=None,
             thresh: float = 0.5, od_thresh: float = None, oc_thresh: float = None,
             binary_labels: list[int] = None, base_model=None, inverse_transform=None,
             inter_process_fn=None, post_process_fn=None, tta: bool = False, **morph_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    if binary_labels is None:
        binary_labels = [1, 2]

    mean_metrics = None
    history = defaultdict(list)
    loop = tqdm(loader, desc=f'Evaluating {mode} segmentation')
    labels = [binary_labels] if mode == 'binary' else [[1, 2], [2]]
    in_polar = False if inverse_transform is None else True
    to_cartesian = False  # Don't convert because metrics need to be calculated in same space as masks

    with torch.no_grad():
        for images, masks in loop:
            preds, _, loss = predict(
                mode, model, images, masks, device, thresh, od_thresh, oc_thresh,
                criterion, binary_labels, base_model, in_polar, to_cartesian,
                inter_process_fn, post_process_fn, tta, **morph_kwargs,
            )
            if inverse_transform is not None:
                images, masks, preds = inverse_transform(images, masks, preds)

            update_metrics(masks, preds, history, labels)
            if loss is not None:
                history['loss'].append(loss.item())

            # show updated average metrics
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def predict(mode: str, model, images, masks=None, device=None,
            thresh: float = 0.5, od_thresh: float = None, oc_thresh: float = None,
            criterion=None, binary_labels=None, base_model=None,
            in_polar: bool = False, to_cartesian: bool = False,
            inter_process_fn=None, post_process_fn=None, tta: bool = False, **morph_kwargs):
    # Morph kwargs = operation: str, kernel_size: int, iterations: int, kernel_shape: int
    if mode == 'multiclass':  # Multi-class segmentation
        return predict_multiclass(
            model, images, masks, device, criterion,
            in_polar, to_cartesian, post_process_fn, tta, **morph_kwargs,
        )

    if mode == 'multilabel':  # Multi-label segmentation
        return predict_multilabel(
            model, images, masks, device, criterion, thresh,
            in_polar, to_cartesian, post_process_fn, tta, **morph_kwargs,
        )

    if mode == 'binary':  # Binary segmentation
        if binary_labels is None:
            binary_labels = [1, 2]
        return predict_binary(
            model, images, masks, device, criterion, thresh,
            in_polar, to_cartesian, binary_labels, post_process_fn, tta, **morph_kwargs,
        )

    if mode == 'cascade':  # Cascade architecture
        assert base_model is not None, 'Cascade model needs a base model'
        return predict_cascade(
            base_model, model, images, masks, device, criterion,
            thresh, od_thresh, oc_thresh,
            in_polar, to_cartesian, inter_process_fn, post_process_fn, tta, **morph_kwargs,
        )

    if mode == 'dual':  # Dual architecture
        return predict_dual(
            model, images, masks, device, criterion,
            thresh, od_thresh, oc_thresh,
            in_polar, to_cartesian, post_process_fn, tta, **morph_kwargs,
        )

    raise ValueError(f'Invalid mode: {mode}')


def predict_multiclass(model, images, masks=None, device=None, criterion=None,
                       in_polar: bool = False, to_cartesian: bool = False,
                       post_process_fn=None, tta: bool = False, **morph_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    images = images.float().to(device)
    if masks is not None:
        masks = masks.long().to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    if tta:
        images, masks = d4_transform(images, masks, polar=in_polar)

    model.eval()
    with torch.no_grad():
        logits = model(images)

    loss = None
    if criterion is not None and masks is not None:
        loss = criterion(logits, masks)

    probabilities = F.softmax(logits, dim=1)  # (N, C, H, W)
    if tta:
        probabilities = d4_inverse_transform(probabilities, polar=in_polar)
    predictions = torch.argmax(probabilities, dim=1).long()  # (N, H, W)

    if morph_kwargs:
        predictions = apply_morphological_operation(predictions, **morph_kwargs)

    if to_cartesian:
        predictions = polar_to_cartesian(predictions)

    if post_process_fn is not None:
        predictions = post_process_fn(predictions)

    return predictions, probabilities, loss


def predict_multilabel(model, images, masks=None, device=None, criterion=None, threshold: float = 0.5,
                       in_polar: bool = False, to_cartesian: bool = False,
                       post_process_fn=None, tta: bool = False, **morph_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    images = images.float().to(device)
    if masks is not None:
        masks = masks.long().to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    if tta:
        images, masks = d4_transform(images, masks, polar=in_polar)

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
    if tta:
        probabilities = d4_inverse_transform(probabilities, polar=in_polar)

    predictions = torch.zeros_like(probabilities[:, 0, :, :]).long()  # (N, H, W)
    for i in range(1, probabilities.shape[1]):
        predictions += (probabilities[:, i] > threshold).long()

    if morph_kwargs:
        predictions = apply_morphological_operation(predictions, **morph_kwargs)

    if to_cartesian:
        predictions = polar_to_cartesian(predictions)

    if post_process_fn is not None:
        predictions = post_process_fn(predictions)

    return predictions, probabilities, loss


def predict_binary(model, images, masks=None, device=None, criterion=None, threshold: float = 0.5,
                   in_polar: bool = False, to_cartesian: bool = False, binary_labels: list[int] = None,
                   post_process_fn=None, tta: bool = False, **morph_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    images = images.float().to(device)
    if masks is not None:
        masks = masks.long().to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    if tta:
        images, masks = d4_transform(images, masks, polar=in_polar)

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

    probabilities = torch.sigmoid(logits)  # (N, 1, H, W)
    if tta:
        probabilities = d4_inverse_transform(probabilities, polar=in_polar)  # (8 * N, 1, H, W) -> (N, 1, H, W)
    predictions = (probabilities > threshold).squeeze(1).long()  # (N, H, W)

    if morph_kwargs:
        predictions = apply_morphological_operation(predictions, **morph_kwargs)

    if to_cartesian:
        predictions = polar_to_cartesian(predictions)

    if post_process_fn is not None:
        predictions = post_process_fn(predictions)

    return predictions, probabilities, loss


def predict_cascade(base_model, model, images, masks=None, device=None, criterion=None,
                    threshold: float = 0.5, od_threshold: float = None, oc_threshold: float = None,
                    in_polar: bool = False, to_cartesian: bool = False, inter_process_fn=None,
                    post_process_fn=None, tta: bool = False, **morph_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    base_model = base_model.to(device)
    images = images.float().to(device)
    if masks is not None:
        masks = masks.long().to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    if tta:
        images, masks = d4_transform(images, masks, polar=in_polar)

    base_model.eval()
    with torch.no_grad():
        od_logits = base_model(images)
    od_probabilities = torch.sigmoid(od_logits)  # (N, 1, H, W)
    if tta:
        od_probabilities = d4_inverse_transform(od_probabilities, polar=in_polar)
    od_predictions = (od_probabilities > (od_threshold or threshold)).long()  # (N, 1, H, W)

    if inter_process_fn is not None:
        od_predictions = inter_process_fn(od_predictions)

    # Cascading effect: crop everything that is not inside the optic disc
    cropped_images = images * od_predictions  # (N, C, H, W)

    model.eval()
    with torch.no_grad():
        oc_logits = model(cropped_images)
    oc_probabilities = torch.sigmoid(oc_logits)  # (N, 1, H, W)
    if tta:
        oc_probabilities = d4_inverse_transform(oc_probabilities, polar=in_polar)
    oc_predictions = (oc_probabilities > (oc_threshold or threshold)).long()  # (N, 1, H, W)

    loss = None
    if criterion is not None and masks is not None:
        oc_masks = (masks == 2).long()
        loss = criterion(oc_logits, oc_masks)  # total loss is only from the second model

    # Join probabilities from both models along the channel dimension
    probabilities = torch.cat([od_probabilities, oc_probabilities], dim=1)  # (N, 2, H, W)

    # Join predictions from both models
    predictions = torch.zeros_like(oc_predictions).long()  # (N, 1, H, W)
    predictions[od_predictions == 1] = 1
    predictions[oc_predictions == 1] = 2
    predictions = predictions.squeeze(1)  # (N, H, W)

    if morph_kwargs:
        predictions = apply_morphological_operation(predictions, **morph_kwargs)

    if to_cartesian:
        predictions = polar_to_cartesian(predictions)

    if post_process_fn is not None:
        predictions = post_process_fn(predictions)

    return predictions, probabilities, loss


def predict_dual(model, images, masks=None, device=None, criterion=None,
                 threshold: float = 0.5, od_threshold: float = None, oc_threshold: float = None,
                 in_polar: bool = False, to_cartesian: bool = False,
                 post_process_fn=None, tta: bool = False, **morph_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    images = images.float().to(device)
    if masks is not None:
        masks = masks.long().to(device)
    if criterion is not None:
        criterion = criterion.to(device)

    if tta:
        images, masks = d4_transform(images, masks, polar=in_polar)

    model.eval()
    with torch.no_grad():
        od_logits, oc_logits = model(images)
    od_probabilities = torch.sigmoid(od_logits)  # (N, 1, H, W)
    oc_probabilities = torch.sigmoid(oc_logits)  # (N, 1, H, W)

    if tta:
        od_probabilities = d4_inverse_transform(od_probabilities, polar=in_polar)
        oc_probabilities = d4_inverse_transform(oc_probabilities, polar=in_polar)

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
    predictions = predictions.squeeze(1).long()  # (N, H, W)

    if morph_kwargs:
        predictions = apply_morphological_operation(predictions, **morph_kwargs)

    if to_cartesian:
        predictions = polar_to_cartesian(predictions)

    if post_process_fn is not None:
        predictions = post_process_fn(predictions)

    return predictions, probabilities, loss


def d4_transform(images: torch.Tensor, masks: torch.Tensor = None,
                 polar: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    return d4_transform_polar(images, masks) if polar else d4_transform_cartesian(images, masks)


def d4_inverse_transform(probabilities: torch.Tensor, polar: bool = False) -> torch.Tensor:
    return d4_inverse_transform_polar(probabilities) if polar else d4_inverse_transform_cartesian(probabilities)


def d4_transform_cartesian(images: torch.Tensor, masks: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    # images.shape = (B, C, H, W), masks.shape = (B, H, W)
    B, C, H, W = images.shape
    d4_images = []
    d4_masks = []

    for i in range(B):
        image = images[i].unsqueeze(0)  # (1, C, H, W)
        mask = None if masks is None else masks[i].unsqueeze(0)  # (1, H, W)

        for k in range(4):
            rotated_image = torch.rot90(image, k, dims=(2, 3))
            flipped_image = rotated_image.flip(3)
            d4_images.append(rotated_image)
            d4_images.append(flipped_image)

            if mask is not None:
                rotated_mask = torch.rot90(mask, k, dims=(1, 2))
                flipped_mask = rotated_mask.flip(2)
                d4_masks.append(rotated_mask)
                d4_masks.append(flipped_mask)

    return torch.cat(d4_images, dim=0), None if masks is None else torch.cat(d4_masks, dim=0)


def d4_inverse_transform_cartesian(probabilities: torch.Tensor) -> torch.Tensor:
    # predictions.shape = (8*B, C, H, W) for 8 transformations per image
    B, C, H, W = probabilities.shape
    B //= 8

    # Prepare a tensor to hold the aggregated predicted probabilities for the original images
    aggregated_probabilities = torch.zeros_like(probabilities[:B])  # (B, C, H, W)

    for i in range(B):
        original_probabilities = []
        for k in range(8):
            pred = probabilities[8 * i + k].unsqueeze(0)

            if k % 2 == 0:  # Rotated images (no flip)
                de_transformed_pred = torch.rot90(pred, -k // 2, dims=(2, 3))
            else:  # Flipped images
                flipped_pred = pred.flip(3)  # undo the flip
                de_transformed_pred = torch.rot90(flipped_pred, -(k - 1) // 2, dims=(2, 3))  # rotate back

            original_probabilities.append(de_transformed_pred)

        # Average the predictions for the current image
        aggregated_probabilities[i] = torch.mean(
            torch.stack(original_probabilities), dim=0
        ).squeeze(0)

    return aggregated_probabilities


def d4_transform_polar(images: torch.Tensor, masks: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    # images.shape = (B, C, H, W), masks.shape = (B, H, W)
    B, C, H, W = images.shape
    d4_images = []
    d4_masks = []

    for i in range(B):
        image = images[i].unsqueeze(0)  # (1, C, H, W)
        mask = None if masks is None else masks[i].unsqueeze(0)  # (1, H, W)

        for k in range(4):
            shifted_image = torch.roll(image, shifts=k * (W // 4), dims=3)
            flipped_image = torch.flip(shifted_image, [3])
            d4_images.append(shifted_image)
            d4_images.append(flipped_image)

            if mask is not None:
                shifted_mask = torch.roll(mask, shifts=k * (W // 4), dims=2)
                flipped_mask = torch.flip(shifted_mask, [2])
                d4_masks.append(shifted_mask)
                d4_masks.append(flipped_mask)

    return torch.cat(d4_images, dim=0), None if masks is None else torch.cat(d4_masks, dim=0)


def d4_inverse_transform_polar(probabilities: torch.Tensor) -> torch.Tensor:
    # predictions.shape = (8*B, C, H, W) for 8 transformations per image
    B, C, H, W = probabilities.shape
    B //= 8

    # Prepare a tensor to hold the aggregated predicted probabilities for the original images
    aggregated_probabilities = torch.zeros_like(probabilities[:B])  # (B, C, H, W)

    for i in range(B):
        original_probabilities = []
        for k in range(8):
            pred = probabilities[8 * i + k].unsqueeze(0)

            if k % 2 == 0:  # Shifted images (no flip)
                de_transformed_pred = torch.roll(pred, shifts=-(k // 2) * (W // 4), dims=3)
            else:  # Flipped images
                flipped_pred = torch.flip(pred, [3])  # undo the flip
                de_transformed_pred = torch.roll(flipped_pred, shifts=-((k - 1) // 2) * (W // 4), dims=3)  # shift back

            original_probabilities.append(de_transformed_pred)

        # Average the predictions for the current image
        aggregated_probabilities[i] = torch.mean(
            torch.stack(original_probabilities), dim=0
        ).squeeze(0)

    return aggregated_probabilities


def apply_morphological_operation(predictions: torch.Tensor, operation: str, kernel_size: int = 5,
                                  iterations: int = 1, kernel_shape: int = cv.MORPH_ELLIPSE) -> torch.Tensor:
    # predictions.shape = (B, H, W)
    morphed_predictions = []
    kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))

    for pred in predictions.detach().cpu().numpy().astype(np.uint8):
        if operation in ('dilate', 'dilation'):
            morphed = cv.dilate(pred, kernel, iterations=iterations)
        elif operation in ('erode', 'erosion'):
            morphed = cv.erode(pred, kernel, iterations=iterations)
        elif operation in ('open', 'opening'):
            morphed = cv.morphologyEx(pred, cv.MORPH_OPEN, kernel, iterations=iterations)
        elif operation in ('close', 'closing'):
            morphed = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError("Invalid operation. Choose from 'dilation', 'erosion', 'opening', 'closing'.")
        morphed_predictions.append(morphed)

    return torch.tensor(np.array(morphed_predictions), dtype=predictions.dtype, device=predictions.device)
