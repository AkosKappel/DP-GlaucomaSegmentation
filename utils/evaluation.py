from collections import defaultdict
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.nn import functional as F
from tqdm import tqdm

from utils.metrics import update_metrics

__all__ = [
    'predict', 'evaluate', 'tta_evaluate', 'morph_evaluate', 'save_predictions_as_images',
]


def predict(mode: str, model, images, masks, thresh: float = 0.5, labels=None, model0=None, criterion=None):
    preds = loss = od_loss = oc_loss = None

    # Multi-class segmentation
    if mode == 'multiclass':
        outputs = model(images)
        if criterion is not None:
            loss = criterion(outputs, masks)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    # Binary segmentation
    elif mode == 'binary':
        masks = torch.where(torch.isin(masks, labels), 1, 0)
        outputs = model(images)
        if criterion is not None:
            loss = criterion(outputs, masks)
        probs = torch.sigmoid(outputs)
        preds = (probs > thresh).squeeze(1).long()

    # Cascade architecture
    elif mode == 'cascade':
        od_masks = (masks == 1).long() + (masks == 2).long()
        oc_masks = (masks == 2).long()

        od_outputs = model0(images)
        od_probs = torch.sigmoid(od_outputs)
        od_preds = (od_probs > thresh).long()

        images = images * od_preds

        oc_outputs = model(images)
        oc_probs = torch.sigmoid(oc_outputs)
        oc_preds = (oc_probs > thresh).long()

        if criterion is not None:
            od_loss = criterion(od_outputs, od_masks)
            oc_loss = criterion(oc_outputs, oc_masks)
            loss = od_loss + oc_loss
        preds = torch.zeros_like(oc_preds)
        preds[od_preds == 1] = 1
        preds[oc_preds == 1] = 2

    # Dual architecture
    elif mode == 'dual':
        od_masks = (masks == 1).long() + (masks == 2).long()
        oc_masks = (masks == 2).long()

        od_outputs, oc_outputs = model(images)
        od_probs = torch.sigmoid(od_outputs)
        oc_probs = torch.sigmoid(oc_outputs)
        od_preds = (od_probs > thresh).long()
        oc_preds = (oc_probs > thresh).long()

        if criterion is not None:
            od_loss = criterion(od_outputs, od_masks)
            oc_loss = criterion(oc_outputs, oc_masks)
            loss = od_loss + oc_loss
        preds = torch.zeros_like(oc_preds)
        preds[od_preds == 1] = 1
        preds[oc_preds == 1] = 2

    return preds, loss, od_loss, oc_loss


def evaluate(mode: str, model, loader, criterion, device, thresh: float = 0.5, class_ids: list = None, model0=None,
             inverse_transform=None):
    assert mode in ('binary', 'multiclass', 'cascade', 'dual')

    if mode in ('multiclass', 'cascade', 'dual') and class_ids is None:
        class_ids = [[1, 2], [2]]
    elif class_ids is None:
        class_ids = [[1, 2]]
    elif isinstance(class_ids, int):
        class_ids = [[class_ids]]
    elif isinstance(class_ids[0], int):
        class_ids = [class_ids]
    labels = torch.tensor(class_ids).to(device)

    model.eval()
    model = model.to(device)
    criterion = criterion.to(device)
    if model0 is not None:
        model0.eval()
        model0 = model0.to(device)

    mean_metrics = None
    history = defaultdict(list)
    loop = tqdm(loader, total=len(loader), leave=True, desc='Evaluating')

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.float().to(device)
            masks = masks.long().to(device)

            preds, loss, od_loss, oc_loss = predict(mode, model, images, masks, thresh, labels, model0, criterion)
            if inverse_transform is not None:
                images, masks, preds = inverse_transform(images, masks, preds)

            update_metrics(masks, preds, history, class_ids)
            history['loss'].append(loss.item())
            if od_loss is not None:
                history['loss_OD'].append(od_loss.item())
            if oc_loss is not None:
                history['loss_OC'].append(oc_loss.item())

            # show mean metrics after every batch
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def tta_evaluate(model, device, loader, show_example=False):
    model.eval()
    model = model.to(device=device)
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Evaluating')
    mean_metrics = None

    with torch.no_grad():
        for images, masks in loop:
            images = images.float().to(device=device)
            masks = masks.long().to(device=device)

            all_averaged_outputs_preds = []
            all_averaged_probs_preds = []

            for image, mask in zip(images, masks):
                augmented_masks = \
                    [torch.rot90(mask, k=k, dims=(0, 1)).unsqueeze(0) for k in range(4)] + \
                    [torch.rot90(torch.flip(mask, dims=(0,)), k=k, dims=(0, 1)).unsqueeze(0) for k in range(4)]

                augmented_images = torch.cat(
                    [torch.rot90(image, k=k, dims=(1, 2)).unsqueeze(0) for k in range(4)] + \
                    [torch.rot90(torch.flip(image, dims=(1,)), k=k, dims=(1, 2)).unsqueeze(0) for k in range(4)], dim=0)

                outputs = model(augmented_images)
                outputs = torch.stack([
                    torch.rot90(torch.flip(outputs[k], dims=(1,)), k=-k, dims=(1, 2)) if k > 3 else
                    torch.rot90(outputs[k], k=-k, dims=(1, 2))
                    for k in range(8)
                ], dim=0)

                # average logits
                averaged_outputs = torch.mean(outputs, dim=0)
                averaged_outputs_probs = F.softmax(averaged_outputs, dim=0)
                averaged_outputs_preds = torch.argmax(averaged_outputs_probs, dim=0)
                all_averaged_outputs_preds.append(averaged_outputs_preds)

                # average probabilities
                probs = F.softmax(outputs, dim=1)
                averaged_probs = torch.mean(probs, dim=0)
                averaged_probs_preds = torch.argmax(averaged_probs, dim=0)
                all_averaged_probs_preds.append(averaged_probs_preds)

                if show_example:
                    preds = torch.argmax(probs, dim=1)  # 8 augmented predictions

                    # plot original image
                    plt.imshow(image.detach().cpu().numpy().transpose(1, 2, 0) / 255.0)
                    plt.show()

                    # plot rotated images
                    _, ax = plt.subplots(2, 4, figsize=(15, 8))
                    for i in range(8):
                        ax[i // 4, i % 4].imshow(augmented_images[i].detach().cpu().numpy().transpose(1, 2, 0) / 255.0)
                    plt.show()

                    # plot masks
                    _, ax = plt.subplots(2, 4, figsize=(15, 8))
                    for i in range(8):
                        ax[i // 4, i % 4].imshow(augmented_masks[i].squeeze(0).detach().cpu().numpy() / 255.0)
                    plt.show()

                    # plot predictions
                    _, ax = plt.subplots(2, 4, figsize=(15, 8))
                    for i in range(8):
                        ax[i // 4, i % 4].imshow(preds[i].detach().cpu().numpy())
                    plt.show()

                    # plot averaged predictions
                    _, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(mask.detach().cpu().numpy())
                    ax[1].imshow(averaged_outputs_preds.detach().cpu().numpy())
                    ax[2].imshow(averaged_probs_preds.detach().cpu().numpy())
                    plt.show()
                    return

            all_averaged_outputs_preds = torch.stack(all_averaged_outputs_preds, dim=0)
            all_averaged_probs_preds = torch.stack(all_averaged_probs_preds, dim=0)

            # performance metrics
            update_metrics(masks, all_averaged_outputs_preds, history, [[1, 2], [2]], prefix='averaged_logits_')
            update_metrics(masks, all_averaged_probs_preds, history, [[1, 2], [2]], prefix='averaged_probs_')

            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def morph_evaluate(model, device, loader, show_example=False, iterations=1, kernel_size=3):
    model.eval()
    model = model.to(device=device)
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Evaluating')
    mean_metrics = None

    struc_elem = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    with torch.no_grad():
        for images, masks in loop:
            images = images.float().to(device=device)
            masks = masks.long().to(device=device)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_dilated_preds = []
            all_eroded_preds = []
            all_opened_preds = []
            all_closed_preds = []

            for mask, pred in zip(masks, preds):
                pred = pred.detach().cpu().numpy()
                pred = pred.astype(np.uint8)

                dilated_pred = cv.dilate(pred, kernel=struc_elem, iterations=iterations)
                eroded_pred = cv.erode(pred, kernel=struc_elem, iterations=iterations)
                opened_pred = cv.morphologyEx(pred, cv.MORPH_OPEN, kernel=struc_elem, iterations=iterations)
                closed_pred = cv.morphologyEx(pred, cv.MORPH_CLOSE, kernel=struc_elem, iterations=iterations)

                all_dilated_preds.append(torch.tensor(dilated_pred))
                all_eroded_preds.append(torch.tensor(eroded_pred))
                all_opened_preds.append(torch.tensor(opened_pred))
                all_closed_preds.append(torch.tensor(closed_pred))

                if show_example:
                    fig, ax = plt.subplots(1, 6, figsize=(15, 5))
                    ax[0].title.set_text('Ground truth')
                    ax[0].imshow(mask.detach().cpu().numpy())
                    ax[1].title.set_text('Prediction')
                    ax[1].imshow(pred)
                    ax[2].title.set_text('Dilated')
                    ax[2].imshow(dilated_pred)
                    ax[3].title.set_text('Eroded')
                    ax[3].imshow(eroded_pred)
                    ax[4].title.set_text('Opened')
                    ax[4].imshow(opened_pred)
                    ax[5].title.set_text('Closed')
                    ax[5].imshow(closed_pred)
                    plt.show()
                    return

            all_dilated_preds = torch.stack(all_dilated_preds, dim=0)
            all_eroded_preds = torch.stack(all_eroded_preds, dim=0)
            all_opened_preds = torch.stack(all_opened_preds, dim=0)
            all_closed_preds = torch.stack(all_closed_preds, dim=0)

            # performance metrics
            update_metrics(masks, all_dilated_preds, history, [[1, 2], [2]], prefix='dilated_')
            update_metrics(masks, all_eroded_preds, history, [[1, 2], [2]], prefix='eroded_')
            update_metrics(masks, all_opened_preds, history, [[1, 2], [2]], prefix='opened_')
            update_metrics(masks, all_closed_preds, history, [[1, 2], [2]], prefix='closed_')

            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def save_predictions_as_images(mode: str, model, loader, device, thresh: float = 0.5, class_ids: list = None,
                               model0=None, path: str = 'predictions'):
    assert mode in ('binary', 'multiclass', 'cascade', 'dual')

    if mode in ('multiclass', 'cascade', 'dual') and class_ids is None:
        class_ids = [[1, 2], [2]]
    elif class_ids is None:
        class_ids = [[1, 2]]
    elif isinstance(class_ids, int):
        class_ids = [[class_ids]]
    elif isinstance(class_ids[0], int):
        class_ids = [class_ids]
    labels = torch.tensor(class_ids).to(device)

    model.eval()
    model = model.to(device)
    if model0 is not None:
        model0.eval()
        model0 = model0.to(device)

    os.makedirs(path, exist_ok=True)
    img_num = 1

    with torch.no_grad():
        for images, masks in tqdm(loader, total=len(loader), leave=True, desc='Saving predictions'):
            images = images.float().to(device)
            masks = masks.long().to(device)

            preds, *_ = predict(mode, model, images, masks, thresh, labels, model0)

            images = images.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy().astype(np.uint8)

            for image, mask, pred in zip(images, masks, preds):
                image = image.transpose(1, 2, 0)

                mask[mask == 1] = 128
                mask[mask == 2] = 255

                pred[pred == 1] = 128
                pred[pred == 2] = 255

                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                pred = np.repeat(pred[:, :, np.newaxis], 3, axis=2)

                image = image.astype(np.uint8)
                mask = mask.astype(np.uint8)
                pred = pred.astype(np.uint8)

                side_by_side = np.concatenate((image, mask, pred), axis=1)
                plt.imsave(f'{path}/{img_num}.png', side_by_side)
                img_num += 1
