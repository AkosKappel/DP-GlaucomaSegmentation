import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import defaultdict
from torch.nn import functional as F
from tqdm import tqdm

from modules.metrics import update_metrics
from modules.inference import predict

__all__ = ['evaluate', 'tta_evaluate', 'morph_evaluate']


def evaluate(mode: str, model, loader, device, criterion=None,
             thresh: float = 0.5, od_thresh: float = None, oc_thresh: float = None,
             binary_labels: list[int] = None, base_model=None, inverse_transform=None):
    assert mode in ('binary', 'multiclass', 'multilabel', 'cascade', 'dual')

    mean_metrics = None
    history = defaultdict(list)
    loop = tqdm(loader, desc=f'Evaluating {mode} segmentation')

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            preds, _, loss = predict(
                mode, model, images, masks, device, thresh, od_thresh, oc_thresh, criterion, binary_labels, base_model
            )
            if inverse_transform is not None:
                images, masks, preds = inverse_transform(images, masks, preds)

            update_metrics(masks, preds, history, [binary_labels] if mode == 'binary' else [[1, 2], [2]])
            if loss is not None:
                history['loss'].append(loss.item())

            # show updated average metrics
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
