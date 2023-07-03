from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from utils.metrics import update_metrics


def evaluate(model, criterion, device, loader):
    model.eval()
    model = model.to(device=device)
    history = defaultdict(list)
    total = len(loader)
    loop = tqdm(loader, total=total, leave=True, desc='Evaluating')
    mean_metrics = None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.float().to(device=device)
            masks = masks.long().to(device=device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1)

            # performance metrics
            update_metrics(masks, preds, history)
            history['loss'].append(loss.item())

            # show mean metrics after every batch
            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics


def evaluate_tta(model, criterion, device, loader, show_example=False):
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
                    plt.imshow(image.cpu().numpy().transpose(1, 2, 0) / 255.0)
                    plt.show()

                    # plot rotated images
                    _, ax = plt.subplots(2, 4, figsize=(15, 8))
                    for i in range(8):
                        ax[i // 4, i % 4].imshow(augmented_images[i].cpu().numpy().transpose(1, 2, 0) / 255.0)
                    plt.show()

                    # plot masks
                    _, ax = plt.subplots(2, 4, figsize=(15, 8))
                    for i in range(8):
                        ax[i // 4, i % 4].imshow(augmented_masks[i].squeeze(0).cpu().numpy() / 255.0)
                    plt.show()

                    # plot predictions
                    _, ax = plt.subplots(2, 4, figsize=(15, 8))
                    for i in range(8):
                        ax[i // 4, i % 4].imshow(preds[i].cpu().numpy())
                    plt.show()

                    # plot averaged predictions
                    _, ax = plt.subplots(1, 3, figsize=(15, 5))
                    ax[0].imshow(mask.cpu().numpy())
                    ax[1].imshow(averaged_outputs_preds.cpu().numpy())
                    ax[2].imshow(averaged_probs_preds.cpu().numpy())
                    plt.show()
                    return

            all_averaged_outputs_preds = torch.stack(all_averaged_outputs_preds, dim=0)
            all_averaged_probs_preds = torch.stack(all_averaged_probs_preds, dim=0)

            loss = criterion(all_averaged_outputs_preds, masks)
            history['loss_logits'].append(loss.item())
            loss = criterion(all_averaged_probs_preds, masks)
            history['loss_probs'].append(loss.item())

            # performance metrics
            update_metrics(masks, all_averaged_outputs_preds, history, prefix='averaged_logits_')
            update_metrics(masks, all_averaged_probs_preds, history, prefix='averaged_probs_')

            mean_metrics = {k: np.mean(v) for k, v in history.items()}
            loop.set_postfix(**mean_metrics)

    return mean_metrics
