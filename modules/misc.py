import numpy as np
import torch

from modules.preprocessing import inverse_polar_transform

__all__ = ['undo_polar_transform']


def undo_polar_transform(images_batch: torch.Tensor, masks_batch: torch.Tensor, preds_batch: torch.Tensor):
    np_images = images_batch.detach().cpu().numpy().transpose(0, 2, 3, 1)
    np_masks = masks_batch.detach().cpu().numpy()
    np_preds = preds_batch.detach().cpu().numpy()

    new_images = np.zeros_like(np_images)
    new_masks = np.zeros_like(np_masks)
    new_preds = np.zeros_like(np_preds)

    for i, _ in enumerate(np_images):
        new_images[i] = inverse_polar_transform(np_images[i])
        new_masks[i] = inverse_polar_transform(np_masks[i])
        new_preds[i] = inverse_polar_transform(np_preds[i])

    images_batch = torch.from_numpy(new_images.transpose(0, 3, 1, 2)).float().to(images_batch.device)
    masks_batch = torch.from_numpy(new_masks).long().to(masks_batch.device)
    preds_batch = torch.from_numpy(new_preds).long().to(preds_batch.device)

    return images_batch, masks_batch, preds_batch
