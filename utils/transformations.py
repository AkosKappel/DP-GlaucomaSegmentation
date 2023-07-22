import cv2 as cv
import numpy as np
import torch

__all__ = [
    'polar_transform', 'inverse_polar_transform', 'undo_polar_transform', 'arctan',
    'keep_red_channel', 'keep_green_channel', 'keep_blue_channel', 'keep_gray_channel',
]


def polar_transform(img, **kwargs):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    value = np.sqrt(((width / 2.0) ** 2.0) + ((height / 2.0) ** 2.0))
    polar_img = cv.linearPolar(img, center, value, cv.WARP_FILL_OUTLIERS)
    polar_img = cv.rotate(polar_img, cv.ROTATE_90_COUNTERCLOCKWISE)
    return polar_img


def inverse_polar_transform(img, **kwargs):
    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    value = np.sqrt(((width / 2.0) ** 2.0) + ((height / 2.0) ** 2.0))
    cartesian_img = cv.linearPolar(img, center, value, cv.WARP_INVERSE_MAP | cv.WARP_FILL_OUTLIERS)
    return cartesian_img


def undo_polar_transform(images, masks, preds):
    np_images = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    np_masks = masks.detach().cpu().numpy()
    np_preds = preds.detach().cpu().numpy()

    new_images = np.zeros_like(np_images)
    new_masks = np.zeros_like(np_masks)
    new_preds = np.zeros_like(np_preds)

    for i, _ in enumerate(np_images):
        new_images[i] = inverse_polar_transform(np_images[i])
        new_masks[i] = inverse_polar_transform(np_masks[i])
        new_preds[i] = inverse_polar_transform(np_preds[i])

    images = torch.from_numpy(new_images.transpose(0, 3, 1, 2)).float().to(images.device)
    masks = torch.from_numpy(new_masks).long().to(masks.device)
    preds = torch.from_numpy(new_preds).long().to(preds.device)

    return images, masks, preds


# Activation function instead of sigmoid
# see: https://lars76.github.io/2021/09/05/activations-segmentation.html
def arctan(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x) / torch.tensor(np.pi))


def keep_red_channel(img, **kwargs):
    return img[:, :, 0]


def keep_green_channel(img, **kwargs):
    return img[:, :, 1]


def keep_blue_channel(img, **kwargs):
    return img[:, :, 2]


def keep_gray_channel(img, **kwargs):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
