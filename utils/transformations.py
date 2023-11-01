import cv2 as cv
import numpy as np
import torch

__all__ = [
    'polar_transform', 'inverse_polar_transform', 'undo_polar_transform',
    'keep_red_channel', 'keep_green_channel', 'keep_blue_channel', 'keep_gray_channel',
    'occlude', 'sharpen', 'arctan',
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


def keep_red_channel(img, **kwargs):
    return img[:, :, 0]


def keep_green_channel(img, **kwargs):
    return img[:, :, 1]


def keep_blue_channel(img, **kwargs):
    return img[:, :, 2]


def keep_gray_channel(img, **kwargs):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Different activation function instead of sigmoid
# see: https://lars76.github.io/2021/09/05/activations-segmentation.html
def arctan(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x) / torch.tensor(np.pi))


def occlude(img, p: float = 0.5, occlusion_size: int = 32, occlusion_value: int = 0, **kwargs):
    if np.random.rand() > p:
        return img

    h, w = img.shape[:2]
    assert h >= occlusion_size and w >= occlusion_size, \
        f'Image size ({h}, {w}) must be greater than occlusion size ({occlusion_size}, {occlusion_size})'

    x = np.random.randint(0, w - occlusion_size)
    y = np.random.randint(0, h - occlusion_size)

    img[y:y + occlusion_size, x:x + occlusion_size] = occlusion_value

    return img


def sharpen(img, p: float = 0.5, **kwargs):
    if np.random.rand() > p:
        return img

    # kernel = np.array([
    #     [-1, -1, -1],
    #     [-1, 9, -1],
    #     [-1, -1, -1],
    # ])
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    # kernel = np.array([
    #     [-1, -1, -1, -1, -1],
    #     [-1, 2, 2, 2, -1],
    #     [-1, 2, 8, 2, -1],
    #     [-1, 2, 2, 2, -1],
    #     [-1, -1, -1, -1, -1],
    # ])
    # kernel = np.array([
    #     [0, 0, -1, 0, 0],
    #     [0, -1, -2, -1, 0],
    #     [-1, -2, 16, -2, -1],
    #     [0, -1, -2, -1, 0],
    #     [0, 0, -1, 0, 0],
    # ])
    return cv.filter2D(img, -1, kernel)
