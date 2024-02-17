import cv2 as cv
import numpy as np
import torch

__all__ = [
    'keep_red_channel', 'keep_green_channel', 'keep_blue_channel', 'keep_gray_channel',
    'binarize', 'extract_disc_mask', 'extract_cup_mask', 'separate_disc_and_cup_mask',
    'occlude', 'sharpen', 'arctan',
    'polar_transform', 'inverse_polar_transform', 'undo_polar_transform',
    'calculate_rgb_cumsum', 'source_to_target_correction', 'get_bounding_box',
]


def keep_red_channel(img, **kwargs):
    return img[:, :, 0]


def keep_green_channel(img, **kwargs):
    return img[:, :, 1]


def keep_blue_channel(img, **kwargs):
    return img[:, :, 2]


def keep_gray_channel(img, **kwargs):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def binarize(image, labels: list | int, **kwargs):
    if isinstance(labels, int):
        return np.where(image > labels, 1, 0).astype(np.uint8)
    return np.isin(image, labels).astype(np.uint8)


def extract_disc_mask(image, **kwargs):
    return binarize(image, labels=[1, 2])


def extract_cup_mask(image, **kwargs):
    return binarize(image, labels=[2])


def separate_disc_and_cup_mask(mask: np.ndarray) -> (np.ndarray, np.ndarray):
    # OD: 1, OC: 2, BG: 0
    od_mask = np.where(mask >= 1, 1, 0).astype(np.uint8)
    oc_mask = np.where(mask >= 2, 1, 0).astype(np.uint8)
    return od_mask, oc_mask


def occlude(image, p: float = 0.5, occlusion_size: int = 32, occlusion_value: int = 0, **kwargs):
    if np.random.rand() > p:
        return image

    h, w = image.shape[:2]
    assert h >= occlusion_size and w >= occlusion_size, \
        f'Image size ({h}, {w}) must be greater than occlusion size ({occlusion_size}, {occlusion_size})'

    x = np.random.randint(0, w - occlusion_size)
    y = np.random.randint(0, h - occlusion_size)

    image[y:y + occlusion_size, x:x + occlusion_size] = occlusion_value

    return image


def sharpen(image, p: float = 0.5, connectivity: int = 4, **kwargs):
    if np.random.rand() > p:
        return image

    if connectivity == 4:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ])
    elif connectivity == 8:
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ])
    else:
        raise ValueError(f'Connectivity must be 4 or 8, got {connectivity}')
    return cv.filter2D(image, -1, kernel)


# Different activation function instead of sigmoid
# see: https://lars76.github.io/2021/09/05/activations-segmentation.html
def arctan(x):
    return 1e-7 + (1 - 2 * 1e-7) * (0.5 + torch.arctan(x) / torch.tensor(np.pi))


def polar_transform(cartesian_image, radius_ratio: float = 1.0, **kwargs):
    height, width = cartesian_image.shape[:2]
    center = (width // 2, height // 2)

    # Linear interpolation between inner and outer radius
    inner_radius = np.min([width, height]) / 2.0
    outer_radius = np.sqrt(((width / 2.0) ** 2.0) + ((height / 2.0) ** 2.0))
    radius = inner_radius + (outer_radius - inner_radius) * radius_ratio

    polar_image = cv.linearPolar(cartesian_image, center, radius, cv.WARP_FILL_OUTLIERS)
    polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    return polar_image


def inverse_polar_transform(polar_image, radius_ratio: float = 1.0, **kwargs):
    polar_image = cv.rotate(polar_image, cv.ROTATE_90_CLOCKWISE)
    height, width = polar_image.shape[:2]
    center = (width // 2, height // 2)

    inner_radius = np.min([width, height]) / 2.0
    outer_radius = np.sqrt(((width / 2.0) ** 2.0) + ((height / 2.0) ** 2.0))
    radius = inner_radius + (outer_radius - inner_radius) * radius_ratio

    cartesian_image = cv.linearPolar(polar_image, center, radius, cv.WARP_INVERSE_MAP | cv.WARP_FILL_OUTLIERS)
    return cartesian_image


def undo_polar_transform(images_batch, masks_batch, preds_batch):
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


def calculate_rgb_cumsum(image):
    r_hist = cv.calcHist([image], [0], None, [256], [0, 256])
    g_hist = cv.calcHist([image], [1], None, [256], [0, 256])
    b_hist = cv.calcHist([image], [2], None, [256], [0, 256])

    r_cdf = r_hist.cumsum()
    g_cdf = g_hist.cumsum()
    c_blue = b_hist.cumsum()

    # normalize to [0, 1]
    r_cdf /= r_cdf[-1]
    g_cdf /= g_cdf[-1]
    c_blue /= c_blue[-1]

    return r_cdf, g_cdf, c_blue


def source_to_target_correction(source_image, target_image):
    cdf_source_red, cdf_source_green, cdf_source_blue = calculate_rgb_cumsum(source_image)
    cdf_target_red, cdf_target_green, cdf_target_blue = calculate_rgb_cumsum(target_image)

    # interpolate CDFs
    red_lookup = np.interp(cdf_source_red, cdf_target_red, np.arange(256))
    green_lookup = np.interp(cdf_source_green, cdf_target_green, np.arange(256))
    blue_lookup = np.interp(cdf_source_blue, cdf_target_blue, np.arange(256))

    # apply the lookup tables to the source image
    corrected = source_image.copy()
    corrected[..., 0] = red_lookup[source_image[..., 0]].reshape(source_image.shape[:2])
    corrected[..., 1] = green_lookup[source_image[..., 1]].reshape(source_image.shape[:2])
    corrected[..., 2] = blue_lookup[source_image[..., 2]].reshape(source_image.shape[:2])

    return corrected


def get_bounding_box(binary_mask: np.ndarray) -> tuple[int, int, int, int]:
    # Find the contours of the binary mask (there should be only one)
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, 0, 0

    # Get the largest / only contour
    max_contour = max(contours, key=cv.contourArea)

    # Get the bounding box of the contour
    x, y, w, h = cv.boundingRect(max_contour)

    return x, y, w, h
