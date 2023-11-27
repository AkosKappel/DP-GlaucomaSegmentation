import cv2 as cv
import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from pydensecrf.utils import unary_from_softmax
from scipy.interpolate import splprep, splev
from skimage.segmentation import active_contour

__all__ = [
    'separate_disc_and_cup_mask',
    'to_numpy', 'to_tensor', 'unpack', 'pack',
    'erode', 'dilate', 'opening', 'closing',
    'remove_small_components', 'keep_largest_component', 'fill_holes', 'fit_ellipse',
    'dense_crf', 'douglas_peucker', 'smooth_contours', 'snakes',
]


def separate_disc_and_cup_mask(mask: np.ndarray) -> (np.ndarray, np.ndarray):
    # OD: 1, OC: 2, BG: 0
    od_mask = np.where(mask >= 1, 1, 0).astype(np.uint8)
    oc_mask = np.where(mask >= 2, 1, 0).astype(np.uint8)
    return od_mask, oc_mask


def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.uint8)


def to_tensor(array):
    return torch.from_numpy(array)


def unpack(array, axis: int = 0):
    return [array[i] for i in range(array.shape[axis])]


def pack(array, axis: int = 0):
    return np.stack(array, axis=axis)


def erode(mask: np.ndarray | list[np.ndarray], kernel_size: int = 3,
          iterations: int = 1, shape: int = cv.MORPH_RECT) -> np.ndarray | list[np.ndarray]:
    if isinstance(mask, list):
        return [erode(m, kernel_size, iterations, shape) for m in mask]
    kernel = cv.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv.erode(mask.astype(np.uint8), kernel, iterations=iterations)


def dilate(mask: np.ndarray | list[np.ndarray], kernel_size: int = 3,
           iterations: int = 1, shape: int = cv.MORPH_RECT) -> np.ndarray | list[np.ndarray]:
    if isinstance(mask, list):
        return [dilate(m, kernel_size, iterations, shape) for m in mask]
    kernel = cv.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv.dilate(mask.astype(np.uint8), kernel, iterations=iterations)


def opening(mask: np.ndarray | list[np.ndarray], kernel_size: int = 3,
            iterations: int = 1, shape: int = cv.MORPH_RECT) -> np.ndarray | list[np.ndarray]:
    if isinstance(mask, list):
        return [opening(m, kernel_size, iterations, shape) for m in mask]
    kernel = cv.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_OPEN, kernel, iterations=iterations)


def closing(mask: np.ndarray | list[np.ndarray], kernel_size: int = 3,
            iterations: int = 1, shape: int = cv.MORPH_RECT) -> np.ndarray | list[np.ndarray]:
    if isinstance(mask, list):
        return [closing(m, kernel_size, iterations, shape) for m in mask]
    kernel = cv.getStructuringElement(shape, (kernel_size, kernel_size))
    return cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, kernel, iterations=iterations)


def remove_small_components(mask: np.ndarray | list[np.ndarray],
                            min_size: int | float = 0.05, binary: bool = True) -> np.ndarray | list[np.ndarray]:
    # Min size can be given as number of pixels (int) or as a percentage of the total number of pixels (float in (0, 1))
    def _remove_small_components(binary_mask: np.ndarray):
        # Find connected components in the binary mask
        binary_mask = binary_mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_mask)

        # Remove small components
        sizes = stats[:, cv.CC_STAT_AREA]
        sizes[0] = 0  # ignore background

        small_components = np.where(sizes < int(min_size))[0]
        small_components_removed_mask = np.ones_like(binary_mask, dtype=np.uint8)

        for component_idx in small_components:
            small_components_removed_mask[labels == component_idx] = 0

        return small_components_removed_mask

    # Calculate the minimum size in pixels if it was specified as a fraction of the total number of pixels
    if isinstance(min_size, float) and 0 < min_size < 1:
        min_size = min_size * mask.size if isinstance(mask, np.ndarray) else min_size * mask[0].size

    if binary:
        if isinstance(mask, list):  # batch of binary masks
            return [_remove_small_components(m) for m in mask]
        # single binary mask
        return _remove_small_components(mask)

    # batch of multi-class masks
    if isinstance(mask, list):
        return [remove_small_components(m, min_size, binary=False) for m in mask]

    # single multi-class mask
    masks = separate_disc_and_cup_mask(mask)
    result_mask = np.zeros_like(mask, dtype=np.uint8)

    # Remove small components from each mask
    for mask in masks:
        result_mask += _remove_small_components(mask)

    return result_mask


def keep_largest_component(mask: np.ndarray | list[np.ndarray], binary: bool = True) -> np.ndarray | list[np.ndarray]:
    def _keep_largest_component(binary_mask: np.ndarray):
        binary_mask = binary_mask.astype(np.uint8)
        # Find connected components in the binary mask
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_mask)

        # Find the index of the largest connected component (excluding the background component)
        largest_component_index = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1

        # Create a new mask with only the largest connected component
        largest_component_mask = (labels == largest_component_index).astype(np.uint8)

        return largest_component_mask

    if binary:
        if isinstance(mask, list):
            return [_keep_largest_component(m) for m in mask]
        return _keep_largest_component(mask)

    if isinstance(mask, list):
        return [keep_largest_component(m, binary=False) for m in mask]

    masks = separate_disc_and_cup_mask(mask)
    result_mask = np.zeros_like(mask, dtype=np.uint8)

    # Find the largest connected component in each mask and then combine them back into one mask
    for mask in masks:
        result_mask += _keep_largest_component(mask)

    return result_mask


def fill_holes(mask: np.ndarray | list[np.ndarray], binary: bool = True) -> np.ndarray | list[np.ndarray]:
    def _fill_holes(binary_mask: np.ndarray):
        # Find all enclosed contours in the binary mask
        binary_mask = binary_mask.astype(np.uint8)
        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Create a blank mask for drawing the areas with filled holes
        filled_mask = np.zeros_like(binary_mask, dtype=np.uint8)

        # Draw the contours on the blank mask with filled interiors
        for contour in contours:
            cv.drawContours(filled_mask, [contour], 0, 1, -1)

        return filled_mask

    if binary:
        if isinstance(mask, list):
            return [_fill_holes(m) for m in mask]
        return _fill_holes(mask)

    if isinstance(mask, list):
        return [fill_holes(m, binary=False) for m in mask]

    masks = separate_disc_and_cup_mask(mask)
    result_mask = np.zeros_like(mask, dtype=np.uint8)

    for i, mask in enumerate(masks, start=1):
        # Fill the holes in the binary mask
        filled = _fill_holes(mask)

        # Join the filled masks into a single mask
        result_mask[filled == 1] = i

    return result_mask


def fit_ellipse(mask: np.ndarray | list[np.ndarray], binary: bool = True) -> np.ndarray | list[np.ndarray]:
    def _fit_ellipse(binary_mask: np.ndarray):
        # Find the contours of the binary mask (there should be only one)
        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary_mask.copy()

        # Get the largest / only contour (it needs to have at least 5 points to fit an ellipse to it)
        max_contour = max(contours, key=cv.contourArea)
        if len(max_contour) < 5:
            return binary_mask.copy()

        # Fit an ellipse to the largest contour and draw it on a blank mask
        ellipse = cv.fitEllipse(max_contour)
        fitted_mask = np.zeros_like(binary_mask)
        cv.ellipse(fitted_mask, ellipse, 1, -1)

        return fitted_mask

    if binary:
        if isinstance(mask, list):
            return [_fit_ellipse(m) for m in mask]
        return _fit_ellipse(mask)

    if isinstance(mask, list):
        return [fit_ellipse(m, binary=False) for m in mask]

    masks = separate_disc_and_cup_mask(mask)
    result_mask = np.zeros_like(mask, dtype=np.uint8)

    # Fit an ellipse to each mask
    for mask in masks:
        ellipse_mask = _fit_ellipse(mask)

        # TODO: maybe add erosion to the ellipse mask to make it smaller to avoid changing the boundaries too much

        # Preserve original boundaries by joining the ellipse mask with the original mask
        ellipse_mask = np.logical_or(ellipse_mask, mask)

        # Combine the masks back into one
        result_mask += ellipse_mask.astype(np.uint8)

    return result_mask


def dense_crf(image, probab, n_iterations: int = 5, gaussian_kwargs: dict = None, bilateral_kwargs: dict = None):
    if gaussian_kwargs is None:
        gaussian_kwargs = {
            'sxy': (3, 3),
            'compat': 3,
            'kernel': dcrf.DIAG_KERNEL,
            'normalization': dcrf.NORMALIZE_SYMMETRIC,
        }
    if bilateral_kwargs is None:
        bilateral_kwargs = {
            'sxy': (80, 80),
            'srgb': (13, 13, 13),
            'rgbim': np.ascontiguousarray(image).astype(np.uint8),
            'compat': 10,
            'kernel': dcrf.DIAG_KERNEL,
            'normalization': dcrf.NORMALIZE_SYMMETRIC,
        }

    width, height = image.shape[:2]
    n_labels = probab.shape[0]

    d = dcrf.DenseCRF2D(width, height, n_labels)

    U = unary_from_softmax(probab)  # shape of probabilities must be (n_classes, ...)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(**gaussian_kwargs)  # adds the color-independent term, features are the locations only
    d.addPairwiseBilateral(**bilateral_kwargs)  # adds the color-dependent term, i.e. features are (x, y, r, g, b)

    Q = d.inference(n_iterations)
    crf_image = np.argmax(Q, axis=0).reshape((width, height))

    return crf_image


def douglas_peucker(binary_mask: np.ndarray, epsilon: float = 3.0) -> np.ndarray:
    # Find the contours
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv.findContours(binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create a new binary mask
    dp_mask = np.zeros(binary_mask.shape, dtype=np.uint8)

    for contour in contours:
        # Approximate contour with Douglas-Peucker algorithm
        approx = cv.approxPolyDP(contour, epsilon, closed=True)

        # Draw the approximated contour on the mask
        cv.drawContours(dp_mask, [approx], -1, color=1, thickness=-1)

    return dp_mask


def smooth_contours(binary_mask: np.ndarray, s: float = 2.0) -> np.ndarray:
    # Find contours in the binary mask
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return binary_mask

    # Create an empty mask for the displaying the smoothed contours
    smoothed_mask = np.zeros_like(binary_mask)

    for contour in contours:
        # Convert the contour points to float32 for spline interpolation
        contour_points = contour.squeeze().astype(np.float32)

        # Use spline interpolation to smooth the contour
        tck, u = splprep([contour_points[:, 0], contour_points[:, 1]], s=s)
        smoothed_points = np.column_stack(splev(u, tck))

        # Draw the smoothed contour on the mask
        cv.drawContours(smoothed_mask, [smoothed_points.astype(np.int32)], -1, 1, thickness=-1)

    return smoothed_mask


def snakes(binary_mask: np.ndarray, alpha: float = 0.1, beta: float = 10.5, gamma: float = 10.5) -> np.ndarray:
    # Find contours in the mask
    binary_mask = binary_mask.astype(np.uint8)
    contours, _ = cv.findContours(binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    snake_mask = np.zeros_like(binary_mask)

    for contour in contours:
        # Run the Active Contour Model
        snake = active_contour(binary_mask, np.squeeze(contour), alpha=alpha, beta=beta, gamma=gamma)

        # Draw the snake on the mask with thickness -1 (filled)
        cv.drawContours(snake_mask, [snake.astype(np.int32)], -1, color=1, thickness=-1)

    return snake_mask
