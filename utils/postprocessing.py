import cv2 as cv
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.interpolate import splprep, splev
from skimage.segmentation import active_contour

__all__ = [
    'separate_disc_and_cup_mask',
    'keep_largest_component', 'apply_largest_component_selection',
    'fill_holes', 'apply_hole_filling',
    'fit_ellipse', 'apply_ellipse_fitting',
    'dense_crf', 'douglas_peucker', 'smooth_contours', 'snakes',
]


def separate_disc_and_cup_mask(mask: np.ndarray) -> (np.ndarray, np.ndarray):
    # OD: 1, OC: 2, BG: 0
    od_mask = np.where(mask >= 1, 1, 0).astype(np.uint8)
    oc_mask = np.where(mask >= 2, 1, 0).astype(np.uint8)
    return od_mask, oc_mask


# TODO: Morphological operations (erosion, dilation, opening, closing, etc.) for removing noisy pixels (like in ResNet)


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    # Find connected components in the binary mask
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_mask)

    # Find the index of the largest connected component (excluding the background component)
    largest_component_index = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1

    # Create a new mask with only the largest connected component
    largest_component_mask = (labels == largest_component_index).astype(np.uint8)

    return largest_component_mask


def apply_largest_component_selection(mask: np.ndarray) -> np.ndarray:
    masks = separate_disc_and_cup_mask(mask)
    result_mask = np.zeros_like(mask, dtype=np.uint8)

    # Find the largest connected component in each mask and then combine them back into one mask
    for mask in masks:
        largest_component_mask = keep_largest_component(mask)
        result_mask += largest_component_mask

    return result_mask


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    # Find all enclosed contours in the binary mask
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for drawing the areas with filled holes
    filled_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Draw the contours on the blank mask with filled interiors
    for contour in contours:
        cv.drawContours(filled_mask, [contour], 0, 1, -1)

    return filled_mask


def apply_hole_filling(mask: np.ndarray) -> np.ndarray:
    masks = separate_disc_and_cup_mask(mask)
    result_mask = np.zeros_like(mask, dtype=np.uint8)

    for i, mask in enumerate(masks, start=1):
        # Fill the holes in the mask
        filled_mask = fill_holes(mask)

        # Join the filled masks into a single mask
        result_mask[filled_mask == 1] = i

    return result_mask


def fit_ellipse(binary_mask: np.ndarray) -> np.ndarray:
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


def apply_ellipse_fitting(mask: np.ndarray) -> np.ndarray:
    masks = separate_disc_and_cup_mask(mask)
    fitted_mask = np.zeros_like(mask, dtype=np.uint8)

    # Fit an ellipse to each mask
    for mask in masks:
        ellipse_mask = fit_ellipse(mask)

        # TODO: maybe add erosion to the ellipse mask to make it smaller to avoid changing the boundaries too much

        # Preserve original boundaries by joining the ellipse mask with the original mask
        ellipse_mask = np.logical_or(ellipse_mask, mask)

        # Combine the masks back into one
        fitted_mask += ellipse_mask.astype(np.uint8)

    return fitted_mask


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
