import cv2 as cv
import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from pydensecrf.utils import unary_from_softmax, unary_from_labels
from scipy.interpolate import splprep, splev
from skimage.segmentation import active_contour

__all__ = [
    'postprocess', 'tensor_to_numpy', 'numpy_to_tensor', 'erosion', 'dilation', 'opening', 'closing',
    'separate_disc_and_cup', 'join_disc_and_cup', 'remove_small_components', 'keep_largest_component',
    'fill_holes', 'fit_ellipse', 'douglas_peucker', 'smooth_contours', 'snakes', 'dense_crf'
]


# Main post-processing function
def postprocess(predictions: torch.Tensor, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert the predictions tensor to a numpy array
    predictions = tensor_to_numpy(predictions)

    # Apply postprocessing to the predictions
    discs, cups = separate_disc_and_cup(predictions)

    # Fill holes in the masks
    discs = fill_holes(discs)
    cups = fill_holes(cups)

    # Keep only the largest component in the masks
    discs = keep_largest_component(discs)
    cups = keep_largest_component(cups)

    # Join the disc and cup masks
    predictions = join_disc_and_cup(discs, cups)

    # Convert the predictions back to a tensor and return it
    return numpy_to_tensor(predictions, device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(array).to(device)


def erosion(masks: np.ndarray, kernel_size: int = 5, iterations: int = 1,
            kernel_shape: int = cv.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return np.array([cv.erode(mask.astype(np.uint8), kernel, iterations=iterations) for mask in masks])


def dilation(masks: np.ndarray, kernel_size: int = 5, iterations: int = 1,
             kernel_shape: int = cv.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return np.array([cv.dilate(mask.astype(np.uint8), kernel, iterations=iterations) for mask in masks])


def opening(masks: np.ndarray, kernel_size: int = 5, iterations: int = 1,
            kernel_shape: int = cv.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return np.array([
        cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_OPEN, kernel, iterations=iterations) for mask in masks
    ])


def closing(masks: np.ndarray | list[np.ndarray], kernel_size: int = 5, iterations: int = 1,
            kernel_shape: int = cv.MORPH_ELLIPSE) -> np.ndarray:
    kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return np.array([
        cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, kernel, iterations=iterations) for mask in masks
    ])


def separate_disc_and_cup(masks: np.ndarray) -> (np.ndarray, np.ndarray):
    # Background: 0, Optic disc: 1, Optic cup: 2
    optic_disc_masks = np.where(masks >= 1, 1, 0).astype(np.uint8)
    optic_cup_masks = np.where(masks >= 2, 1, 0).astype(np.uint8)
    return optic_disc_masks, optic_cup_masks


def join_disc_and_cup(optic_disc_masks: np.ndarray, optic_cup_masks: np.ndarray) -> np.ndarray:
    return optic_disc_masks + optic_cup_masks


def remove_small_components(binary_masks: np.ndarray, min_size: int | float = 0.05):
    # Min size can be given as number of pixels (int) or as a percentage of the total number of pixels (float in (0, 1))
    if isinstance(min_size, float) and 0 < min_size < 1:
        min_size = min_size * np.prod(binary_masks[0].shape)

    processed_masks = np.zeros_like(binary_masks, dtype=np.uint8)

    for idx, mask in enumerate(binary_masks.astype(np.uint8)):
        # Find connected components in the binary mask
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8, ltype=cv.CV_32S)
        small_components_removed_mask = np.ones_like(mask, dtype=np.uint8)

        # Remove small components
        sizes = stats[:, cv.CC_STAT_AREA]
        sizes[0] = 0  # ignore background

        small_components = np.where(sizes < int(min_size))[0]

        for component_idx in small_components:
            small_components_removed_mask[labels == component_idx] = 0

        processed_masks[idx] = small_components_removed_mask

    return processed_masks


def keep_largest_component(binary_masks: np.ndarray) -> np.ndarray:
    largest_component_masks = np.zeros_like(binary_masks, dtype=np.uint8)

    for idx, mask in enumerate(binary_masks.astype(np.uint8)):
        # Find connected components in the binary mask
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8, ltype=cv.CV_32S)

        # If no additional components are found, continue to the next mask
        if num_labels <= 1:
            largest_component_masks[idx] = mask
            continue

        # Find the index of the largest connected component (excluding the background component)
        largest_component_index = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1

        # Create a new mask with only the largest connected component
        largest_component_masks[idx] = (labels == largest_component_index).astype(np.uint8)

    return largest_component_masks


def fill_holes(binary_masks: np.ndarray) -> np.ndarray:
    # Initialize the filled_masks array to zeros with the same shape and type as binary_masks
    filled_masks = np.zeros_like(binary_masks, dtype=np.uint8)

    for idx, mask in enumerate(binary_masks.astype(np.uint8)):
        # Find all enclosed contours in the binary mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the corresponding index in filled_masks with filled interiors
        cv.drawContours(filled_masks[idx], contours, -1, 1, -1)

    return filled_masks


def fit_ellipse(binary_masks: np.ndarray) -> np.ndarray:
    ellipse_masks = np.zeros_like(binary_masks, dtype=np.uint8)

    for idx, mask in enumerate(binary_masks.astype(np.uint8)):
        # Find the contours of the binary mask (there should be only one significant contour for fitting an ellipse)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            ellipse_masks[idx] = mask
            continue

        # Get the largest (only) contour (it needs at least 5 points to fit an ellipse)
        max_contour = max(contours, key=cv.contourArea)
        if len(max_contour) < 5:
            ellipse_masks[idx] = mask
            continue

        # Fit an ellipse to the largest contour and draw it on a blank mask
        ellipse = cv.fitEllipse(max_contour)
        cv.ellipse(ellipse_masks[idx], ellipse, (1,), -1)  # Fill ellipse with 1's

        # Preserve original boundaries by combining the ellipse mask with the original mask
        ellipse_masks[idx] = np.logical_or(ellipse_masks[idx], mask).astype(np.uint8)

    return ellipse_masks


def douglas_peucker(binary_masks: np.ndarray, epsilon: float = 3.0) -> np.ndarray:
    dp_masks = np.zeros_like(binary_masks, dtype=np.uint8)

    for idx, mask in enumerate(binary_masks.astype(np.uint8)):
        # Find the contours
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Create a new binary mask
        dp_mask = np.zeros(mask.shape, dtype=np.uint8)

        for contour in contours:
            # Approximate contour with Douglas-Peucker algorithm
            approx = cv.approxPolyDP(contour, epsilon, closed=True)

            # Draw the approximated contour on the mask
            cv.drawContours(dp_mask, [approx], -1, color=1, thickness=-1)

        dp_masks[idx] = dp_mask

    return dp_masks


def smooth_contours(binary_masks: np.ndarray, s: float = 0.5) -> np.ndarray:
    smoothened_masks = np.zeros_like(binary_masks, dtype=np.uint8)

    for idx, mask in enumerate(binary_masks.astype(np.uint8)):
        # Find contours in the binary mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            smoothened_masks[idx] = mask
            continue

        # Create an empty mask for displaying the smoothed contours
        smoothed_mask = np.zeros_like(mask)

        for contour in contours:
            # Convert the contour points to float32 for spline interpolation
            contour_points = contour.squeeze().astype(np.float32)

            # Use spline interpolation to smooth the contour
            tck, u = splprep([contour_points[:, 0], contour_points[:, 1]], s=s)
            smoothed_points = np.column_stack(splev(u, tck))

            # Draw the smoothed contour on the mask
            cv.drawContours(smoothed_mask, [smoothed_points.astype(np.int32)], -1, 1, thickness=-1)

        smoothened_masks[idx] = smoothed_mask

    return smoothened_masks


def snakes(binary_masks: np.ndarray, alpha: float = 0.1, beta: float = 10.5, gamma: float = 10.5) -> np.ndarray:
    snake_masks = np.zeros_like(binary_masks, dtype=np.uint8)

    for idx, mask in enumerate(binary_masks.astype(np.uint8)):
        # Find contours in the mask
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        snake_mask = np.zeros_like(mask)

        for contour in contours:
            # Run the Active Contour Model
            snake = active_contour(mask, np.squeeze(contour), alpha=alpha, beta=beta, gamma=gamma)

            # Draw the snake on the mask with thickness -1 (filled)
            cv.drawContours(snake_mask, [snake.astype(np.int32)], -1, color=1, thickness=-1)

        snake_masks[idx] = snake_mask

    return snake_masks


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

    # U = unary_from_softmax(probab)  # shape of probabilities must be (n_classes, ...)
    U = unary_from_labels(probab, n_labels, gt_prob=0.7, zero_unsure=True)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(**gaussian_kwargs)  # adds the color-independent term, features are the locations only
    d.addPairwiseBilateral(**bilateral_kwargs)  # adds the color-dependent term, i.e. features are (x, y, r, g, b)

    Q = d.inference(n_iterations)
    crf_mask = np.argmax(Q, axis=0).reshape((width, height))

    return crf_mask
