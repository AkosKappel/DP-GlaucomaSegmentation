import cv2 as cv
import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from pydensecrf.utils import unary_from_softmax, unary_from_labels
from scipy.interpolate import splprep, splev
from skimage.segmentation import active_contour

from modules.preprocessing import polar_transform, inverse_polar_transform

__all__ = [
    'postprocess', 'interprocess', 'tensor_to_numpy', 'numpy_to_tensor', 'cartesian_to_polar', 'polar_to_cartesian',
    'separate_disc_and_cup', 'join_disc_and_cup', 'erosion', 'dilation', 'opening', 'closing',
    'remove_small_components', 'keep_largest_component', 'fill_holes', 'fit_ellipse',
    'douglas_peucker', 'smooth_contours', 'snakes', 'dense_crf',
]


# Main post-processing function
def postprocess(predictions: torch.Tensor, is_in_polar: bool = True, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_shape = predictions.shape

    # Convert the predictions tensor to a numpy array
    predictions = tensor_to_numpy(predictions)

    if is_in_polar:
        predictions = polar_to_cartesian(predictions)

    # Apply postprocessing to the predictions
    discs, cups = separate_disc_and_cup(predictions)

    # Fill holes in the masks
    discs = fill_holes(discs)
    cups = fill_holes(cups)

    # Keep only the largest component in the masks
    discs = keep_largest_component(discs)
    cups = keep_largest_component(cups)

    # Fit an ellipse to the masks
    discs = fit_ellipse(discs, morph='erosion', kernel_size=15)
    cups = fit_ellipse(cups, morph='erosion', kernel_size=15)

    # Join the disc and cup masks
    predictions = join_disc_and_cup(discs, cups)

    if is_in_polar:
        predictions = cartesian_to_polar(predictions)

    # Convert the predictions back to a tensor and return it
    predictions = numpy_to_tensor(predictions, device)

    assert predictions.shape == input_shape, f'Invalid shape: {predictions.shape} != {input_shape}'
    return predictions


# Main post-processing function for cascade architecture (takes place between the first model's predictions
# and the second model's input, to improve the first model's predictions which later modify the input images)
def interprocess(predictions: torch.Tensor, is_in_polar: bool = True, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_shape = predictions.shape  # (B, 1, H, W)

    # Convert the predictions tensor to a numpy array
    predictions = tensor_to_numpy(predictions.squeeze(1))  # (B, H, W)

    if is_in_polar:
        predictions = polar_to_cartesian(predictions)

    # Apply postprocessing to the predictions
    discs, _ = separate_disc_and_cup(predictions)

    # Fill holes in the disc masks
    discs = fill_holes(discs)

    # Fit an ellipse to the disc masks
    discs = fit_ellipse(discs)

    # Enlarge the disc masks a bit
    discs = dilation(discs, kernel_size=9, iterations=1, kernel_shape=cv.MORPH_ELLIPSE)

    if is_in_polar:
        discs = cartesian_to_polar(discs)

    # Convert the predictions back to a tensor and return it
    predictions = numpy_to_tensor(discs, device).unsqueeze(1)  # (B, 1, H, W)

    assert predictions.shape == input_shape, f'Invalid shape: {predictions.shape} != {input_shape}'
    return predictions


def separate_disc_and_cup(masks: np.ndarray) -> (np.ndarray, np.ndarray):
    # Background: 0, Optic disc: 1, Optic cup: 2
    optic_disc_masks = np.where(masks >= 1, 1, 0).astype(np.uint8)
    optic_cup_masks = np.where(masks >= 2, 1, 0).astype(np.uint8)
    return optic_disc_masks, optic_cup_masks


def join_disc_and_cup(optic_disc_masks: np.ndarray, optic_cup_masks: np.ndarray) -> np.ndarray:
    return optic_disc_masks + optic_cup_masks


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(array).to(device)


def cartesian_to_polar(cartesian_masks: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(cartesian_masks, torch.Tensor):
        cartesian_masks = tensor_to_numpy(cartesian_masks)

    polar_masks = np.zeros_like(cartesian_masks)

    for idx, mask in enumerate(cartesian_masks):
        polar_masks[idx] = polar_transform(mask)

    return polar_masks


def polar_to_cartesian(polar_masks: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(polar_masks, torch.Tensor):
        polar_masks = tensor_to_numpy(polar_masks)

    cartesian_masks = np.zeros_like(polar_masks)

    for idx, mask in enumerate(polar_masks):
        cartesian_masks[idx] = inverse_polar_transform(mask)

    return cartesian_masks


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


def fit_ellipse(binary_masks: np.ndarray, morph: str = None, kernel_size: int = 5, iterations: int = 1,
                kernel_shape: int = cv.MORPH_ELLIPSE) -> np.ndarray:
    operations = {
        'erosion': erosion, 'dilation': dilation, 'opening': opening, 'closing': closing,
    }
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
        ellipse_mask = np.zeros_like(mask, dtype=np.uint8)
        cv.ellipse(ellipse_mask, ellipse, (1,), -1)  # Fill ellipse with 1's

        # Enlarge or shrink the ellipse with a morphological operation
        if morph in operations:
            ellipse_mask = operations[morph](
                np.array([ellipse_mask]), kernel_size=kernel_size, iterations=iterations, kernel_shape=kernel_shape
            )[0]

        # Preserve original boundaries by combining the ellipse mask with the original mask
        ellipse_masks[idx] = np.logical_or(ellipse_mask, mask).astype(np.uint8)

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


def dense_crf(predictions: np.ndarray, images: torch.Tensor, n_iterations: int = 5,
              gaussian_kwargs: dict = None, bilateral_kwargs: dict = None) -> np.ndarray:
    # predictions: (B, C, H, W)
    # images: (B, C, H, W)
    images = images.cpu().numpy().transpose((0, 2, 3, 1))

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
            'rgbim': np.ascontiguousarray(images[0], dtype=np.uint8),
            'compat': 10,
            'kernel': dcrf.DIAG_KERNEL,
            'normalization': dcrf.NORMALIZE_SYMMETRIC,
        }

    crf_masks = np.zeros_like(predictions)

    for idx, (prediction, image) in enumerate(zip(predictions, images)):
        n_labels = prediction.shape[0]
        width, height = image.shape[:2]

        d = dcrf.DenseCRF2D(width, height, n_labels)

        # U = unary_from_softmax(probab)  # shape of probabilities must be flat: (n_classes, -1)
        U = unary_from_labels(prediction, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(**gaussian_kwargs)  # adds the color-independent term, features are the locations only
        d.addPairwiseBilateral(**bilateral_kwargs)  # adds the color-dependent term, i.e. features are (x, y, r, g, b)

        Q = d.inference(n_iterations)
        crf_mask = np.argmax(Q, axis=0).reshape((width, height))
        crf_masks[idx] = crf_mask

    return crf_masks
