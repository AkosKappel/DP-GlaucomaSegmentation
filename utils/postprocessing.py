import cv2 as cv
import numpy as np

__all__ = [
    'separate_disc_and_cup_mask',
    'keep_largest_component', 'apply_largest_component_selection',
    'fill_holes', 'apply_hole_filling',
    'fit_ellipse', 'apply_ellipse_fitting',
]


def separate_disc_and_cup_mask(mask):
    # OD: 1, OC: 2, BG: 0
    od_mask = np.where(mask >= 1, 1, 0).astype(np.uint8)
    oc_mask = np.where(mask >= 2, 1, 0).astype(np.uint8)
    return od_mask, oc_mask


def keep_largest_component(binary_mask):
    # Find connected components in the binary mask
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_mask)

    # Find the index of the largest connected component (excluding the background component)
    largest_component_index = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1

    # Create a new mask with only the largest connected component
    largest_component_mask = (labels == largest_component_index).astype(np.uint8)

    return largest_component_mask


def apply_largest_component_selection(mask):
    masks = separate_disc_and_cup_mask(mask)
    result_mask = np.zeros_like(mask, dtype=np.uint8)

    # Find the largest connected component in each mask and then combine them back into one mask
    for mask in masks:
        largest_component_mask = keep_largest_component(mask)
        result_mask += largest_component_mask

    return result_mask


def fill_holes(binary_mask):
    # Find all enclosed contours in the binary mask
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for drawing the areas with filled holes
    filled_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # Draw the contours on the blank mask with filled interiors
    for contour in contours:
        cv.drawContours(filled_mask, [contour], 0, 1, -1)

    return filled_mask


def apply_hole_filling(mask):
    masks = separate_disc_and_cup_mask(mask)
    result_mask = np.zeros_like(mask, dtype=np.uint8)

    for i, mask in enumerate(masks, start=1):
        # Fill the holes in the mask
        filled_mask = fill_holes(mask)

        # Join the filled masks into a single mask
        result_mask[filled_mask == 1] = i

    return result_mask


def fit_ellipse(binary_mask):
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


def apply_ellipse_fitting(mask):
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
