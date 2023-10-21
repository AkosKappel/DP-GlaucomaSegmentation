import cv2 as cv
import numpy as np


def gaussian_kernel(width: int, height: int = None, sigma: float = None) -> np.ndarray:
    if height is None:
        height = width

    shape = (height, width)
    size = np.max(shape)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]

    sigma = size / 2 if sigma is None else size / sigma
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    if shape != (size, size):
        kernel = cv.resize(kernel, shape[::-1], interpolation=cv.INTER_LINEAR)

    return kernel / kernel.max()


def circular_kernel(width: int, height: int = None) -> np.ndarray:
    if height is None:
        height = width

    center_x, center_y = width // 2, height // 2
    y_coords, x_coords = np.ogrid[:height, :width]

    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    kernel = 1 - (distances / max_distance)
    return kernel


def parabolic_kernel(width: int, height: int = None) -> np.ndarray:
    if height is None:
        height = width

    center_x, center_y = width // 2, height // 2
    y_coords, x_coords = np.ogrid[:height, :width]

    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)

    kernel = 1 - (distances / max_distance) ** 2
    return kernel
