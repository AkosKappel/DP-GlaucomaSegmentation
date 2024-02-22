import cv2 as cv
import numpy as np

__all__ = [
    'to_red_channel', 'to_green_channel', 'to_blue_channel', 'to_gray_channel',
    'binarize', 'extract_disc_mask', 'extract_cup_mask',
    'distance_transform', 'boundary_transform', 'polar_transform', 'inverse_polar_transform',
    'occlude', 'sharpen', 'sharpening', 'otsu', 'clahe', 'histogram_equalization',
    'calculate_rgb_cumsum', 'source_to_target_correction', 'get_bounding_box',
]


def to_red_channel(image: np.ndarray, **kwargs) -> np.ndarray:
    return image[:, :, 0]


def to_green_channel(image: np.ndarray, **kwargs) -> np.ndarray:
    return image[:, :, 1]


def to_blue_channel(image: np.ndarray, **kwargs) -> np.ndarray:
    return image[:, :, 2]


def to_gray_channel(image: np.ndarray, **kwargs) -> np.ndarray:
    if image.ndim == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image


def binarize(mask: np.ndarray, labels: int | list[int], **kwargs) -> np.ndarray:
    assert mask.ndim == 2, f'Mask must be grayscale, but got shape: {mask.shape}'
    if isinstance(labels, int):
        return np.where(mask > labels, 1, 0).astype(np.uint8)
    return np.isin(mask, labels).astype(np.uint8)


def extract_disc_mask(mask: np.ndarray, **kwargs) -> np.ndarray:
    return binarize(mask, labels=[1, 2])


def extract_cup_mask(mask: np.ndarray, **kwargs) -> np.ndarray:
    return binarize(mask, labels=[2])


def extract_rim_mask(mask: np.ndarray, **kwargs) -> np.ndarray:
    return binarize(mask, labels=[1])


def distance_transform(mask: np.ndarray, mode: str = 'L2', normalize: bool = True, invert: bool = False,
                       add_fg: bool = True, add_bg: bool = True, negate_fg: bool = False, negate_bg: bool = False):
    assert mode in ('L1', 'L2'), f'Invalid distance transform mode: {mode}'
    assert add_fg or add_bg, 'At least one of add_fg and add_bg must be True'
    assert mask.ndim == 2, f'Invalid mask shape: {mask.shape}. Must be 2D (grayscale)'

    fg = np.where(mask > 0, 1, 0).astype(np.uint8)
    bg = np.where(mask == 0, 1, 0).astype(np.uint8)

    if mode == 'L1':
        fg = cv.distanceTransform(fg, cv.DIST_L1, 3)
        bg = cv.distanceTransform(bg, cv.DIST_L1, 3)
    elif mode == 'L2':
        fg = cv.distanceTransform(fg, cv.DIST_L2, 3)
        bg = cv.distanceTransform(bg, cv.DIST_L2, 3)

    if normalize:
        fg = cv.normalize(fg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        bg = cv.normalize(bg, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    if invert:
        fg = np.max(fg) - fg
        bg = np.max(bg) - bg

    if negate_fg:
        fg = -fg
    if negate_bg:
        bg = -bg

    if add_fg and add_bg:
        mask = fg + bg
    elif add_fg:
        mask = fg
    elif add_bg:
        mask = bg

    return (mask * 255).astype(np.uint8)


def boundary_transform(mask: np.ndarray, kernel_size: int = 5, structure: int = cv.MORPH_ELLIPSE, iterations: int = 1):
    assert mask.ndim == 2, f'Invalid mask shape: {mask.shape}. Must be 2D (grayscale)'

    kernel = cv.getStructuringElement(structure, (kernel_size, kernel_size))
    dilation = cv.dilate(mask, kernel, iterations=iterations)
    erosion = cv.erode(mask, kernel, iterations=iterations)

    # Extracts the boundary of the mask as a difference between the dilation and erosion
    boundary = dilation - erosion
    return (boundary * 255).astype(np.uint8)


def polar_transform(cartesian_image: np.ndarray, radius_ratio: float = 1.0, **kwargs) -> np.ndarray:
    height, width = cartesian_image.shape[:2]
    center = (width // 2, height // 2)

    # Linear interpolation between inner and outer radius
    inner_radius = np.min([width, height]) / 2.0
    outer_radius = np.sqrt(((width / 2.0) ** 2.0) + ((height / 2.0) ** 2.0))
    radius = inner_radius + (outer_radius - inner_radius) * radius_ratio

    polar_image = cv.linearPolar(cartesian_image, center, radius, cv.WARP_FILL_OUTLIERS)
    polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    return polar_image


def inverse_polar_transform(polar_image: np.ndarray, radius_ratio: float = 1.0, **kwargs) -> np.ndarray:
    polar_image = cv.rotate(polar_image, cv.ROTATE_90_CLOCKWISE)
    height, width = polar_image.shape[:2]
    center = (width // 2, height // 2)

    inner_radius = np.min([width, height]) / 2.0
    outer_radius = np.sqrt(((width / 2.0) ** 2.0) + ((height / 2.0) ** 2.0))
    radius = inner_radius + (outer_radius - inner_radius) * radius_ratio

    cartesian_image = cv.linearPolar(polar_image, center, radius, cv.WARP_INVERSE_MAP | cv.WARP_FILL_OUTLIERS)
    return cartesian_image


def occlude(image: np.ndarray, p: float = 0.5, occlusion_size: int = 32, occlusion_value: int = 0, **kwargs):
    if np.random.rand() > p:
        return image

    h, w = image.shape[:2]
    assert h >= occlusion_size and w >= occlusion_size, \
        f'Image size ({h}x{w}) must be greater than occlusion size ({occlusion_size}x{occlusion_size})'

    x = np.random.randint(0, w - occlusion_size)
    y = np.random.randint(0, h - occlusion_size)

    final_image = image.copy()
    final_image[y:y + occlusion_size, x:x + occlusion_size] = occlusion_value
    return final_image


def sharpen(image: np.ndarray, p: float = 0.5, connectivity: int = 4, **kwargs):
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
        raise ValueError(f'Invalid connectivity: {connectivity!r}. Must be 4 or 8.')
    return cv.filter2D(image, -1, kernel)


def sharpening(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0, threshold: int = 0):
    blurred = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)

    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened


def otsu(gray_image: np.ndarray, ignore_value: int = None) -> (int, np.ndarray):
    if gray_image.ndim == 3:
        gray_image = cv.cvtColor(gray_image, cv.COLOR_BGR2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image * 255).astype(np.uint8)

    pixel_number = gray_image.shape[0] * gray_image.shape[1]
    mean_weight = 1.0 / pixel_number

    hist, bins = np.histogram(gray_image, np.arange(0, 257))
    intensity_arr = np.arange(256)

    final_thresh = -1
    final_value = -1

    if ignore_value is not None:
        hist[ignore_value] = 0

    for t in bins[1:-1]:
        pcb = np.sum(hist[:t])
        pcf = np.sum(hist[t:])

        if pcb == 0:
            continue
        if pcf == 0:
            break

        wb = pcb * mean_weight
        wf = pcf * mean_weight

        mub = np.sum(intensity_arr[:t] * hist[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * hist[t:]) / float(pcf)
        value = wb * wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value

    final_image = gray_image.copy()
    final_image[gray_image > final_thresh] = 255
    final_image[gray_image < final_thresh] = 0
    return final_thresh - 1, final_image


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8), channel: str = 'grey'):
    if image.ndim == 3:
        if channel == 'grey':
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif channel == 'red':
            image = image[:, :, 2]
        elif channel == 'green':
            image = image[:, :, 1]
        elif channel == 'blue':
            image = image[:, :, 0]
        else:
            raise ValueError(f'Invalid image channel mode: {channel}')

    c = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return c.apply(image.copy())


def histogram_equalization(image: np.ndarray, channel: str = 'grey'):
    if image.shape[-1] == 3:
        if channel == 'grey':
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif channel == 'red':
            image = image[:, :, 2]
        elif channel == 'green':
            image = image[:, :, 1]
        elif channel == 'blue':
            image = image[:, :, 0]
        else:
            raise ValueError(f'Invalid image channel mode: {channel}')

    return cv.equalizeHist(image)


def calculate_rgb_cumsum(image: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
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


def source_to_target_correction(source_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
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
