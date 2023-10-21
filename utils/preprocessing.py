import cv2 as cv
import numpy as np
from pathlib import Path

__all__ = [
    'extract_optic_disc', 'extract_optic_cup',
    'calculate_rgb_cumsum', 'source_to_target_correction',
    'localize_roi', 'get_bounding_box', 'otsu', 'clahe', 'histogram_equalization',
    'split_rgb_channels', 'to_greyscale', 'brightness_contrast', 'sharpen', 'blur',
    'split_train_val_test', 'distance_transform', 'boundary_transform',
]


def extract_optic_disc(src_dir: Path, dst_dir: Path, value: int = 0):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        disc = np.where(img > value, 1, 0).astype(np.uint8)

        cv.imwrite(str(dst_dir / img_path.name), disc)
        num += 1

    print(f'Extracted optic disc from {num} images from {src_dir} and saved to {dst_dir}')


def extract_optic_cup(src_dir: Path, dst_dir: Path, value: int = 1):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        cup = np.where(img > value, 1, 0).astype(np.uint8)

        cv.imwrite(str(dst_dir / img_path.name), cup)
        num += 1

    print(f'Extracted optic cup from {num} images from {src_dir} and saved to {dst_dir}')


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


def source_to_target_correction(source, target):
    cdf_source_red, cdf_source_green, cdf_source_blue = calculate_rgb_cumsum(source)
    cdf_target_red, cdf_target_green, cdf_target_blue = calculate_rgb_cumsum(target)

    # interpolate CDFs
    red_lookup = np.interp(cdf_source_red, cdf_target_red, np.arange(256))
    green_lookup = np.interp(cdf_source_green, cdf_target_green, np.arange(256))
    blue_lookup = np.interp(cdf_source_blue, cdf_target_blue, np.arange(256))

    # apply the lookup tables to the source image
    corrected = source.copy()
    corrected[..., 0] = red_lookup[source[..., 0]].reshape(source.shape[:2])
    corrected[..., 1] = green_lookup[source[..., 1]].reshape(source.shape[:2])
    corrected[..., 2] = blue_lookup[source[..., 2]].reshape(source.shape[:2])

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


def otsu(gray, ignore_value=None):
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weight = 1.0 / pixel_number

    his, bins = np.histogram(gray, np.arange(0, 257))
    intensity_arr = np.arange(256)

    final_thresh = -1
    final_value = -1

    if ignore_value is not None:
        his[ignore_value] = 0

    for t in bins[1:-1]:
        pcb = np.sum(his[:t])
        pcf = np.sum(his[t:])

        if pcb == 0:
            continue
        if pcf == 0:
            break

        wb = pcb * mean_weight
        wf = pcf * mean_weight

        mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
        muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
        value = wb * wf * (mub - muf) ** 2

        if value > final_value:
            final_thresh = t
            final_value = value

    final_img = gray.copy()
    final_img[gray > final_thresh] = 255
    final_img[gray < final_thresh] = 0
    return final_thresh - 1, final_img


def localize_roi(src_dir: Path, dst_dir: Path):
    # TODO: add ROI localization technique from article
    pass


def clahe(src_dir: Path, dst_dir: Path, mode='grey', clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_COLOR)

        if img.shape[-1] == 3:
            if mode == 'grey':
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            elif mode == 'red':
                img = img[:, :, 2]
            elif mode == 'green':
                img = img[:, :, 1]
            elif mode == 'blue':
                img = img[:, :, 0]
            else:
                raise ValueError(f'Invalid image channel mode: {mode}')

        c = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img = c.apply(img)
        cv.imwrite(str(dst_dir / img_path.name), img)
        num += 1

    print(f'CLAHE ({mode}) applied to {num} images from {src_dir} and saved to {dst_dir}')


def histogram_equalization(src_dir: Path, dst_dir: Path, mode='grey'):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'
    assert mode in ['grey', 'red', 'green', 'blue'], f'Invalid image channel mode: {mode}'

    dst_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_COLOR)

        if img.shape[-1] == 3:
            if mode == 'grey':
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            elif mode == 'red':
                img = img[:, :, 2]
            elif mode == 'green':
                img = img[:, :, 1]
            elif mode == 'blue':
                img = img[:, :, 0]

        img = cv.equalizeHist(img)
        cv.imwrite(str(dst_dir / img_path.name), img)
        num += 1

    print(f'Histogram equalization ({mode}) applied to {num} images from {src_dir} and saved to {dst_dir}')


def split_rgb_channels(src_dir: Path, dst_dir: Path,
                       red_name: str = 'red', green_name: str = 'green', blue_name: str = 'blue'):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    red_dir = dst_dir / red_name
    green_dir = dst_dir / green_name
    blue_dir = dst_dir / blue_name

    red_dir.mkdir(parents=True, exist_ok=True)
    green_dir.mkdir(parents=True, exist_ok=True)
    blue_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_COLOR)

        if img.shape[-1] == 3:
            img_r, img_g, img_b = cv.split(img)
            cv.imwrite(str(red_dir / img_path.name), img_r)
            cv.imwrite(str(green_dir / img_path.name), img_g)
            cv.imwrite(str(blue_dir / img_path.name), img_b)
            num += 1

    print(f'RGB channels split for {num} images from {src_dir} and saved to {red_dir}, {green_dir}, {blue_dir}')


def to_greyscale(src_dir: Path, dst_dir: Path):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_COLOR)

        if img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.imwrite(str(dst_dir / img_path.name), img)
            num += 1

    print(f'Converted {num} images from {src_dir} to greyscale and saved to {dst_dir}')


def brightness_contrast(src_dir: Path, dst_dir: Path, alpha: float = 1.0, beta: float = 0.0):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_COLOR)
        img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
        cv.imwrite(str(dst_dir / img_path.name), img)
        num += 1

    print(f'Applied brightness/contrast to {num} images from {src_dir} and saved to {dst_dir}')


def sharpen(src_dir: Path, dst_dir: Path, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0,
            threshold: float = 0):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1],
    ])

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_COLOR)
        # sharpened = cv.filter2D(img, -1, kernel)

        blurred = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        sharpened = float(amount + 1) * img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)

        if threshold > 0:
            low_contrast_mask = np.absolute(img - blurred) < threshold
            np.copyto(sharpened, img, where=low_contrast_mask)

        cv.imwrite(str(dst_dir / img_path.name), sharpened)

    print(f'Applied sharpening to {num} images from {src_dir} and saved to {dst_dir}')


def blur(src_dir: Path, dst_dir: Path, kernel_size: int = 5, sigma: float = 1.0):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for img_path in src_dir.iterdir():
        if not img_path.is_file():
            continue

        img = cv.imread(str(img_path), cv.IMREAD_COLOR)
        blurred = cv.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        cv.imwrite(str(dst_dir / img_path.name), blurred)
        num += 1

    print(f'Applied Gaussian blur to {num} images from {src_dir} and saved to {dst_dir}')


def split_train_val_test(img_src_dir: Path, img_dst_dir: Path, mask_src_dir: Path, mask_dst_dir: Path,
                         train: float = 0.8, val: float = 0.1, test: float = 0.1, seed: int = 4118,
                         train_name: str = 'train', val_name: str = 'val', test_name: str = 'test'):
    assert img_src_dir.exists(), f'{img_src_dir} not found'
    assert img_src_dir.is_dir(), f'{img_src_dir} is not a directory'
    assert mask_src_dir.exists(), f'{mask_src_dir} not found'
    assert mask_src_dir.is_dir(), f'{mask_src_dir} is not a directory'

    img_train_dir = img_dst_dir / train_name
    img_val_dir = img_dst_dir / val_name
    img_test_dir = img_dst_dir / test_name

    mask_train_dir = mask_dst_dir / train_name
    mask_val_dir = mask_dst_dir / val_name
    mask_test_dir = mask_dst_dir / test_name

    img_train_dir.mkdir(parents=True, exist_ok=True)
    img_val_dir.mkdir(parents=True, exist_ok=True)
    img_test_dir.mkdir(parents=True, exist_ok=True)

    mask_train_dir.mkdir(parents=True, exist_ok=True)
    mask_val_dir.mkdir(parents=True, exist_ok=True)
    mask_test_dir.mkdir(parents=True, exist_ok=True)

    files = list(img_src_dir.iterdir())
    np.random.seed(seed)
    np.random.shuffle(files)

    train_num = 0
    val_num = 0
    test_num = 0
    num = 0
    for file in files:
        if not file.is_file():
            continue

        if num < train:
            img_dir = img_train_dir
            mask_dir = mask_train_dir
            train_num += 1
        elif num < train + val:
            img_dir = img_val_dir
            mask_dir = mask_val_dir
            val_num += 1
        else:
            img_dir = img_test_dir
            mask_dir = mask_test_dir
            test_num += 1

        img = cv.imread(str(file), cv.IMREAD_COLOR)
        mask = cv.imread(str(file), cv.IMREAD_GRAYSCALE)

        cv.imwrite(str(img_dir / file.name), img)
        cv.imwrite(str(mask_dir / file.name), mask)
        num += 1

    print(f'Split {num} images from {img_src_dir} into train ({train}), val ({val}), test ({test}) sets and saved to '
          f'{img_dst_dir / "train"}, {img_dst_dir / "val"}, {img_dst_dir / "test"}')


def distance_transform(src_dir: Path, dst_dir: Path, mode: str = 'L2', normalize: bool = True, invert: bool = False,
                       add_fg: bool = True, add_bg: bool = True, negate_fg: bool = False, negate_bg: bool = False):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'
    assert mode in ['L1', 'L2'], f'Invalid distance transform mode: {mode}'
    assert add_fg or add_bg, 'At least one of add_fg and add_bg must be True'

    dst_dir.mkdir(parents=True, exist_ok=True)

    num = 0
    for mask_path in src_dir.iterdir():
        if not mask_path.is_file():
            continue

        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

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

        # Convert to uint8
        mask = (mask * 255).astype(np.uint8)

        cv.imwrite(str(dst_dir / mask_path.name), mask)
        num += 1

    print(f'Distance transform ({mode}) applied to {num} images from {src_dir} and saved to {dst_dir}')


def boundary_transform(src_dir: Path, dst_dir: Path, kernel_size: int = 5):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))

    num = 0
    for mask_path in src_dir.iterdir():
        if not mask_path.is_file():
            continue

        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        dilation = cv.dilate(mask, kernel, iterations=1)
        erosion = cv.erode(mask, kernel, iterations=1)

        # Extracts the boundary of the mask as a difference between the dilation and erosion
        boundary = dilation - erosion

        # Convert to uint8
        boundary = (boundary * 255).astype(np.uint8)

        cv.imwrite(str(dst_dir / mask_path.name), boundary)
        num += 1

    print(f'Boundary transform applied to {num} images from {src_dir} and saved to {dst_dir}')


if __name__ == '__main__':
    cropped_images_dir = Path('../data/ORIGA/images_Cropped')
    cropped_masks_dir = Path('../data/ORIGA/masks_Cropped')

    disc_masks_dir = Path('../data/preprocessed/ORIGA/disc')
    cup_masks_dir = Path('../data/preprocessed/ORIGA/cup')

    # Images
    extract_optic_disc(cropped_masks_dir, disc_masks_dir)
    extract_optic_cup(cropped_masks_dir, cup_masks_dir)

    clahe(cropped_images_dir, Path('../data/preprocessed/ORIGA/clahe-grey'))
    clahe(cropped_images_dir, Path('../data/preprocessed/ORIGA/clahe-red'), mode='red')
    clahe(cropped_images_dir, Path('../data/preprocessed/ORIGA/clahe-green'), mode='green')
    clahe(cropped_images_dir, Path('../data/preprocessed/ORIGA/clahe-blue'), mode='blue')

    histogram_equalization(cropped_images_dir, Path('../data/preprocessed/ORIGA/hist-eq-grey'))
    histogram_equalization(cropped_images_dir, Path('../data/preprocessed/ORIGA/hist-eq-red'), mode='red')
    histogram_equalization(cropped_images_dir, Path('../data/preprocessed/ORIGA/hist-eq-green'), mode='green')
    histogram_equalization(cropped_images_dir, Path('../data/preprocessed/ORIGA/hist-eq-blue'), mode='blue')

    split_rgb_channels(cropped_images_dir, Path('../data/preprocessed/ORIGA/'))
    to_greyscale(cropped_images_dir, Path('../data/preprocessed/ORIGA/greyscale'))

    brightness_contrast(cropped_images_dir, Path('../data/preprocessed/ORIGA/brightness-contrast'))
    sharpen(cropped_images_dir, Path('../data/preprocessed/ORIGA/sharpened'))
    blur(cropped_images_dir, Path('../data/preprocessed/ORIGA/blurred'))

    # Masks
    distance_transform(disc_masks_dir, Path('../data/preprocessed/ORIGA/dist-transform-disc'))
    distance_transform(cup_masks_dir, Path('../data/preprocessed/ORIGA/dist-transform-cup'))

    boundary_transform(disc_masks_dir, Path('../data/preprocessed/ORIGA/boundary-transform-disc'))
    boundary_transform(cup_masks_dir, Path('../data/preprocessed/ORIGA/boundary-transform-cup'))
