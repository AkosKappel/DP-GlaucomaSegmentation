import cv2 as cv
import numpy as np
from pathlib import Path

__all__ = [
    'extract_optic_disc', 'extract_optic_cup',
    'otsu', 'clahe', 'histogram_equalization',
    'split_rgb_channels', 'to_greyscale', 'brightness_contrast', 'sharpening',
    'distance_transform', 'boundary_transform',
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


def otsu(gray_image, ignore_value=None):
    pixel_number = gray_image.shape[0] * gray_image.shape[1]
    mean_weight = 1.0 / pixel_number

    his, bins = np.histogram(gray_image, np.arange(0, 257))
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

    final_img = gray_image.copy()
    final_img[gray_image > final_thresh] = 255
    final_img[gray_image < final_thresh] = 0
    return final_thresh - 1, final_img


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


def sharpening(src_dir: Path, dst_dir: Path, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0,
               threshold: float = 0):
    assert src_dir.exists(), f'{src_dir} not found'
    assert src_dir.is_dir(), f'{src_dir} is not a directory'

    dst_dir.mkdir(parents=True, exist_ok=True)

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
