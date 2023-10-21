import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from tqdm.notebook import tqdm
from .kernels import circular_kernel, gaussian_kernel, parabolic_kernel


class IntensityWeightedCentroid:

    def __init__(self, width: int = 512, height: int = 512, channel: int = -1,
                 equalize: bool = True, clahe: bool = True, square: bool = True, dampening: str = 'circular',
                 k_size: int = None, quantile: float = None, shape: tuple | int = None):
        self.width = width
        self.height = height
        self.channel = channel
        self.dampening = dampening.lower()
        self.equalize = equalize
        self.clahe = clahe
        self.square = square
        self.k_size = (k_size, k_size) if k_size is not None else None
        self.quantile = quantile
        self.shape = shape if shape is not None else (height, width)

    def show(self, image):
        if isinstance(image, str):
            image = cv.imread(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            image = image.copy()
        x, y, w, h = self.apply(image)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 10)
        plt.imshow(image)
        plt.show()

    def generate_dataset(self, src_images_dir, src_masks_dir, dst_images_dir, dst_masks_dir):
        src_images_dir = Path(src_images_dir)
        src_masks_dir = Path(src_masks_dir)
        dst_images_dir = Path(dst_images_dir)
        dst_masks_dir = Path(dst_masks_dir)
        overlay_dir = dst_images_dir / '../Overlaid_IntensityWeightedCentroid_Images'

        assert src_images_dir.exists()
        assert src_masks_dir.exists()

        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_masks_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([f for f in os.listdir(src_images_dir) if not f.startswith('.')])
        masks = sorted([f for f in os.listdir(src_masks_dir) if not f.startswith('.')])

        title = 'Generating intensity weighted centroid dataset'
        total_coverage = 0
        pbar = tqdm(zip(images, masks), total=len(images), desc=title)
        for i, (image_name, mask_name) in enumerate(pbar, start=1):
            image = cv.imread(str(src_images_dir / image_name))
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mask = cv.imread(str(src_masks_dir / mask_name), cv.IMREAD_GRAYSCALE)

            x, y, w, h = self.apply(image)

            cropped_image = image[y:y + h, x:x + w]
            cropped_mask = mask[y:y + h, x:x + w]

            coverage = np.sum(cropped_mask) / np.sum(mask)
            total_coverage += coverage

            cropped_image = cv.cvtColor(cropped_image, cv.COLOR_RGB2BGR)

            overlay_image = cropped_image.copy()
            cropped_mask = np.repeat(cropped_mask[..., np.newaxis], 3, axis=-1)
            overlay_image[cropped_mask > 0] = 255
            overlay_image[cropped_mask > 1] = 127

            # Resize
            if self.shape != (h, w):
                cropped_image = cv.resize(cropped_image, self.shape[::-1], interpolation=cv.INTER_AREA)
                cropped_mask = cv.resize(cropped_mask, self.shape[::-1], interpolation=cv.INTER_AREA)
                overlay_image = cv.resize(overlay_image, self.shape[::-1], interpolation=cv.INTER_AREA)

            cv.imwrite(str(dst_images_dir / image_name), cropped_image)
            cv.imwrite(str(dst_masks_dir / mask_name), cropped_mask)
            cv.imwrite(str(overlay_dir / image_name), overlay_image)

            pbar.set_postfix({'coverage': f'{total_coverage * 100 / i:.2f}%'})
        print(f'Final average coverage: {total_coverage * 100 / len(images):.2f}%')

    def apply(self, image, debug: bool = False):
        if isinstance(image, str):
            image = cv.imread(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Select the channel that is going to be used for centroid calculation
        weights = image[..., self.channel] if self.channel != -1 else cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # Equalize histogram to increase contrast
        if self.equalize:
            weights = cv.equalizeHist(weights)

        # Contrast Limited Adaptive Histogram Equalization
        if self.clahe:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
            weights = clahe.apply(weights)

        # Convert to 32-bit float and normalize
        weights = weights.astype(np.float32)
        weights -= weights.min()
        weights = weights / weights.max()

        # Apply Gaussian blur to smooth out the image
        if self.k_size is not None:
            weights = cv.GaussianBlur(weights, self.k_size, 0)

        # Square the weights to increase the contrast
        if self.square:
            weights = weights ** 2
            weights = weights / weights.sum()

        # Dampen the weights close to the corners
        if self.dampening == 'circular':
            damping_map = circular_kernel(weights.shape[1], weights.shape[0])
        elif self.dampening == 'gaussian':
            damping_map = gaussian_kernel(weights.shape[1], weights.shape[0])
        elif self.dampening == 'parabolic':
            damping_map = parabolic_kernel(image.shape[1], image.shape[0])
        else:
            damping_map = np.ones_like(weights)
        weights = weights * damping_map

        # Cut off the bottom % of the weights
        if self.quantile is not None:
            weights[weights < np.quantile(weights, self.quantile)] = 0

        # Find the centroid
        x = np.arange(weights.shape[1])
        y = np.arange(weights.shape[0])
        x, y = np.meshgrid(x, y)

        x_weighted = x * weights
        y_weighted = y * weights
        total_intensity = np.sum(weights)

        x_mean = np.sum(x_weighted) / total_intensity
        y_mean = np.sum(y_weighted) / total_intensity

        if debug:
            _, ax = plt.subplots(2, 3, figsize=(15, 8))
            ax = ax.ravel()
            ax[0].imshow(x)
            ax[1].imshow(y)
            ax[2].imshow(weights)
            ax[3].imshow(x_weighted)
            ax[4].imshow(y_weighted)
            ax[5].imshow(image)
            plt.show()

        return int(x_mean - self.width / 2), int(y_mean - self.height / 2), self.width, self.height
