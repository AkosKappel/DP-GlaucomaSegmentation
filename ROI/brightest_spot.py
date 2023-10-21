import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from tqdm.notebook import tqdm
from .kernels import circular_kernel, gaussian_kernel, parabolic_kernel


class BrightestSpot:

    def __init__(self, width: int = 512, height: int = 512, channel: int = -1,
                 dampening: str = 'circular', k_size: int = 65, shape: tuple | int = None):
        self.width = width
        self.height = height
        self.channel = channel
        self.dampening = dampening.lower()
        self.k_size = (k_size, k_size) if k_size is not None else None
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
        overlay_dir = dst_images_dir / '../Overlaid_BrightestSpot_Images'

        assert src_images_dir.exists()
        assert src_masks_dir.exists()

        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_masks_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([f for f in os.listdir(src_images_dir) if not f.startswith('.')])
        masks = sorted([f for f in os.listdir(src_masks_dir) if not f.startswith('.')])

        title = 'Generating brightest spot dataset'
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

            cv.imwrite(str(dst_images_dir / image_name), cropped_image)
            cv.imwrite(str(dst_masks_dir / mask_name), cropped_mask)
            cv.imwrite(str(overlay_dir / image_name), cropped_image)

            pbar.set_postfix({'coverage': f'{total_coverage * 100 / i:.2f}%'})

    def apply(self, image):
        if isinstance(image, str):
            image = cv.imread(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Convert to channel and normalize
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if self.channel == -1 else image[..., self.channel]
        image = image.astype(np.float32) / 255.0

        # Dampen the weights close to the corners
        if self.dampening == 'circular':
            damping_map = circular_kernel(image.shape[1], image.shape[0])
        elif self.dampening == 'gaussian':
            damping_map = gaussian_kernel(image.shape[1], image.shape[0])
        elif self.dampening == 'parabolic':
            damping_map = parabolic_kernel(image.shape[1], image.shape[0])
        else:
            damping_map = np.ones_like(image)
        image = image * damping_map

        # Apply Gaussian blur to smooth out the image
        blurred_img = cv.GaussianBlur(image, self.k_size, cv.BORDER_DEFAULT)

        # Find the brightest spot
        max_val = np.max(blurred_img)
        max_idx = np.where(blurred_img == max_val)

        # Average the brightest spots in case there are multiple
        max_idx = np.mean(max_idx, axis=1)

        # Find the centroid of the brightest spot
        center_x = np.mean(max_idx[1])
        center_y = np.mean(max_idx[0])

        return int(center_x - self.width / 2), int(center_y - self.height / 2), self.width, self.height
