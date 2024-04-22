import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from tqdm.notebook import tqdm

__all__ = ['Thresholding']


class Thresholding:

    def __init__(self, margin: int = 50, channel: int = 2, crop_size: int | float = 256, k_size: int = 35):
        self.margin = margin
        self.channel = channel
        self.crop_size = crop_size
        self.k_size = (k_size, k_size)

    def show(self, image):
        if isinstance(image, str):
            image = cv.imread(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            image = image.copy()
        x, y, w, h = self.apply(image)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 10)
        plt.imshow(image)
        plt.title('Thresholding')
        plt.show()

    def generate_dataset(self, src_images_dir, src_masks_dir, dst_images_dir, dst_masks_dir):
        src_images_dir = Path(src_images_dir)
        src_masks_dir = Path(src_masks_dir)
        dst_images_dir = Path(dst_images_dir)
        dst_masks_dir = Path(dst_masks_dir)
        overlay_dir = dst_images_dir / '../Overlaid_Threshold_Images'

        assert src_images_dir.exists()
        assert src_masks_dir.exists()

        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_masks_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([f for f in os.listdir(src_images_dir) if not f.startswith('.')])
        masks = sorted([f for f in os.listdir(src_masks_dir) if not f.startswith('.')])

        title = 'Generating thresholding dataset'
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

            cv.imwrite(str(dst_images_dir / image_name), cropped_image)
            cv.imwrite(str(dst_masks_dir / mask_name), cropped_mask)
            cv.imwrite(str(overlay_dir / image_name), overlay_image)

            pbar.set_postfix({'coverage': f'{total_coverage * 100 / i:.2f}%'})
        print(f'Final average coverage: {total_coverage * 100 / len(images):.2f}%')

    def apply(self, image):
        # Crop the image to remove the black background
        if isinstance(self.crop_size, float):
            crop_size = int(self.crop_size * image.shape[0])
        if self.crop_size > 0:
            image = image[self.crop_size: -self.crop_size, self.crop_size: -self.crop_size, :]

        # Get the specified channel (red, green, blue, or grey)
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY) if self.channel == -1 else image[..., self.channel]

        # Otsu thresholding
        _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Dilate the mask
        kernel = cv.getStructuringElement(cv.MORPH_RECT, self.k_size)
        image = cv.dilate(image, kernel)

        # Get bounding box of largest connected component
        image = keep_largest_component(image)
        top_left_x, top_left_y, width, height = get_bounding_box(image)

        top_left_x += self.crop_size - self.margin
        top_left_y += self.crop_size - self.margin
        width += 2 * self.margin
        height += 2 * self.margin

        return top_left_x, top_left_y, width, height


def keep_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    # Find connected components in the binary mask
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_mask)

    # Find the index of the largest connected component (excluding the background component)
    largest_component_index = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1

    # Create a new mask with only the largest connected component
    largest_component_mask = (labels == largest_component_index).astype(np.uint8)

    return largest_component_mask


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
