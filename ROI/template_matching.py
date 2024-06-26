import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from tqdm.notebook import tqdm

__all__ = ['TemplateMatching', 'get_templates', 'align_to_square']


def get_templates(images_dir, n: int = -1, show: bool = False):
    files = sorted([f for f in os.listdir(images_dir) if not f.startswith('.')])

    templates = []
    for idx in range(n if n > 0 else len(files)):
        template_file = os.path.join(images_dir, files[idx])
        template = cv.imread(template_file, cv.IMREAD_COLOR)

        if template is None:
            raise ValueError(f'Could not read template file: {template_file}')

        template = cv.cvtColor(template, cv.COLOR_BGR2RGB)
        templates.append(template)

    if show:
        ncols = min(len(templates), 6)
        nrows = (len(templates) - 1) // ncols + 1

        _, ax = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        ax = ax.flatten()

        for i in range(len(templates), nrows * ncols):
            ax[i].axis('off')

        for i, template in enumerate(templates, start=0):
            ax[i].imshow(template)
            ax[i].set_title(f'Template {i + 1}')

        plt.tight_layout()
        plt.show()

    return templates


class TemplateMatching:

    def __init__(self, templates, margin_x: int = 50, margin_y: int = 50, shape: tuple | int = None,
                 scale: float = 0.1, min_confidence: float = 0.5, reduce: str = 'max'):
        self.templates = templates
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.shape = shape if isinstance(shape, tuple) else (shape, shape)
        self.scale = scale
        self.min_confidence = min_confidence
        self.reduce = reduce.lower()

    def show(self, image):
        if isinstance(image, str):
            image = cv.imread(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            image = image.copy()
        x, y, w, h = self.apply(image)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 10)
        plt.imshow(image)
        plt.title('Template Matching')
        plt.show()

    def generate_dataset(self, src_images_dir, src_masks_dir, dst_images_dir, dst_masks_dir):
        src_images_dir = Path(src_images_dir)
        src_masks_dir = Path(src_masks_dir)
        dst_images_dir = Path(dst_images_dir)
        dst_masks_dir = Path(dst_masks_dir)
        overlay_dir = dst_images_dir / '../Overlaid_TemplateMatching_Images'

        assert src_images_dir.exists()
        assert src_masks_dir.exists()

        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_masks_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([f for f in os.listdir(src_images_dir) if not f.startswith('.')])
        masks = sorted([f for f in os.listdir(src_masks_dir) if not f.startswith('.')])

        title = 'Generating template matching dataset'
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

    def apply(self, image):
        if isinstance(image, str):
            image = cv.imread(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Resize image to smaller size for faster computation
        if self.scale < 1.0:
            image = cv.resize(image, (0, 0), fx=self.scale, fy=self.scale)

        bboxes = []
        for template in self.templates:
            # Resize template to smaller size
            if self.scale < 1.0:
                template = cv.resize(template, (0, 0), fx=self.scale, fy=self.scale)

            # Safety check to make sure the template is smaller than the image
            if image.shape[0] < template.shape[0] or image.shape[1] < template.shape[1]:
                continue

            # Apply template Matching
            matched_image = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(matched_image)

            # Get the bounding box
            top_left_x, top_left_y = max_loc
            height, width = template.shape[:2]

            # Rescale back to original size
            if self.scale < 1.0:
                top_left_x = int(top_left_x / self.scale)
                top_left_y = int(top_left_y / self.scale)
                width = int(width / self.scale)
                height = int(height / self.scale)

            # Add margin to the bounding box
            top_left_x -= self.margin_x
            top_left_y -= self.margin_y
            width += 2 * self.margin_x
            height += 2 * self.margin_y

            # Align to square
            top_left_x, top_left_y, width, height = align_to_square(top_left_x, top_left_y, width, height)

            # Move bounding box inside the image
            top_left_x = max(0, top_left_x)
            top_left_y = max(0, top_left_y)
            top_left_x = min(top_left_x, image.shape[1] / self.scale - width)
            top_left_y = min(top_left_y, image.shape[0] / self.scale - height)

            # Add to list of bounding boxes
            bboxes.append((top_left_x, top_left_y, width, height, max_val))

        # Combine the bounding boxes
        bboxes = np.array(bboxes)
        # Sort by confidence
        bboxes = bboxes[np.argsort(bboxes[:, 4])[::-1]]
        best_bbox = bboxes[0]
        # Remove low confidence bounding boxes
        bboxes = bboxes[bboxes[:, 4] > self.min_confidence]
        # Return at least the best bounding box if no other bounding boxes are above the confidence threshold
        if len(bboxes) == 0:
            return best_bbox[:4].astype(int)

        if self.reduce == 'mean':
            bbox = np.mean(bboxes, axis=0)[:4].astype(int)
        elif self.reduce == 'median':
            bbox = np.median(bboxes, axis=0)[:4].astype(int)
        elif self.reduce == 'max':
            bbox = np.max(bboxes, axis=0)[:4].astype(int)
        elif self.reduce == 'min':
            bbox = np.min(bboxes, axis=0)[:4].astype(int)
        elif self.reduce == 'join':
            # Join the bounding boxes in extreme points
            x_min = np.min(bboxes[:, 0])
            y_min = np.min(bboxes[:, 1])
            x_max = np.max(bboxes[:, 0] + bboxes[:, 2])
            y_max = np.max(bboxes[:, 1] + bboxes[:, 3])
            bbox = np.array([x_min, y_min, x_max - x_min, y_max - y_min]).astype(int)
        else:
            return bboxes[:, :4].astype(int)

        # Align to square
        bbox = align_to_square(*bbox)

        # Move bounding box inside the image
        bbox = np.maximum(bbox, 0)
        bbox[0] = min(bbox[0], image.shape[1] / self.scale - bbox[2])
        bbox[1] = min(bbox[1], image.shape[0] / self.scale - bbox[3])

        return bbox


def align_to_square(x, y, w, h):
    if w > h:
        diff = w - h
        y -= diff // 2
        h += diff
    elif h > w:
        diff = h - w
        x -= diff // 2
        w += diff
    return x, y, w, h
