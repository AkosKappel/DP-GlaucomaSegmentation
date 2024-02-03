import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from collections import defaultdict
from pathlib import Path
from tqdm.notebook import tqdm

__all__ = [
    'preprocess_centernet_input', 'detect_roi', 'prediction_to_bbox', 'prediction_to_bboxes',
    'generate_centernet_dataset', 'generate_ground_truth_bbox_csv', 'generate_roi_dataset',
    'pool_duplicates', 'merge_overlapping_boxes', 'calculate_iou', 'calculate_coverage', 'calculate_metrics',
    'pad_to_square', 'resize_box_to_square',
]


def preprocess_centernet_input(image: str | np.ndarray, mask=None, otsu_crop: bool = False,
                               crop_margin: int = 0, pad_margin: int = 0):
    if isinstance(image, str):
        image = cv.imread(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    assert image is not None, 'Image not found'

    if isinstance(mask, str):
        mask = cv.imread(mask, cv.IMREAD_GRAYSCALE)
        assert mask is not None, 'Mask not found'

    if mask is not None:
        assert image.shape[:2] == mask.shape[:2], 'Image and mask shapes do not match'
    else:
        # Create empty dummy mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Crop black edges from image using OTSU thresholding:
    if otsu_crop:
        # Convert image to grayscale (or any single channel)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # Get binary image using OTSU thresholding
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Find bounding box of non-zero pixels
        row_sums = np.sum(thresh, axis=1)
        col_sums = np.sum(thresh, axis=0)
        x1 = np.argmax(col_sums > 0)
        x2 = len(col_sums) - np.argmax(col_sums[::-1] > 0)
        y1 = np.argmax(row_sums > 0)
        y2 = len(row_sums) - np.argmax(row_sums[::-1] > 0)

        # Add margin to bounding box
        if crop_margin > 0:
            x1 = max(0, x1 - crop_margin)
            y1 = max(0, y1 - crop_margin)
            x2 = min(image.shape[1], x2 + crop_margin)
            y2 = min(image.shape[0], y2 + crop_margin)

        # Crop image using bounding box
        image = image[y1:y2, x1:x2]
        assert np.sum(mask) == np.sum(mask[y1:y2, x1:x2]), 'Cropped mask misses some foreground object pixels'
        mask = mask[y1:y2, x1:x2]

    # Pad image and mask to square shapes
    image, mask = pad_to_square(image, mask, value=0)

    # Add margin to squares
    if pad_margin > 0:
        image = cv.copyMakeBorder(image, pad_margin, pad_margin, pad_margin, pad_margin, cv.BORDER_CONSTANT, value=0)
        mask = cv.copyMakeBorder(mask, pad_margin, pad_margin, pad_margin, pad_margin, cv.BORDER_CONSTANT, value=0)

    if np.any(mask):
        return image, mask
    return image


def generate_centernet_dataset(src_images_dir: str, src_masks_dir: str, dst_images_dir: str, dst_masks_dir: str,
                               otsu_crop: bool = True, crop_margin: int = 0, pad_margin: int = 0):
    src_images_dir = Path(src_images_dir)
    src_masks_dir = Path(src_masks_dir)
    dst_images_dir = Path(dst_images_dir)
    dst_masks_dir = Path(dst_masks_dir)

    assert src_images_dir.exists()
    assert src_masks_dir.exists()

    dst_images_dir.mkdir(exist_ok=True, parents=True)
    dst_masks_dir.mkdir(exist_ok=True, parents=True)

    images = sorted([f for f in os.listdir(src_images_dir) if not f.startswith('.')])
    masks = sorted([f for f in os.listdir(src_masks_dir) if not f.startswith('.')])

    title = f'Generating CenterNet dataset'
    for image_name, mask_name in tqdm(zip(images, masks), total=len(images), desc=title):
        image_path = src_images_dir / image_name
        mask_path = src_masks_dir / mask_name

        image = cv.imread(str(image_path))
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        image, mask = preprocess_centernet_input(image, mask, otsu_crop, crop_margin, pad_margin)

        cv.imwrite(str(dst_images_dir / image_name), image)
        cv.imwrite(str(dst_masks_dir / mask_name), mask)


def generate_ground_truth_bbox_csv(images_dir: str | list[str], masks_dir: str | list[str],
                                   csv_file: str, margin: int = 0):
    if isinstance(images_dir, str):
        images_dir = [images_dir]
    if isinstance(masks_dir, str):
        masks_dir = [masks_dir]

    images_dir = [Path(d) for d in images_dir]
    masks_dir = [Path(d) for d in masks_dir]
    csv_file = Path(csv_file)

    for d in images_dir + masks_dir:
        assert d.exists(), f'Directory {d} not found'

    image_paths = []
    for d in images_dir:
        image_paths.extend(sorted([f'{d}/{f}' for f in os.listdir(d) if not f.startswith('.')]))
    mask_paths = []
    for d in masks_dir:
        mask_paths.extend(sorted([f'{d}/{f}' for f in os.listdir(d) if not f.startswith('.')]))

    df = pd.DataFrame()
    title = f'Generating csv file with bounding boxes'
    for i, (image_path, mask_path) in enumerate(tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc=title)):
        # image = cv.imread(str(image_path))
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        # Create binary masks
        mask_disc = np.where(mask >= 1, 1, 0).astype(np.uint8)
        mask_cup = np.where(mask >= 2, 1, 0).astype(np.uint8)

        # Find contours
        disc_contours, _ = cv.findContours(mask_disc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cup_contours, _ = cv.findContours(mask_cup, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Get OD and OC bounding boxes
        disc_x, disc_y, disc_w, disc_h = cv.boundingRect(disc_contours[0])
        cup_x, cup_y, cup_w, cup_h = cv.boundingRect(cup_contours[0])

        # Get the bounding box that covers both the optic disc and the optic cup
        x, y, w, h = min(cup_x, disc_x), min(cup_y, disc_y), max(cup_w, disc_w), max(cup_h, disc_h)

        # Add margin to the bounding box (watch out to not exceed the image boundaries)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(mask.shape[1], w + 2 * margin)
        h = min(mask.shape[0], h + 2 * margin)
        x = min(mask.shape[1] - w, x)
        y = min(mask.shape[0] - h, y)

        # Add new entry to the dataframe
        row = {
            'image_id': str(image_path),
            'mask_id': str(mask_path),
            'x': float(x),
            'y': float(y),
            'w': float(w),
            'h': float(h),
        }
        df = pd.concat([df, pd.DataFrame(row, index=[i])])

    # Save results to CSV file (overwrite if already exists)
    if csv_file.exists():
        os.remove(csv_file)
    df.to_csv(csv_file, index=False)

    return df


@torch.no_grad()
def generate_roi_dataset(model, images: list[str], masks: list[str], dst_images_dir: str, dst_masks_dir: str, transform,
                         input_size: int, device=None, small_margin: int = 16, large_margin: int = 0,
                         threshold: float = 0.6, roi_size: int = 512, interpolation: int = cv.INTER_AREA):
    dst_images_dir = Path(dst_images_dir)
    dst_masks_dir = Path(dst_masks_dir)
    overlay_dir = dst_images_dir / '../Overlaid_CenterNet_Images'

    dst_images_dir.mkdir(exist_ok=True, parents=True)
    dst_masks_dir.mkdir(exist_ok=True, parents=True)
    overlay_dir.mkdir(exist_ok=True, parents=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_coverage, total_area, total_bg_area, total_disc_area, total_cup_area = 0, 0, 0, 0, 0
    model = model.eval().to(device)
    pbar = tqdm(images, desc='Generating RoI dataset')
    for i, image_file in enumerate(pbar, start=1):
        mask_file = masks[i - 1] if masks else None

        # Read image
        image = cv.imread(image_file)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Read mask if given
        if mask_file:
            mask = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Detect Region of Interest (RoI)
        x, y, w, h = detect_roi(
            model, image, mask, transform, input_size, device,
            small_margin, large_margin, threshold, roi_size, interpolation, return_bbox=True,
        )

        roi_image = image[y:y + h, x:x + w]
        roi_mask = mask[y:y + h, x:x + w]

        # Resize RoI images
        roi_image = cv.resize(roi_image, (roi_size, roi_size), interpolation=interpolation)
        roi_mask = cv.resize(roi_mask, (roi_size, roi_size), interpolation=interpolation)

        # Save image
        roi_image = cv.cvtColor(roi_image, cv.COLOR_RGB2BGR)
        cv.imwrite(str(dst_images_dir / Path(image_file).name), roi_image)

        if mask_file:
            # Save ground truth mask
            cv.imwrite(str(dst_masks_dir / Path(mask_file).name), roi_mask)

            # If the ground truth is available, check if the predicted bounding box covers the entire optic disc
            coverage = np.sum(mask[y:y + h, x:x + w]) / np.sum(mask)
            total_coverage += coverage

            # Calculate how much of the RoI area is covered by the optic disc and the optic cup
            area = np.prod(roi_mask.shape[:2])
            bg_area = np.sum(roi_mask == 0)
            disc_area = np.sum(roi_mask == 1)
            cup_area = np.sum(roi_mask == 2)
            total_area += area
            total_bg_area += bg_area
            total_disc_area += disc_area
            total_cup_area += cup_area

            pbar.set_postfix({
                'coverage': f'{total_coverage * 100 / i:.2f}%',
                'bg': f'{total_bg_area / total_area * 100:.2f}%',
                'disc': f'{total_disc_area / total_area * 100:.2f}%',
                'cup': f'{total_cup_area / total_area * 100:.2f}%',
            })

            if coverage < 1.0:
                print(f'Bounding box of {image_file} covers {coverage * 100:.2f}% of the optic disc')
                # Show the predicted bounding box on the whole image
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                image[mask > 0] = 255
                image[mask > 1] = 127
                image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
                plt.figure(figsize=(8, 8))
                plt.title(f'Coverage: {coverage * 100:.2f}%')
                plt.imshow(image)
                plt.show()

            # Visualize OD and OC overlay on the cropped and resized image
            roi_mask = np.repeat(roi_mask[:, :, np.newaxis], 3, axis=2)
            roi_image[roi_mask > 0] = 255
            roi_image[roi_mask > 1] = 127
            cv.imwrite(str(overlay_dir / Path(image_file).name), roi_image)


@torch.no_grad()
def detect_roi(model, image_file: str | np.ndarray, mask_file: str | np.ndarray | None, transform, input_size: int,
               device=None, small_margin: int = 16, large_margin: int = 0, threshold: float = 0.6,
               roi_size: int = 512, interpolation: int = cv.INTER_AREA, return_bbox: bool = False):
    # Read image
    if isinstance(image_file, str):
        original_image = cv.imread(image_file)
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    else:
        original_image = image_file

    # Read mask if given
    if mask_file is not None:
        if isinstance(mask_file, str):
            original_mask = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
        else:
            original_mask = mask_file
    else:
        # Dummy mask to avoid if statements later
        original_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get original size for rescaling the bounding box back to the larger image
    original_h, original_w = original_image.shape[:2]

    # Apply transformations
    image = transform(image=original_image, bboxes=[], labels=[])['image']

    # Get transformed sizes
    transformed_h, transformed_w = image.shape[1:]

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    # Make prediction
    model.eval().to(device)
    heatmap, regression = model(image)
    heatmap = heatmap.sigmoid()

    # Move to CPU and convert to numpy
    heatmap = heatmap.detach().cpu().squeeze().numpy()
    regression = regression.detach().cpu().squeeze().numpy()
    # image = image.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    # image = (image - image.min()) / (image.max() - image.min())
    # image = (image * 255).astype(np.uint8)

    # Get bounding boxes from heatmap and regression
    x, y, w, h = prediction_to_bbox(heatmap, regression, input_size, model.scale, threshold)

    # Add margin to merged bounding box of the smaller image
    x = int(max(0, x - small_margin))
    y = int(max(0, y - small_margin))
    w = int(min(input_size, w + 2 * small_margin))
    h = int(min(input_size, h + 2 * small_margin))

    # Resize to a square shaped box
    x, y, w, h = resize_box_to_square([x, y, w, h])

    # Move box coordinates to ensure they are within the image bounds without changing the box size
    x = np.clip(x, 0, input_size - w)
    y = np.clip(y, 0, input_size - h)

    # Compute the ratio between the original and the resized image sizes
    ratio_x = original_w / transformed_w
    ratio_y = original_h / transformed_h

    # Recompute bounding box for larger image
    scaled_x = int(np.round(x * ratio_x))
    scaled_y = int(np.round(y * ratio_y))
    scaled_w = int(np.round(w * ratio_x))
    scaled_h = int(np.round(h * ratio_y))

    # Add margin to the enlarged bounding box
    scaled_x = int(max(0, scaled_x - large_margin))
    scaled_y = int(max(0, scaled_y - large_margin))
    scaled_w = int(min(original_w, scaled_w + 2 * large_margin))
    scaled_h = int(min(original_h, scaled_h + 2 * large_margin))

    # Align bounding box to a square shape
    x, y, w, h = resize_box_to_square([scaled_x, scaled_y, scaled_w, scaled_h])

    # Clip box coordinates to ensure they are within the image bounds
    x = np.clip(x, 0, original_w - w)
    y = np.clip(y, 0, original_h - h)

    if return_bbox:
        return x, y, w, h

    # Get final bounding box coordinates and crop the images
    roi_image = original_image[y:y + h, x:x + w]
    roi_mask = original_mask[y:y + h, x:x + w]

    # Resize RoI images
    roi_image = cv.resize(roi_image, (roi_size, roi_size), interpolation=interpolation)
    roi_mask = cv.resize(roi_mask, (roi_size, roi_size), interpolation=interpolation)

    return roi_image, roi_mask


def prediction_to_bbox(heatmap, regression, input_size, model_scale, threshold):
    # Get bounding boxes from heatmap and regression
    heatmap = pool_duplicates(heatmap)
    bboxes, scores = prediction_to_bboxes(heatmap, regression, input_size, model_scale, threshold)

    # Take the bounding box with the highest score if none of the boxes has a score above the threshold
    if len(bboxes) == 0:
        bboxes, scores = prediction_to_bboxes(heatmap, regression, input_size, model_scale, heatmap.max() - 0.01)

    # Join overlapping bounding boxes into a single box
    merged_bbox = merge_overlapping_boxes(bboxes, scores, iou_threshold=0.5)

    return merged_bbox


def prediction_to_bboxes(heatmap, regression, input_size, model_scale, threshold: float = 0.9):
    # Get predicted center locations
    prediction_mask = heatmap > threshold
    predicted_centers = np.where(heatmap > threshold)

    # Get regression values for the predicted centers
    predicted_regressions = regression[:, prediction_mask].T

    # Create bounding boxes and adjust for the original image size
    bboxes = []
    scores = heatmap[prediction_mask]
    for i, reg in enumerate(predicted_regressions):
        bbox = np.array([
            predicted_centers[1][i] * model_scale - reg[0] * input_size // 2,
            predicted_centers[0][i] * model_scale - reg[1] * input_size // 2,
            int(reg[0] * input_size),
            int(reg[1] * input_size),
        ])
        # Clip box coordinates to ensure they are within the image bounds
        bbox = np.clip(bbox, 0, input_size)
        bboxes.append(bbox)

    return np.asarray(bboxes), scores


def pool_duplicates(data, stride: int = 3):
    # Iterate over the data array with the specified stride
    for y in np.arange(1, data.shape[1] - 1, stride):
        for x in np.arange(1, data.shape[0] - 1, stride):
            # Extract a 3x3 subarray
            subarray = data[x - 1:x + 2, y - 1:y + 2]

            # Find the indices of the maximum value in the subarray
            max_indices = np.asarray(np.unravel_index(np.argmax(subarray), subarray.shape))

            # Iterate over the elements in the 3x3 subarray
            for c1 in range(3):
                for c2 in range(3):
                    if not (c1 == max_indices[0] and c2 == max_indices[1]):
                        # Set non-maximum values to -1
                        data[x + c1 - 1, y + c2 - 1] = -1

    return data


def merge_overlapping_boxes(bboxes, scores, iou_threshold):
    assert len(bboxes) == len(scores), 'Number of bounding boxes and scores must be the same.'
    assert len(bboxes) > 0, 'No bounding boxes to merge.'

    # Sort the boxes in descending order based on their scores
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sorted_bboxes = [bboxes[i] for i in indices]
    sorted_scores = [scores[i] for i in indices]

    # Remove boxes with low IoU with the highest scoring box
    filtered_bboxes = [
        box
        for box, score in zip(sorted_bboxes, sorted_scores)
        if calculate_iou(box, bboxes[0]) >= iou_threshold
    ]

    if len(filtered_bboxes) == 0:
        return sorted_bboxes[0]

    # Compute a single merged bounding box that covers all the boxes
    x = min(box[0] for box in filtered_bboxes)
    y = min(box[1] for box in filtered_bboxes)
    w = max(box[0] + box[2] for box in filtered_bboxes) - x
    h = max(box[1] + box[3] for box in filtered_bboxes) - y

    return [x, y, w, h]


def calculate_iou(box1, box2):
    # Box format: [x, y, w, h]

    # Calculate the coordinates of the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    # If there is no intersection, return 0
    if x1 >= x2 or y1 >= y2:
        return 0.0

    # Calculate the areas of the two bounding boxes
    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]

    # Calculate intersection over union
    intersection = (x2 - x1) * (y2 - y1)
    union = area_box1 + area_box2 - intersection
    return intersection / union


def calculate_coverage(inner_box, outer_box):
    inner_x1, inner_y1, inner_w, inner_h = inner_box
    inner_x2, inner_y2 = inner_x1 + inner_w, inner_y1 + inner_h

    outer_x1, outer_y1, outer_w, outer_h = outer_box
    outer_x2, outer_y2 = outer_x1 + outer_w, outer_y1 + outer_h

    # Calculate the area of the inner box
    inner_area = inner_w * inner_h

    # Inner box is completely outside of outer box
    if inner_x1 >= outer_x2 or inner_x2 <= outer_x1 or inner_y1 >= outer_y2 or inner_y2 <= outer_y1:
        return 0.0

    # Calculate the intersection area
    intersection_x1 = max(inner_x1, outer_x1)
    intersection_x2 = min(inner_x2, outer_x2)
    intersection_y1 = max(inner_y1, outer_y1)
    intersection_y2 = min(inner_y2, outer_y2)
    intersection_area = max(0, (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1))

    # Calculate the coverage percentage
    coverage = intersection_area / inner_area
    return coverage


def calculate_metrics(pred_heatmaps, pred_regressions, true_bboxes, input_size, model_scale,
                      threshold: float = 0.9, tolerance_margin: int = 0):
    # Move to CPU and convert to numpy
    pred_heatmaps = pred_heatmaps.detach().cpu().numpy()
    pred_regressions = pred_regressions.detach().cpu().numpy()

    results = defaultdict(list)
    for i in range(pred_heatmaps.shape[0]):
        # Get bounding boxes from heatmap and regression
        pred_heatmap = pred_heatmaps[i].squeeze()
        pred_regression = pred_regressions[i].squeeze()
        true_bbox = true_bboxes[i].squeeze().numpy()

        pred_bbox = prediction_to_bbox(pred_heatmap, pred_regression, input_size, model_scale, threshold)
        pred_bbox = [
            max(0, pred_bbox[0] - tolerance_margin),
            max(0, pred_bbox[1] - tolerance_margin),
            min(input_size, pred_bbox[2] + tolerance_margin),
            min(input_size, pred_bbox[3] + tolerance_margin),
        ]

        # Calculate IoU
        iou = calculate_iou(true_bbox, pred_bbox)
        results['iou'].append(iou)

        # Calculate coverage
        coverage = calculate_coverage(true_bbox, pred_bbox)
        results['coverage'].append(coverage)

    return results


def pad_to_square(img, mask=None, value: int = 0):
    h, w, _ = img.shape
    if h > w:
        pad = (h - w) // 2
        img = cv.copyMakeBorder(img, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=value)
        if mask is not None:
            mask = cv.copyMakeBorder(mask, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=value)
    elif w > h:
        pad = (w - h) // 2
        img = cv.copyMakeBorder(img, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=value)
        if mask is not None:
            mask = cv.copyMakeBorder(mask, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=value)
    if mask is None:
        return img
    return img, mask


def resize_box_to_square(bbox, keep_larger_side: bool = True):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2

    side = max(w, h) if keep_larger_side else min(w, h)
    x = center_x - side / 2
    y = center_y - side / 2

    return [int(x), int(y), int(side), int(side)]
