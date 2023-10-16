import cv2 as cv
import numpy as np
import os
import pandas as pd
from pathlib import Path


def generate_bbox_csv(image_dir: str = '', mask_dir: str = '', csv_file: str = '', margin: int = 0):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    csv_file = Path(csv_file)

    image_files = sorted([image_dir / f for f in os.listdir(image_dir) if not f.startswith('.')])
    mask_files = sorted([mask_dir / f for f in os.listdir(mask_dir) if not f.startswith('.')])

    df = pd.DataFrame()
    for i, (image_path, mask_path) in enumerate(zip(image_files, mask_files)):
        # image = cv.imread(str(image_path))
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        mask_cup = np.where(mask >= 2, 1, 0).astype(np.uint8)
        mask_disc = np.where(mask >= 1, 1, 0).astype(np.uint8)

        disc_contours, _ = cv.findContours(mask_disc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cup_contours, _ = cv.findContours(mask_cup, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        disc_x, disc_y, disc_w, disc_h = cv.boundingRect(disc_contours[0])
        cup_x, cup_y, cup_w, cup_h = cv.boundingRect(cup_contours[0])

        x, y, w, h = min(cup_x, disc_x), min(cup_y, disc_y), max(cup_w, disc_w), max(cup_h, disc_h)

        row = {
            'image_id': image_path,
            'mask_id': mask_path,
            'x': float(x - margin),
            'y': float(y - margin),
            'w': float(w + 2 * margin),
            'h': float(h + 2 * margin),
            'disc_center_x': float(disc_x + disc_w // 2),
            'disc_center_y': float(disc_y + disc_h // 2),
            'cup_center_x': float(cup_x + cup_w // 2),
            'cup_center_y': float(cup_y + cup_h // 2),
            'disc_x': float(disc_x),
            'disc_y': float(disc_y),
            'disc_w': float(disc_w),
            'disc_h': float(disc_h),
            'cup_x': float(cup_x),
            'cup_y': float(cup_y),
            'cup_w': float(cup_w),
            'cup_h': float(cup_h),
        }
        df = pd.concat([df, pd.DataFrame(row, index=[i])])

    if csv_file.exists():
        os.remove(csv_file)
    df.to_csv(csv_file, index=False)


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


def merge_overlapping_boxes(bboxes, scores, iou_threshold):
    assert len(bboxes) == len(scores), 'Number of bounding boxes and scores must be the same.'

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

    assert len(filtered_bboxes) > 0, 'No bounding boxes left for merging.'

    # Compute a single merged bounding box that covers all the boxes
    x = min(box[0] for box in filtered_bboxes)
    y = min(box[1] for box in filtered_bboxes)
    w = max(box[0] + box[2] for box in filtered_bboxes) - x
    h = max(box[1] + box[3] for box in filtered_bboxes) - y

    return [x, y, w, h]


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


def prediction_to_bboxes(heatmap, regression, input_size, model_scale, thresh: float = 0.9):
    # Get predicted center locations
    prediction_mask = heatmap > thresh
    predicted_centers = np.where(heatmap > thresh)

    # Get regression values for the predicted centers
    predicted_regressions = regression[:, prediction_mask].T

    # Create bounding boxes and adjust for the original image size
    bboxes = []
    scores = heatmap[prediction_mask]
    for i, reg in enumerate(predicted_regressions):
        bbox = np.array([
            predicted_centers[1][i] * model_scale - reg[0] * input_size // 2,
            predicted_centers[0][i] * model_scale - reg[1] * input_size // 2,
            int(reg[0] * input_size), int(reg[1] * input_size),
        ])
        # Clip box coordinates to ensure they are within the image bounds
        bbox = np.clip(bbox, 0, input_size)
        bboxes.append(bbox)

    return np.asarray(bboxes), scores
