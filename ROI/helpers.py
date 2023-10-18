import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from tqdm.notebook import tqdm


def resize_to_square(box, keep_larger_side: bool = True):
    x, y, w, h = box
    center_x = x + w / 2
    center_y = y + h / 2

    side = max(w, h) if keep_larger_side else min(w, h)
    x = center_x - side / 2
    y = center_y - side / 2

    return [int(x), int(y), int(side), int(side)]


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


def detect_optic_discs(model, loader, device, input_size, model_scale,
                       margin: int = 16, threshold: float = 0.6, out_file: str = ''):
    df = pd.DataFrame()

    title = 'Detecting optic discs'
    for imgs, _, _, gt_boxes, img_files, mask_files in tqdm(loader, total=len(loader), desc=title):
        # Move to default device
        imgs = imgs.to(device)

        # Make predictions
        with torch.no_grad():
            heatmap, regression = model(imgs)

        # Move to CPU
        imgs = imgs.detach().cpu().numpy()
        gt_boxes = gt_boxes.detach().cpu().numpy()
        regression = regression.detach().cpu().numpy()

        # Iterate over batch
        for img, hm, reg, gt_box, img_file, mask_file in zip(
                imgs, heatmap, regression, gt_boxes, img_files, mask_files):

            # Get bounding boxes from heatmap and regression
            hm = hm.squeeze()
            hm = torch.sigmoid(hm).cpu().numpy()
            hm = pool_duplicates(hm)
            bboxes, scores = prediction_to_bboxes(hm, reg, input_size, model_scale, threshold)

            if len(bboxes) == 0:
                bboxes, scores = prediction_to_bboxes(hm, reg, input_size, model_scale, hm.max() - 0.01)

            merged_box = merge_overlapping_boxes(bboxes, scores, 0.5)
            x, y, w, h = merged_box

            # Add margin to merged box
            x = int(max(0, x - margin))
            y = int(max(0, y - margin))
            w = int(min(input_size, w + 2 * margin))
            h = int(min(input_size, h + 2 * margin))

            # Resize to a square shaped box
            x, y, w, h = resize_to_square([x, y, w, h])

            row = {
                'image_id': img_file,
                'mask_id': mask_file,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
            }
            df = pd.concat([df, pd.DataFrame([row])])

            # TODO: remove this later
            # Visualize & check if the box covers the entire optic disc
            coverage = calculate_coverage(gt_box[0], [x, y, w, h])
            if coverage < 1.0:
                print(f'Bounding box of {img_file} does not cover the entire optic disc')

                img = img.transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                img = (img * 255).astype(np.uint8)

                x, y, w, h = gt_box[0]
                x, y, w, h = int(x), int(y), int(w), int(h)
                img = img.copy()
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                img = cv.putText(img, f'{coverage:.2f}', (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                _ = plt.subplots(1, 1, figsize=(8, 8))
                plt.imshow(img)
                plt.title(img_file)
                plt.show()

    # Save results
    if out_file:
        if os.path.exists(out_file):
            os.remove(out_file)
        df.to_csv(out_file, index=False)

    return df
