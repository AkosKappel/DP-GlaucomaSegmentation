import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
from pathlib import Path
from tqdm.notebook import tqdm

__all__ = [
    'pad_to_square', 'resize_to_square', 'calculate_coverage', 'calculate_iou',
    'merge_overlapping_boxes', 'pool_duplicates',
    'prediction_to_bboxes', 'generate_padded_dataset', 'generate_resized_dataset',
    'generate_bbox_csv', 'detect_objects', 'generate_cropped_dataset',
]


def pad_to_square(img, mask, value: int = 0):
    h, w, _ = img.shape
    if h > w:
        pad = (h - w) // 2
        img = cv.copyMakeBorder(img, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=value)
        mask = cv.copyMakeBorder(mask, 0, 0, pad, pad, cv.BORDER_CONSTANT, value=value)
    elif w > h:
        pad = (w - h) // 2
        img = cv.copyMakeBorder(img, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=value)
        mask = cv.copyMakeBorder(mask, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=value)
    return img, mask


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


def generate_padded_dataset(src_images_dir: str, src_masks_dir: str, dst_images_dir: str, dst_masks_dir: str):
    src_images_dir = Path(src_images_dir)
    src_masks_dir = Path(src_masks_dir)
    dst_images_dir = Path(dst_images_dir)
    dst_masks_dir = Path(dst_masks_dir)

    assert src_images_dir.exists()
    assert src_masks_dir.exists()

    dst_images_dir.mkdir(exist_ok=True, parents=True)
    dst_masks_dir.mkdir(exist_ok=True, parents=True)

    imgs = sorted([f for f in os.listdir(src_images_dir) if not f.startswith('.')])
    masks = sorted([f for f in os.listdir(src_masks_dir) if not f.startswith('.')])

    title = f'Generating padded dataset'
    for img_name, mask_name in tqdm(zip(imgs, masks), total=len(imgs), desc=title):
        img_path = src_images_dir / img_name
        mask_path = src_masks_dir / mask_name

        img = cv.imread(str(img_path))
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        img, mask = pad_to_square(img, mask)

        cv.imwrite(str(dst_images_dir / img_name), img)
        cv.imwrite(str(dst_masks_dir / mask_name), mask)


def generate_resized_dataset(src_images_dir: str, src_masks_dir: str, dst_images_dir: str, dst_masks_dir: str,
                             size: tuple | int = 512, interpolation: int = cv.INTER_AREA):
    src_images_dir = Path(src_images_dir)
    src_masks_dir = Path(src_masks_dir)
    dst_images_dir = Path(dst_images_dir)
    dst_masks_dir = Path(dst_masks_dir)

    assert src_images_dir.exists()
    assert src_masks_dir.exists()

    dst_images_dir.mkdir(exist_ok=True, parents=True)
    dst_masks_dir.mkdir(exist_ok=True, parents=True)

    if isinstance(size, int):
        size = (size, size)

    imgs = sorted([f for f in os.listdir(src_images_dir) if not f.startswith('.')])
    masks = sorted([f for f in os.listdir(src_masks_dir) if not f.startswith('.')])

    title = f'Generating resized dataset with size {size}'
    for img_name, mask_name in tqdm(zip(imgs, masks), total=len(imgs), desc=title):
        img_path = src_images_dir / img_name
        mask_path = src_masks_dir / mask_name

        img = cv.imread(str(img_path))
        mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

        img = cv.resize(img, size, interpolation=interpolation)
        mask = cv.resize(mask, size, interpolation=interpolation)

        cv.imwrite(str(dst_images_dir / img_name), img)
        cv.imwrite(str(dst_masks_dir / mask_name), mask)


def generate_bbox_csv(images_dir: str, masks_dir: str, csv_file: str, margin: int = 0):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    csv_file = Path(csv_file)

    assert images_dir.exists()
    assert masks_dir.exists()

    imgs = sorted([f for f in os.listdir(images_dir) if not f.startswith('.')])
    masks = sorted([f for f in os.listdir(masks_dir) if not f.startswith('.')])

    df = pd.DataFrame()
    title = f'Generating csv file with bounding box coordinates'
    for i, (img_name, mask_name) in enumerate(tqdm(zip(imgs, masks), total=len(imgs), desc=title)):
        image_path = images_dir / img_name
        mask_path = masks_dir / mask_name

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


def detect_objects(model, loader, device, input_size, model_scale,
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

            # Move box coordinates to ensure they are within the image bounds
            x = max(0, x)
            y = max(0, y)
            x = min(input_size - w, x)
            y = min(input_size - h, y)

            # Add new entry to the dataframe
            row = {
                'image_id': img_file,
                'mask_id': mask_file,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
            }
            df = pd.concat([df, pd.DataFrame([row])])

            # Check if entire optic disc is covered by the predicted bounding box
            mask = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
            if np.sum(mask[y:y + h, x:x + w]) != np.sum(mask):
                print(f'Bounding box of {img_file} does not cover the entire optic disc')

                img = img.transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                img = (img * 255).astype(np.uint8)

                # Draw predicted bounding box
                x, y, w, h = int(x), int(y), int(w), int(h)
                img = img.copy()
                img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

                # Draw ground truth
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                img[mask > 0] = 255
                img[mask > 1] = 127

                _ = plt.subplots(1, 1, figsize=(8, 8))
                plt.imshow(img)
                plt.title(img_file)
                plt.show()

    # Save results
    if out_file:
        if os.path.exists(out_file):
            os.remove(out_file)
        df.to_csv(out_file, index=False)

    print(f'Found {len(df)} bounding boxes.')

    return df


def generate_cropped_dataset(df, src_images_dir: str, src_masks_dir: str, dst_images_dir: str, dst_masks_dir: str,
                             size: tuple | int = 512, interpolation: int = cv.INTER_AREA, margin: int = 0):
    src_images_dir = Path(src_images_dir)
    src_masks_dir = Path(src_masks_dir)
    dst_masks_dir = Path(dst_masks_dir)
    dst_images_dir = Path(dst_images_dir)
    overlay_dir = dst_images_dir / '../Overlaid_Images'

    assert src_images_dir.exists()
    assert src_masks_dir.exists()

    dst_images_dir.mkdir(exist_ok=True, parents=True)
    dst_masks_dir.mkdir(exist_ok=True, parents=True)
    overlay_dir.mkdir(exist_ok=True, parents=True)

    # Resize to 512x512
    if isinstance(size, int):
        size = (size, size)

    title = f'Generating cropped optic disc dataset'
    for i, row in tqdm(df.iterrows(), total=len(df), desc=title):
        img1_file, mask1_file, box1_x, box1_y, box1_w, box1_h = row[['image_id', 'mask_id', 'x', 'y', 'w', 'h']]

        img1_file = Path(img1_file)
        mask1_file = Path(mask1_file)

        img2_file = src_images_dir / img1_file.name
        mask2_file = src_masks_dir / mask1_file.name

        if not img2_file.exists():
            print(f'Image {img2_file} does not exist')
            continue
        if not mask2_file.exists():
            print(f'Mask {mask2_file} does not exist')
            continue

        img1 = cv.imread(str(img1_file))
        img2 = cv.imread(str(img2_file))
        mask2 = cv.imread(str(mask2_file), cv.IMREAD_GRAYSCALE)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Compute the ratio between the original and the resized image
        res_x = w2 / w1
        res_y = h2 / h1

        # Recompute bounding box for larger image
        box2_x = int(np.round(box1_x * res_x))
        box2_y = int(np.round(box1_y * res_y))
        box2_w = int(np.round(box1_w * res_x))
        box2_h = int(np.round(box1_h * res_y))

        # Add margin to new bounding box
        box2_x = max(0, box2_x - margin)
        box2_y = max(0, box2_y - margin)
        box2_w = min(w2, box2_w + 2 * margin)
        box2_h = min(h2, box2_h + 2 * margin)

        # Align bounding box to a square shape
        box2_x, box2_y, box2_w, box2_h = resize_to_square([box2_x, box2_y, box2_w, box2_h])

        # Move box coordinates to ensure they are within the image bounds
        box2_x = max(0, box2_x)
        box2_y = max(0, box2_y)
        box2_x = min(w2 - box2_w, box2_x)
        box2_y = min(h2 - box2_h, box2_y)

        # Get final bounding box coordinates
        start_x = box2_x
        start_y = box2_y
        end_x = box2_x + box2_w
        end_y = box2_y + box2_h

        # Crop images
        cropped_img = img2[start_y:end_y, start_x:end_x]
        cropped_mask = mask2[start_y:end_y, start_x:end_x]

        # Resize images
        resized_img = cv.resize(cropped_img, size, interpolation=interpolation)
        resized_mask = cv.resize(cropped_mask, size, interpolation=interpolation)

        # Save images
        cv.imwrite(str(dst_images_dir / img1_file.name), resized_img)
        cv.imwrite(str(dst_masks_dir / mask1_file.name), resized_mask)

        # Visualize OD and OC overlay on the cropped and resized image
        resized_mask = np.repeat(resized_mask[:, :, np.newaxis], 3, axis=2)
        resized_img[resized_mask > 0] = 255
        resized_img[resized_mask > 1] = 127
        cv.imwrite(str(overlay_dir / img1_file.name), resized_img)
