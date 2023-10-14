import cv2 as cv
import numpy as np


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


def convert_predictions_to_boxes(heatmap, regression, input_size, model_scale, thresh: float = 0.9):
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


def show_box(image, heatmap, regression, input_size, model_scale, thresh: float = 0.9, color: tuple = (0, 220, 0)):
    boxes, _ = convert_predictions_to_boxes(heatmap, regression, input_size, model_scale, thresh)
    sample = image
    print('boxes:', boxes.shape)

    for box in boxes:
        x, y, w, h = box
        cv.rectangle(sample, (int(x), int(y + h)), (int(x + w), int(y)), color, 3)

    return sample
