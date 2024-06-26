import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


class CenterNetDataset(Dataset):

    def __init__(self, df, input_size, model_scale, transform=None):
        self.df = df
        self.input_size = input_size
        self.model_scale = model_scale
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target = self.df.iloc[idx]
        # Each image has exactly 1 bounding box, but in case of multiple bounding boxes use:
        # targets = self.df[self.df['image_id'] == image_id]
        image_id = target['image_id']
        mask_id = target['mask_id']

        # Read image
        image = cv.imread(image_id)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Get bounding box and its label
        bboxes = target[['x', 'y', 'w', 'h']].values
        bboxes = bboxes.reshape(-1, 4)  # (n_boxes, 4)
        labels = [0] * len(bboxes)

        # Apply data augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = augmented['image']
            bboxes = np.array(augmented['bboxes'])

        # Create heatmap and regression map
        heatmap, regmap = create_heatmap_and_regmap(bboxes, self.input_size, self.model_scale, 512 // self.input_size)
        return image, heatmap, regmap, bboxes, image_id, mask_id


def create_heatmap_and_regmap(bboxes, input_size, model_scale, in_scale):
    # Define the dimensions of the output heatmap and regression maps
    map_size = input_size // model_scale

    # Initialize the heatmap and regression maps with zeros
    heatmap = np.zeros([map_size, map_size])
    regression_map = np.zeros([2, map_size, map_size])

    # If the target is empty, return the initialized maps
    if len(bboxes) == 0:
        return heatmap, regression_map

    # Extract the center and dimensions of the target
    x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    center = np.array([x + w // 2, y + h // 2, w, h]).T

    # Iterate through the centers and create Gaussian heatmaps
    for c in center:
        x = int(c[0]) // model_scale // in_scale
        y = int(c[1]) // model_scale // in_scale
        sigma = np.clip(c[2] * c[3] // 2000, 2, 4)
        heatmap = draw_gaussian_on_heatmap(heatmap, [x, y], sigma=sigma)

    # Convert targets to their centers
    regr_targets = center[:, 2:] / input_size / in_scale

    # Plot regression values to the regression map
    for r, c in zip(regr_targets, center):
        for i in range(-2, 3):
            for j in range(-2, 3):
                x = int(c[0]) // model_scale // in_scale + i
                y = int(c[1]) // model_scale // in_scale + j
                regression_map[:, x, y] = r

    # Transpose the regression maps for consistency
    regression_map[0] = regression_map[0].T
    regression_map[1] = regression_map[1].T

    return heatmap, regression_map


def draw_gaussian_on_heatmap(heatmap, center, sigma: float = 2.0):
    # Calculate the Gaussian kernel size based on the provided sigma value
    kernel_size = sigma * 6

    # Calculate the integer coordinates of the center point
    center_x = int(center[0] + 0.5)
    center_y = int(center[1] + 0.5)

    # Get the dimensions of the heatmap
    heatmap_width, heatmap_height = heatmap.shape[0], heatmap.shape[1]

    # Calculate the upper-left and bottom-right coordinates of the bounding box
    ul_x = int(center_x - kernel_size)
    ul_y = int(center_y - kernel_size)
    br_x = int(center_x + kernel_size + 1)
    br_y = int(center_y + kernel_size + 1)

    # Check if the bounding box is entirely outside the heatmap
    if ul_x >= heatmap_height or ul_y >= heatmap_width or br_x < 0 or br_y < 0:
        return heatmap

    # Calculate the size of the Gaussian kernel
    kernel_size = 2 * kernel_size + 1

    # Create a grid for the Gaussian kernel
    x = np.arange(0, kernel_size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = kernel_size // 2

    # Generate the 2D Gaussian kernel
    gaussian_kernel = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Calculate the overlap between the bounding box and the heatmap
    g_x = max(0, -ul_x), min(br_x, heatmap_height) - ul_x
    g_y = max(0, -ul_y), min(br_y, heatmap_width) - ul_y
    img_x = max(0, ul_x), min(br_x, heatmap_height)
    img_y = max(0, ul_y), min(br_y, heatmap_width)

    # Update the heatmap by taking the maximum value of the Gaussian kernel and the existing heatmap
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        gaussian_kernel[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    )

    return heatmap
