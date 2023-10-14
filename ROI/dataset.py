import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class Normalize:

    def __init__(self):
        self.mean = [0.9400, 0.6225, 0.3316]
        self.std = [0.1557, 0.1727, 0.1556]
        self.norm = transforms.Normalize(self.mean, self.std)

    def __call__(self, image):
        image = image.astype(np.float32) / 255
        image -= self.mean
        image /= self.std
        return image


class RoiDataset(Dataset):
    def __init__(self, files, labels, input_size, in_scale, model_scale, transform=None):
        self.files = files
        self.labels = labels

        self.input_size = input_size
        self.model_scale = model_scale
        self.in_scale = in_scale

        if transform:
            self.transform = transform
        self.normalize = Normalize()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = cv.imread(self.files[idx])
        image = cv.resize(image, (self.input_size, self.input_size))
        image = self.normalize(image)
        image = image.transpose([2, 0, 1])

        target = self.labels[self.labels['image_id'] == self.files[idx]]
        heatmap, regmap = self.create_heatmap_and_regmap(target)
        return image, heatmap, regmap

    def create_heatmap_and_regmap(self, target):
        # Define the dimensions of the output heatmap and regression maps
        map_size = self.input_size // self.model_scale

        # Initialize the heatmap and regression maps with zeros
        heatmap = np.zeros([map_size, map_size])
        regression_map = np.zeros([2, map_size, map_size])

        # If the target is empty, return the initialized maps
        if len(target) == 0:
            return heatmap, regression_map

        # Extract the center and dimensions of the target
        center = np.array([
            target['x'] + target['w'] // 2,
            target['y'] + target['h'] // 2,
            target['w'], target['h'],
        ]).T

        # Iterate through the centers and create Gaussian heatmaps
        for c in center:
            x = int(c[0]) // self.model_scale // self.in_scale
            y = int(c[1]) // self.model_scale // self.in_scale
            sigma = np.clip(c[2] * c[3] // 2000, 2, 4)
            heatmap = RoiDataset.draw_gaussian_on_heatmap(heatmap, [x, y], sigma=sigma)

        # Convert targets to their centers
        regr_targets = center[:, 2:] / self.input_size / self.in_scale

        # Plot regression values to the regression map
        for r, c in zip(regr_targets, center):
            for i in range(-2, 3):
                for j in range(-2, 3):
                    x = int(c[0]) // self.model_scale // self.in_scale + i
                    y = int(c[1]) // self.model_scale // self.in_scale + j
                    regression_map[:, x, y] = r

        # Transpose the regression maps for consistency
        regression_map[0] = regression_map[0].T
        regression_map[1] = regression_map[1].T

        return heatmap, regression_map

    @staticmethod
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
