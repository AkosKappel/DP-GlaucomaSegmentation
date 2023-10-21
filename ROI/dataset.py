import cv2 as cv
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from tqdm.notebook import tqdm


class RoiDataset(Dataset):

    def __init__(self, files, df, input_size, in_scale, model_scale, transform=None):
        self.files = files
        self.df = df
        self.transform = transform

        self.input_size = input_size
        self.model_scale = model_scale
        self.in_scale = in_scale

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        image = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB)

        target = self.df[self.df['image_id'] == file]
        bboxes = target[['x', 'y', 'w', 'h']].values
        mask = target['mask_id'].values[0]
        labels = [0] * len(bboxes)

        if self.transform is not None:
            augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = augmented['image']
            bboxes = np.array(augmented['bboxes'])

        heatmap, regmap = self.create_heatmap_and_regmap(bboxes)
        return image, heatmap, regmap, bboxes, file, mask

    def create_heatmap_and_regmap(self, bboxes):
        # Define the dimensions of the output heatmap and regression maps
        map_size = self.input_size // self.model_scale

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
            x = int(c[0]) // self.model_scale // self.in_scale
            y = int(c[1]) // self.model_scale // self.in_scale
            sigma = np.clip(c[2] * c[3] // 2000, 2, 4)
            heatmap = draw_gaussian_on_heatmap(heatmap, [x, y], sigma=sigma)

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

        mask_disc = np.where(mask >= 1, 1, 0).astype(np.uint8)
        mask_cup = np.where(mask >= 2, 1, 0).astype(np.uint8)

        disc_contours, _ = cv.findContours(mask_disc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cup_contours, _ = cv.findContours(mask_cup, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        disc_x, disc_y, disc_w, disc_h = cv.boundingRect(disc_contours[0])
        cup_x, cup_y, cup_w, cup_h = cv.boundingRect(cup_contours[0])

        x, y, w, h = min(cup_x, disc_x), min(cup_y, disc_y), max(cup_w, disc_w), max(cup_h, disc_h)

        row = {
            'image_id': str(image_path),
            'mask_id': str(mask_path),
            'x': float(x - margin),
            'y': float(y - margin),
            'w': float(w + 2 * margin),
            'h': float(h + 2 * margin),
        }
        df = pd.concat([df, pd.DataFrame(row, index=[i])])

    if csv_file.exists():
        os.remove(csv_file)
    df.to_csv(csv_file, index=False)

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

        # Crop images and masks to bounding box
        start_x = max(0, box2_x)
        end_x = min(w2, box2_x + box2_w)
        start_y = max(0, box2_y)
        end_y = min(h2, box2_y + box2_h)

        cropped_img = img2[start_y:end_y, start_x:end_x]
        cropped_mask = mask2[start_y:end_y, start_x:end_x]

        # Resize to 512x512
        if isinstance(size, int):
            size = (size, size)

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
