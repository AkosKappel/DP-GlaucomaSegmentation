import cv2 as cv
import numpy as np
import pandas as pd
import torch
import os
import scipy.io
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ROI import preprocess_centernet_input

__all__ = [
    'ORIGA_MEANS', 'ORIGA_STDS', 'ROI_ORIGA_MEANS', 'ROI_ORIGA_STDS',
    'DRISHTI_MEANS', 'DRISHTI_STDS', 'ROI_DRISHTI_MEANS', 'ROI_DRISHTI_STDS',
    'EyeFundusDataset', 'load_dataset', 'load_files_from_dir', 'softmap_to_binary_mask',
    'prepare_origa_dataset', 'prepare_drishti_dataset', 'prepare_rimone_dataset',
    'get_mean_and_standard_deviation_from_files', 'get_mean_and_standard_deviation_from_dataloader',
]

# Calculated only from the training set to avoid data leakage
ORIGA_MEANS = (0.5543, 0.3410, 0.1510)  # RGB order
ORIGA_STDS = (0.2541, 0.1580, 0.0823)

ROI_ORIGA_MEANS = (0.9400, 0.6225, 0.3316)
ROI_ORIGA_STDS = (0.1557, 0.1727, 0.1556)

DRISHTI_MEANS = (0.3443, 0.1621, 0.0505)
DRISHTI_STDS = (0.1863, 0.0940, 0.0284)

ROI_DRISHTI_MEANS = ()
ROI_DRISHTI_STDS = ()


class EyeFundusDataset(Dataset):
    def __init__(self, image_paths: list[str], mask_paths: list[str] = None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)

        mask = None
        if self.mask_paths is not None:
            mask_path = self.mask_paths[idx]
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def load_files_from_dir(directory: str | list[str] | None):
    if directory is None:
        return []
    if isinstance(directory, str):
        directory = [directory]
    files = []
    for d in directory:
        if not os.path.exists(d):
            continue
        elif os.path.isfile(d):
            files.append(d)
        elif os.path.isdir(d):
            new_files = [f'{d}/{f}' for f in os.listdir(d) if not f.startswith('.')]
            files.extend(sorted(new_files))
    return files


# Images can be a single directory, a list of directories or a list of files
def load_dataset(images: str | list[str], masks: str | list[str], transform=None, batch_size: int = 4,
                 pin_memory: bool = False, num_workers: int = 1, shuffle: bool = False, return_loader: bool = True):
    # Get the paths to all the images and masks in the provided directories
    image_paths = load_files_from_dir(images)
    mask_paths = load_files_from_dir(masks)

    # Create the dataset and data loader
    dataset = EyeFundusDataset(image_paths, mask_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    print(f'Loaded dataset with {len(dataset)} samples in {len(loader)} batches.')
    return loader if return_loader else dataset


# Binarize softmap maps using a threshold
def softmap_to_binary_mask(softmap_mask, threshold: float = 0.5):
    softmap_mask = softmap_mask.astype(np.float32)
    softmap_mask /= softmap_mask.max()
    binary_mask = (softmap_mask >= threshold).astype(np.uint8)
    return binary_mask


def prepare_origa_dataset(base_dir: str | Path, test_size: float = None,
                          random_state: int = 411, debug: bool = False):
    def prepare_dataset(src_images_dir, src_masks_dir,
                        dst_images_dir, dst_masks_dir,
                        img_names, mask_names, desc=None):
        # Delete existing directories
        if dst_images_dir.exists():
            for f in dst_images_dir.iterdir():
                f.unlink()
            dst_images_dir.rmdir()
        if dst_masks_dir.exists():
            for f in dst_masks_dir.iterdir():
                f.unlink()
            dst_masks_dir.rmdir()

        if len(img_names) == 0:
            return

        # Create new empty directories
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_masks_dir.mkdir(parents=True, exist_ok=True)

        for img_name, mask_name in zip(tqdm(img_names, desc=desc), mask_names):
            img = cv.imread(str(src_images_dir / img_name))
            assert img is not None
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            mat = scipy.io.loadmat(str(src_masks_dir / mask_name))
            mask = mat['mask']

            if img.shape[:2] != mask.shape:
                print(f'Resizing mask {mask_name} with shape {mask.shape} to image shape {img.shape}')
                mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)

            img, mask = preprocess_centernet_input(img, mask, otsu_crop=True)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            if debug:
                mask[mask == 1] = 255
                mask[mask == 2] = 128
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                img[mask == 255] = 255
                img[mask == 128] = 128

            cv.imwrite(str(dst_images_dir / img_name), img)
            cv.imwrite(str(dst_masks_dir / (mask_name[:-4] + '.png')), mask)

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    images_dir = base_dir / 'Images'
    gt_dir = base_dir / 'Semi-automatic-annotations-done-by-doctors-eg-ground-truth'
    labels = pd.read_excel(base_dir / 'labels.xlsx')

    img_file_names = sorted(os.listdir(images_dir))
    mask_file_names = sorted([f for f in os.listdir(gt_dir) if Path(f).suffix == '.mat'])

    if test_size is None:  # Default split to ORIGA-A and ORIGA-B subsets
        def get_name(x):
            return x.strip().replace("'", '').replace('"', '').split('.')[0]

        set_a = set(map(get_name, labels[labels['set'] == 'A']['filename'].values))
        set_b = set(map(get_name, labels[labels['set'] == 'B']['filename'].values))

        train_img_names = [f for f in img_file_names if get_name(f) in set_a]
        train_mask_names = [f for f in mask_file_names if get_name(f) in set_a]

        test_img_names = [f for f in img_file_names if get_name(f) in set_b]
        test_mask_names = [f for f in mask_file_names if get_name(f) in set_b]
    elif test_size == 0:
        train_img_names, test_img_names, train_mask_names, test_mask_names = img_file_names, [], mask_file_names, []
    elif test_size == 1:
        train_img_names, test_img_names, train_mask_names, test_mask_names = [], img_file_names, [], mask_file_names
    else:
        train_img_names, test_img_names, train_mask_names, test_mask_names = train_test_split(
            img_file_names, mask_file_names, test_size=test_size, random_state=random_state,
        )

    train_images_dir = base_dir / 'TrainImages'
    train_masks_dir = base_dir / 'TrainMasks'
    prepare_dataset(images_dir, gt_dir, train_images_dir, train_masks_dir,
                    train_img_names, train_mask_names, 'Preparing ORIGA train dataset')

    test_images_dir = base_dir / 'TestImages'
    test_masks_dir = base_dir / 'TestMasks'
    prepare_dataset(images_dir, gt_dir, test_images_dir, test_masks_dir,
                    test_img_names, test_mask_names, 'Preparing ORIGA test dataset')


def prepare_drishti_dataset(base_dir: str | Path, test_size: float = None,
                            random_state: int = 411, debug: bool = False):
    def prepare_dataset(src_images_dir, src_masks_dir,
                        dst_images_dir, dst_masks_dir,
                        img_names, mask_names, desc=None):
        # Delete existing directories
        if dst_images_dir.exists():
            for f in dst_images_dir.iterdir():
                f.unlink()
            dst_images_dir.rmdir()
        if dst_masks_dir.exists():
            for f in dst_masks_dir.iterdir():
                f.unlink()
            dst_masks_dir.rmdir()

        if len(img_names) == 0:
            return

        # Create new empty directories
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_masks_dir.mkdir(parents=True, exist_ok=True)

        for img_name, mask_name in zip(tqdm(img_names, desc=desc), mask_names):
            img = cv.imread(str(src_images_dir / img_name))
            assert img is not None
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            mask = cv.imread(str(src_masks_dir / mask_name), cv.IMREAD_GRAYSCALE)
            assert mask is not None

            if img.shape[:2] != mask.shape:
                mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)

            mask = 2 - mask

            img, mask = preprocess_centernet_input(img, mask)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            if debug:
                mask[mask == 1] = 255
                mask[mask == 2] = 128
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                img[mask == 255] = 255
                img[mask == 128] = 128

            cv.imwrite(str(dst_images_dir / img_name), img)
            cv.imwrite(str(dst_masks_dir / mask_name), mask)

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    images_dir = base_dir / 'images'
    gt_dir = base_dir / 'annotations-ONH-ground truth'

    img_file_names = sorted(os.listdir(images_dir))
    mask_file_names = sorted([f for f in os.listdir(gt_dir)])

    if test_size is None:  # Default split to Drishti-GS1 and Drishti-GS2 subsets
        set_a = set(os.listdir(base_dir / 'Drishti-GS1_files/Drishti-GS1_files/Training/Images'))
        set_b = set(os.listdir(base_dir / 'Drishti-GS1_files/Drishti-GS1_files/Test/Images'))

        train_img_names = [f for f in img_file_names if f in set_a]
        train_mask_names = [f for f in mask_file_names if f in set_a]

        test_img_names = [f for f in img_file_names if f in set_b]
        test_mask_names = [f for f in mask_file_names if f in set_b]
    elif test_size == 0:
        train_img_names, test_img_names, train_mask_names, test_mask_names = img_file_names, [], mask_file_names, []
    elif test_size == 1:
        train_img_names, test_img_names, train_mask_names, test_mask_names = [], img_file_names, [], mask_file_names
    else:
        train_img_names, test_img_names, train_mask_names, test_mask_names = train_test_split(
            img_file_names, mask_file_names, test_size=test_size, random_state=random_state,
        )

    train_images_dir = base_dir / 'TrainImages'
    train_masks_dir = base_dir / 'TrainMasks'
    prepare_dataset(images_dir, gt_dir, train_images_dir, train_masks_dir,
                    train_img_names, train_mask_names, 'Preparing Drishti train dataset')

    test_images_dir = base_dir / 'TestImages'
    test_masks_dir = base_dir / 'TestMasks'
    prepare_dataset(images_dir, gt_dir, test_images_dir, test_masks_dir,
                    test_img_names, test_mask_names, 'Preparing Drishti test dataset')


def prepare_rimone_dataset(base_dir: str | Path, test_size: float = 0.1, side: str = 'left',
                           random_state: int = 411, debug: bool = False):
    def prepare_dataset(src_images_dir, src_masks_dir,
                        dst_images_dir, dst_masks_dir,
                        img_names, mask_names, desc=None):
        # Delete existing directories
        if dst_images_dir.exists():
            for f in dst_images_dir.iterdir():
                f.unlink()
            dst_images_dir.rmdir()
        if dst_masks_dir.exists():
            for f in dst_masks_dir.iterdir():
                f.unlink()
            dst_masks_dir.rmdir()

        if len(img_names) == 0:
            return

        # Create new empty directories
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_masks_dir.mkdir(parents=True, exist_ok=True)

        for img_name, mask_name in zip(tqdm(img_names, desc=desc), mask_names):
            img = cv.imread(str(src_images_dir / img_name))
            assert img is not None
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            mask = cv.imread(str(src_masks_dir / mask_name), cv.IMREAD_GRAYSCALE)
            assert mask is not None

            # Keep only the left or right side of the image
            if side == 'left':
                img = img[:, :img.shape[1] // 2]
            elif side == 'right':
                img = img[:, img.shape[1] // 2:]

            if img.shape[:2] != mask.shape:
                mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)

            # Mirror images of left eye
            if img_name.endswith('-L.jpg'):
                mask = cv.flip(mask, 1)

            mask[mask == 255] = 1
            mask[mask == 128] = 2

            img, mask = preprocess_centernet_input(img, mask)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            if debug:
                mask[mask == 1] = 255
                mask[mask == 2] = 128
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                img[mask == 255] = 255
                img[mask == 128] = 128

            cv.imwrite(str(dst_images_dir / img_name), img)
            cv.imwrite(str(dst_masks_dir / mask_name), mask)

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    images_dir = base_dir / 'images'
    gt_dir = base_dir / 'SegmentationsAverage-GroundTruth'

    img_file_names = sorted(os.listdir(images_dir))
    mask_file_names = sorted([f for f in os.listdir(gt_dir)])

    if test_size == 0:
        train_img_names, test_img_names, train_mask_names, test_mask_names = img_file_names, [], mask_file_names, []
    elif test_size == 1:
        train_img_names, test_img_names, train_mask_names, test_mask_names = [], img_file_names, [], mask_file_names
    else:
        train_img_names, test_img_names, train_mask_names, test_mask_names = train_test_split(
            img_file_names, mask_file_names, test_size=test_size, random_state=random_state,
        )

    train_images_dir = base_dir / 'TrainImages'
    train_masks_dir = base_dir / 'TrainMasks'
    prepare_dataset(images_dir, gt_dir, train_images_dir, train_masks_dir,
                    train_img_names, train_mask_names, 'Preparing RIMONE train dataset')

    test_images_dir = base_dir / 'TestImages'
    test_masks_dir = base_dir / 'TestMasks'
    prepare_dataset(images_dir, gt_dir, test_images_dir, test_masks_dir,
                    test_img_names, test_mask_names, 'Preparing RIMONE test dataset')


def get_mean_and_standard_deviation_from_files(image_paths: list[str | Path]):
    """
    Calculate the mean and standard deviation of a dataset. The values are calculated per channel
    across all images. The images should be just from the training set, not the entire dataset to
    avoid data leakage.
    """
    # Initialize variables to store running sums for mean and standard deviation
    mean = np.zeros(3)
    std = np.zeros(3)

    # Iterate through the images and update mean and std
    for image_path in image_paths:
        # Load the image using OpenCV and convert to RGB
        image = cv.imread(str(image_path))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Normalize the pixel values to the range [0, 1]
        if image.max() > 1:
            image = image / 255.0

        # Compute the mean and standard deviation for each channel
        mean += np.mean(image, axis=(0, 1))
        std += np.std(image, axis=(0, 1))

    num_images = len(image_paths)
    mean /= num_images
    std /= num_images

    return mean, std


def get_mean_and_standard_deviation_from_dataloader(loader: DataLoader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for images, *_ in loader:
        if images.max() > 1:
            images = images / 255.0
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
