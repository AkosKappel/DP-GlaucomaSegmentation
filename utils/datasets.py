import cv2 as cv
import numpy as np
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
    'EyeFundusDataset', 'load_dataset', 'softmap_to_binary_mask',
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

    def __init__(self, image_dir: str, mask_dir: str, image_names: list[str], mask_names: list[str], transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.mask_names = mask_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)

        mask_name = self.mask_names[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def load_dataset(image_dir: str, mask_dir: str, image_names: list[str] = None, mask_names: list[str] = None,
                 train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1,
                 train_transform=None, val_transform=None, test_transform=None,
                 batch_size: int = 4, pin_memory: bool = False, num_workers: int = 1,
                 return_datasets: bool = False, return_loaders: bool = True, random_state: int = 4118):
    assert train_size + val_size + test_size == 1, 'The sum of train_size, val_size, and test_size must be 1'

    # Get the names of all images in the image directory
    if image_names is None:
        image_names = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
    if mask_names is None:
        mask_names = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

    # Calculate the validation size as a percentage of the training size
    val_size /= train_size + val_size

    # Split the data into train, validation, and test sets
    indices = np.arange(len(image_names))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=random_state)

    # Get the file names in each set
    train_image_names = [image_names[i] for i in train_indices]
    val_image_names = [image_names[i] for i in val_indices]
    test_image_names = [image_names[i] for i in test_indices]

    train_mask_names = [mask_names[i] for i in train_indices]
    val_mask_names = [mask_names[i] for i in val_indices]
    test_mask_names = [mask_names[i] for i in test_indices]

    # Create the datasets
    train_ds = EyeFundusDataset(image_dir, mask_dir, train_image_names, train_mask_names, transform=train_transform)
    val_ds = EyeFundusDataset(image_dir, mask_dir, val_image_names, val_mask_names, transform=val_transform)
    test_ds = EyeFundusDataset(image_dir, mask_dir, test_image_names, test_mask_names, transform=test_transform)

    print(f'''Loading dataset:
    Train size: {len(train_ds)} ({len(train_ds) / len(image_names) * 100:.2f}%)
    Validation size: {len(val_ds)} ({len(val_ds) / len(image_names) * 100:.2f}%)
    Test size: {len(test_ds)} ({len(test_ds) / len(image_names) * 100:.2f}%)

    Image shape: {val_ds[0][0].numpy().shape}
    Mask shape: {val_ds[0][1].numpy().shape}
    Batch size: {batch_size}''')

    if return_datasets:
        return train_ds, val_ds, test_ds

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f'''
    Train loader length: {len(train_loader)}
    Validation loader length: {len(val_loader)}
    Test loader length: {len(test_loader)}''')

    if return_loaders:
        return train_loader, val_loader, test_loader
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# Binarize softmap maps using a threshold
def softmap_to_binary_mask(softmap_mask, threshold: float = 0.5):
    softmap_mask = softmap_mask.astype(np.float32)
    softmap_mask /= softmap_mask.max()
    binary_mask = (softmap_mask >= threshold).astype(np.uint8)
    return binary_mask


def prepare_origa_dataset(base_dir: str | Path, test_size: float = 0.1,
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

    img_file_names = sorted(os.listdir(images_dir))
    mask_file_names = sorted([f for f in os.listdir(gt_dir) if Path(f).suffix == '.mat'])

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
                    train_img_names, train_mask_names, 'Preparing ORIGA train dataset')

    test_images_dir = base_dir / 'TestImages'
    test_masks_dir = base_dir / 'TestMasks'
    prepare_dataset(images_dir, gt_dir, test_images_dir, test_masks_dir,
                    test_img_names, test_mask_names, 'Preparing ORIGA test dataset')


def prepare_drishti_dataset(base_dir: str | Path, test_size: float = 0.1,
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
