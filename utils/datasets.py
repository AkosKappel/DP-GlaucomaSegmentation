import cv2 as cv
import numpy as np
import torch
import os
import scipy.io
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

__all__ = [
    'ORIGA_MEANS', 'ORIGA_STDS', 'ROI_ORIGA_MEANS', 'ROI_ORIGA_STDS',
    'DRISHTI_MEANS', 'DRISHTI_STDS', 'ROI_DRISHTI_MEANS', 'ROI_DRISHTI_STDS',
    'EyeFundusDataset', 'load_dataset', 'prepare_origa_dataset', 'prepare_drishti_dataset',
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


def prepare_origa_dataset(base_dir):
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    images_dir = base_dir / 'Images'
    gt_dir = base_dir / 'Semi-automatic-annotations-done-by-doctors-eg-ground-truth'
    masks_dir = base_dir / 'Masks'

    img_names = sorted(os.listdir(images_dir))
    mask_names = sorted([f for f in os.listdir(gt_dir) if Path(f).suffix == '.mat'])

    masks_dir.mkdir(parents=True, exist_ok=True)

    for img_name, mask_name in zip(tqdm(img_names, desc='Preparing ORIGA dataset'), mask_names):
        img = cv.imread(str(images_dir / img_name))

        assert img is not None

        mat = scipy.io.loadmat(str(gt_dir / mask_name))
        mask = mat['mask']

        if img.shape[:2] != mask.shape:
            mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_AREA)

        cv.imwrite(str(masks_dir / (mask_name[:-4] + '.png')), mask)


def prepare_drishti_dataset(base_dir):
    def prepare_subset(images_dir, gt_dir, masks_dir, desc=None):
        img_names = sorted(os.listdir(images_dir))
        mask_names = sorted([f for f in os.listdir(gt_dir) if (gt_dir / f).is_dir()])

        masks_dir.mkdir(parents=True, exist_ok=True)

        for img_name, mask_name in zip(tqdm(img_names, desc=desc), mask_names):
            img = cv.imread(str(images_dir / img_name))

            assert img is not None

            height, width = img.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)

            disc_file = gt_dir / mask_name / f'AvgBoundary/{mask_name}_ODAvgBoundary.txt'
            cup_file = gt_dir / mask_name / f'AvgBoundary/{mask_name}_CupAvgBoundary.txt'

            assert disc_file.exists()
            assert cup_file.exists()

            disc = np.loadtxt(disc_file, delimiter=' ')
            # Slice array to remove duplicate entries
            for i, val in enumerate(disc[1:]):
                if np.all(val == disc[0]):
                    disc = disc[:i + 1]
                    break
            mask = cv.fillPoly(mask, [disc.astype(np.int32)], 1)

            cup = np.loadtxt(cup_file, delimiter=' ')
            for i, val in enumerate(cup[1:]):
                if np.all(val == cup[0]):
                    cup = cup[:i + 1]
                    break
            mask = cv.fillPoly(mask, [cup.astype(np.int32)], 2)

            cv.imwrite(str(masks_dir / (mask_name + '.png')), mask)

    if isinstance(base_dir, str):
        base_dir = Path(base_dir) / 'Drishti-GS1_files/Drishti-GS1_files'

    train_dir = base_dir / 'Training'
    test_dir = base_dir / 'Test'

    prepare_subset(train_dir / 'Images', train_dir / 'Gt', train_dir / 'Masks', 'Preparing Drishti-GS1 training set')
    prepare_subset(test_dir / 'Images', test_dir / 'Test_Gt', test_dir / 'Masks', 'Preparing Drishti-GS1 test set')


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
