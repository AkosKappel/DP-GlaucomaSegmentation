import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os

__all__ = [
    'ORIGA_MEANS', 'ORIGA_STDS', 'ROI_ORIGA_MEANS', 'ROI_ORIGA_STDS', 'PADDED_ORIGA_MEANS', 'PADDED_ORIGA_STDS',
    'get_mean_and_standard_deviation_from_files', 'get_mean_and_standard_deviation_from_dataloader',
    'OrigaDataset', 'load_origa', 'load_fundus',
]

# calculated from only the training set using the functions below
ORIGA_MEANS = (0.5543, 0.3410, 0.1510)
ORIGA_STDS = (0.2541, 0.1580, 0.0823)

ROI_ORIGA_MEANS = (0.9400, 0.6225, 0.3316)
ROI_ORIGA_STDS = (0.1557, 0.1727, 0.1556)

PADDED_ORIGA_MEANS = (0.4579, 0.2808, 0.1239)
PADDED_ORIGA_STDS = (0.3188, 0.1990, 0.1055)


def get_mean_and_standard_deviation_from_files(image_paths):
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


def get_mean_and_standard_deviation_from_dataloader(loader):
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


class OrigaDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, image_names: list[str], transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_name = image_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


def load_origa(image_dir: str, mask_dir: str, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1,
               train_transform=None, val_transform=None, test_transform=None,
               batch_size: int = 4, pin_memory: bool = False, num_workers: int = 1,
               datasets_only: bool = False, loaders_only: bool = True, random_state: int = 4118):
    assert train_size + val_size + test_size == 1, 'The sum of train_size, val_size, and test_size must be 1'

    base_transform = A.Compose([
        ToTensorV2(),
    ])

    if train_transform is None:
        train_transform = base_transform
    if val_transform is None:
        val_transform = base_transform
    if test_transform is None:
        test_transform = base_transform

    # Get the names of all images in the image directory
    image_names = sorted(os.listdir(image_dir))
    # Calculate the validation size as a percentage of the training size
    val_size /= train_size + val_size

    # Split the data into train, validation, and test sets
    train_names, test_names = train_test_split(image_names, test_size=test_size, random_state=random_state)
    train_names, val_names = train_test_split(train_names, test_size=val_size, random_state=random_state)

    # Create the datasets
    train_dataset = OrigaDataset(image_dir, mask_dir, train_names, transform=train_transform)
    val_dataset = OrigaDataset(image_dir, mask_dir, val_names, transform=val_transform)
    test_dataset = OrigaDataset(image_dir, mask_dir, test_names, transform=test_transform)

    print(f'''Loading ORIGA dataset:
    Train size: {len(train_dataset)} ({len(train_dataset) / len(image_names) * 100:.2f}%)
    Validation size: {len(val_dataset)} ({len(val_dataset) / len(image_names) * 100:.2f}%)
    Test size: {len(test_dataset)} ({len(test_dataset) / len(image_names) * 100:.2f}%)
    
    Image shape: {val_dataset[0][0].numpy().shape}
    Mask shape: {val_dataset[0][1].numpy().shape}
    Batch size: {batch_size}''')

    if datasets_only:
        return train_dataset, val_dataset, test_dataset

    # Create the data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f'''
    Train loader length: {len(train_loader)}
    Validation loader length: {len(val_loader)}
    Test loader length: {len(test_loader)}''')

    if loaders_only:
        return train_loader, val_loader, test_loader
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def load_fundus():
    # TODO: implement when we have the data
    pass
