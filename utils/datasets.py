import cv2 as cv
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os


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


def load_origa(image_dir: str, mask_dir: str, train_transform=None, val_transform=None, test_transform=None,
               train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 4118):
    assert train_size + val_size + test_size == 1, 'The sum of train_size, val_size, and test_size must be 1'

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

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # TODO: add test cases
    pass
