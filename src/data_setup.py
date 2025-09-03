import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int=32,
                       num_workers: int=0):
    """Creates training and testing Dataloaders.
    Takes in a training dir and testing dir path and turns 
    them into PyTorch Datasets and Dataloaders.
    
    Args:
        train_dir (str): Path to training directory
        test_dir (str): Path to testing directory
        transform (transforms.Compose): torchvision transforms to perform on training and testing data
        batch_size (int): Number of samples per batch in each of the Dataloader
        num_workers (int): Number of workers per Dataloader

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names