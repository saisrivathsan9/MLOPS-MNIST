from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_dataloaders(batch_size, num_workers, root):
    """
    Returns train and test DataLoader objects for MNIST dataset.
    """
    train_data = datasets.MNIST(
        root=root, train=True, transform=ToTensor(), download=True
    )
    test_data = datasets.MNIST(
        root=root, train=False, transform=ToTensor(), download=True
    )

    loaders = {
        "train": DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "test": DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
    return loaders