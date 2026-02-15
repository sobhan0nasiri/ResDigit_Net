import torch
import torchvision
from torchvision.datasets import MNIST, EMNIST as TorchEMNIST
from data_transform import train_transform
from torch.utils.data import random_split, ConcatDataset

class EMNIST(TorchEMNIST):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        img = torchvision.transforms.functional.rotate(img, -90)
        img = torchvision.transforms.functional.hflip(img)

        if self.transform:
            img = self.transform(img)
        return img, target

def create_datasets(root='./data'):

    print("Step 1: Loading Datasets (This might take a moment if downloading)...")

    mnist_full = MNIST(root=root, train=True, download=True, transform=train_transform)
    emnist_full = EMNIST(root=root, split='digits', train=True, download=True, transform=train_transform)

    full_dataset = ConcatDataset([mnist_full, emnist_full])

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Done! Total Images: {len(full_dataset)}")
    print(f"Training Samples: {len(train_set)} | Validation Samples: {len(val_set)}")
    
    return train_set, val_set


if __name__ == "__main__":
    
    create_datasets()