import matplotlib.pyplot as plt
import numpy as np
import torch

from data import create_datasets
from data_loader import get_loaders

def inspect_data_quality(loader):

    print("Fetching a batch of images for inspection...")
    try:

        images_d, labels_d = next(iter(loader))
        
        plt.figure(figsize=(14, 7))
        plt.suptitle("Final Quality Check: Correct Orientation & Realistic Noise", fontsize=16)
        
        for i in range(min(12, len(images_d))):
            ax = plt.subplot(3, 4, i + 1)

            img = images_d[i].cpu() * 0.3081 + 0.1307
            img = img.numpy().squeeze()

            img = np.clip(img, 0, 1)
            
            plt.imshow(img, cmap='gray')
            plt.title(f"Label: {labels_d[i].item()}")
            plt.axis('off')
        
        plt.tight_layout()
        print("Displaying plot... Close the window to continue.")
        plt.show(block=True)
        
    except Exception as e:
        print(f"Error during inspection: {e}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    
    train_set, val_set = create_datasets()
    t_loader, _, _ = get_loaders(train_set, val_set, batch_size=12, num_workers=0)
    
    inspect_data_quality(t_loader)