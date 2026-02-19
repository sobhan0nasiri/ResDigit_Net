import os
import torch
from torch.utils.data import DataLoader

def get_loaders(train_set, val_set, batch_size=64, num_workers=None, pin_memory=None):    
    
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    
    cpus = os.cpu_count()
    print(f"cpu count is:  {cpus}")
    
    if num_workers is None:
        num_workers = min(8, cpus) if cpus else 0
        
    if pin_memory is None:
        pin_memory = True if torch.cuda.is_available() else False

    print(f"DEBUG: Creating loaders with num_workers={num_workers}, pin_memory={pin_memory}")

    train_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    val_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }

    if num_workers > 0:
        train_kwargs['prefetch_factor'] = 2
        val_kwargs['prefetch_factor'] = 2
        train_kwargs['persistent_workers'] = True
        val_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_set, **train_kwargs)
    val_loader = DataLoader(val_set, **val_kwargs)

    return train_loader, val_loader, num_workers