import torch
import torch.nn as nn
import torch.optim as optim

from data import create_datasets
from data_loader import get_loaders
from Model_Architecture import ModernCNN
from train import train_model

from inspect_data_quality import inspect_data_quality

def main():
    
    BATCH_SIZE = 64
    NUM_WORKERS = None  
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    MAX_LR = 1e-2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on device: {DEVICE}")

    train_set, val_set = create_datasets()

    train_loader, val_loader, workers_used = get_loaders(
        train_set, 
        val_set, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    inspect_data_quality(train_loader)
    
    print(f"DataLoaders created with {workers_used} workers.")

    model = ModernCNN(num_classes=10).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=MAX_LR, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    print("Starting training...")
    history = train_model(
        model_name="ModernCNN",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        device=DEVICE,
        checkpoint_path="checkpoint.pth"
    )

    print("Training finished successfully!")

if __name__ == "__main__":

    torch.multiprocessing.freeze_support()
    main()