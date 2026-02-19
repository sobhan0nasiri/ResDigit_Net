import torch
import torch.nn as nn
import torch.optim as optim

from data import create_datasets
from data_loader import get_loaders
from train import train_model

from Create_Model import Model_Architecture

from inspect_data_quality import inspect_data_quality

def main():
    
    selected_model_name = ""
    model_class = None
    flag = True

    while flag:
        user_input = input("Enter Model Name to Train (or 'exit' to quit): ").strip()

        if user_input.lower() == 'exit':
            flag = False
            continue

        try:
            if (model_func := getattr(Model_Architecture, user_input, None)) and callable(model_func):
                model_class = model_func
                selected_model_name = user_input
                print(f"Successfully selected: {user_input}")
                
            else:
                print(f"Model '{user_input}' is not callable or doesn't exist.")

        except AttributeError:
            print(f"Not found Model: '{user_input}'!")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("Exited successfully and let's go to train Selected Model")
    
    BATCH_SIZE = 64
    NUM_WORKERS = 12
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

    model = model_class(num_classes=10).to(DEVICE)
    
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
        model_name=selected_model_name,
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