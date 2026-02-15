import torch
import os
from tqdm.auto import tqdm

def save_checkpoint(state, is_best, val_acc, model_name, Checkpoint_filename="checkpoint.pth"):

    save_dir = "../Model_Handler_PTH/Model_PTH"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_name = f"best_{model_name}_complete_acc_{val_acc:.2f}.pth"
    full_path = os.path.join(save_dir, file_name)
        
    torch.save(state, Checkpoint_filename)
    
    if is_best:
        torch.save(state, full_path)
        print(f"Model saved to: {full_path}")
        print(f"⭐ Best model with accuracy {val_acc:.2f}% checkpoint saved!")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):

    print(f"=> Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    history = checkpoint['history']
    
    print(f"=> Loaded checkpoint at epoch {start_epoch}")
    return start_epoch, best_acc, history

def train_model(model_name, model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path=None, device='cpu'):
    
    best_acc = 0.0
    start_epoch = 0
    
    history = {'train_loss': [], 'val_acc': []}
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, best_acc, history = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.6f}"})

        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.2f}%")

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        checkpoint_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
            'history': history
        }
        save_checkpoint(checkpoint_state, is_best, val_acc, model_name)

    return history