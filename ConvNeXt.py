# train_convnext.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from PIL import Image
import time
from model import convnext_tiny, convnext_small, convnext_base


def pil_loader(path):

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def is_valid_file(path):

    valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    return path.lower().endswith(valid_extensions)

# Custom Label Smoothing Cross Entropy Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = F.nll_loss(log_probs, target, reduction='none')
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def main():

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = DatasetFolder(
        root='dataset_folder/train', 
        transform=train_tfm,
        loader=pil_loader,
        is_valid_file=is_valid_file
    )
    
    val_dataset = DatasetFolder(
        root='dataset_folder/val', 
        transform=val_tfm,
        loader=pil_loader,
        is_valid_file=is_valid_file
    )

    print("train_dataset: %d" % len(train_dataset))
    print("val_dataset: %d" % len(val_dataset))
    
    # 显示类别信息
    print("Classes:", train_dataset.classes)
    print("Class to index mapping:", train_dataset.class_to_idx)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # Model setup
    num_classes = len(train_dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = convnext_tiny(num_classes=num_classes).to(device)
    

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Training parameters
    learning_rate = 0.001
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training function
    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=300):
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            epoch_start_time = time.time()
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
            
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            epoch_time = time.time() - epoch_start_time
            

            scheduler.step()
            
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            val_total_samples = 0
            
            val_start_time = time.time()
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item() * inputs.size(0)
                    val_running_corrects += torch.sum(preds == labels.data)
                    val_total_samples += inputs.size(0)
            
            val_epoch_loss = val_running_loss / val_total_samples
            val_epoch_acc = val_running_corrects.double() / val_total_samples
            val_time = time.time() - val_start_time
            
            print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} Time: {val_time:.2f}s')
            
            # Save best model
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc

                model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, 'best_convnext_model.pth')
                print(f'New best model saved with accuracy: {best_acc:.4f}')
        
        print(f'Best val Acc: {best_acc:.4f}')
        return best_acc

    # Evaluation function
    def evaluate_model(model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        return epoch_acc

    # Start training
    print("Starting ConvNeXt training...")
    try:
        best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
        print(f"Training completed! Best validation accuracy: {best_acc:.4f}")
    except Exception as e:
        print(f"Training error: {e}")
        return

    # Load best model and evaluate
    try:
        checkpoint = torch.load('best_convnext_model.pth')
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        final_acc = evaluate_model(model, val_loader, criterion)
        print(f"Final Model Accuracy: {final_acc:.4f}")
    except Exception as e:
        print(f"Error loading or evaluating model: {e}")


def inference(image_path, model_path='best_convnext_model.pth'):


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    

    image = pil_loader(image_path)
    image = transform(image).unsqueeze(0)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convnext_tiny(num_classes=10).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        

        with torch.no_grad():
            image = image.to(device)
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.max(probabilities)
            
        print(f"Predicted class index: {predicted.item()}")
        print(f"Confidence: {confidence.item():.4f}")
        return predicted.item(), confidence.item()
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

if __name__ == "__main__":
    main()
    
    
