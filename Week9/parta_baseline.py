import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from dataset_utils import get_dataset_splits, MapDataset, print_dataset_diagnostics

# 3. Construct baseline CNN
class DefectCNN(nn.Module):
    def __init__(self):
        super(DefectCNN, self).__init__()
        # Input shape: (Batch, 1, 200, 200)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # Output: (Batch, 16, 200, 200)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                 # Output: (Batch, 16, 100, 100)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) # Output: (Batch, 32, 100, 100)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                 # Output: (Batch, 32, 50, 50)
        
        self.flatten = nn.Flatten()                                                        # Output: (Batch, 32 * 50 * 50 = 80000)
        self.fc = nn.Linear(32 * 50 * 50, 6)                                               # Output: (Batch, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x

def calculate_manual_f1(preds, targets, num_classes=6):
    preds = torch.tensor(preds) if not isinstance(preds, torch.Tensor) else preds
    targets = torch.tensor(targets) if not isinstance(targets, torch.Tensor) else targets
    f1_scores = []
    print(f"{'Class ID':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 47)
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        print(f"{c:<10} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")
    return f1_scores

def run_baseline_pipeline():
    os.makedirs("d:/Fusemachines_Fellowship/Week9", exist_ok=True)
    
    # 1. Dataset splits and diagnostics
    train_sub, val_sub, test_sub, class_names = get_dataset_splits(seed=42)
    print_dataset_diagnostics(train_sub, val_sub, test_sub, class_names)
    
    # 2. Define transforms
    # Grayscale image conversion, Tensor conversion, and Normalization
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Wrap in MapDataset to apply transforms
    train_dataset = MapDataset(train_sub, transform=transform_pipeline)
    val_dataset = MapDataset(val_sub, transform=transform_pipeline)
    test_dataset = MapDataset(test_sub, transform=transform_pipeline)
    
    # Verify dimensions from first sample
    img, label = train_dataset[0]
    print(f"Verified single image dimensions: {img.shape} (C x H x W)")
    print(f"Verified label: {label} (Class: {class_names[label]})\n")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize baseline CNN
    model = DefectCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training and validation loop tracking over 15 epochs
    epochs = 15
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("--- Starting Baseline CNN Training (15 Epochs) ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            # Sequence: zero_grad() -> forward() -> loss calculation -> backward() -> optimizer.step()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        
        # Validation evaluation
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
    print("\n--- Training Completed ---\n")
    
    # 5. Plotting Training vs Validation Loss and Accuracy
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='red', marker='x')
    plt.title('Baseline CNN: Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, label='Train Acc', color='blue', marker='o')
    plt.plot(range(1, epochs + 1), val_accs, label='Val Acc', color='red', marker='x')
    plt.title('Baseline CNN: Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = "d:/Fusemachines_Fellowship/Week9/parta_baseline.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved baseline loss/accuracy curves to: {plot_path}\n")
    
    # 6. Evaluation script calculating the per-class F1-score on the test set
    print("--- 6. Test Set Evaluation (Per-class F1-Score) ---")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(labels.numpy())
            
    f1_scores = calculate_manual_f1(all_preds, all_targets, num_classes=6)
    
    # Match F1-scores with class names
    print("\nDetailed Per-class F1 Scores:")
    for idx, f1 in enumerate(f1_scores):
        print(f"Class '{class_names[idx]}' (ID {idx}): F1-Score = {f1:.4f}")
        
if __name__ == "__main__":
    run_baseline_pipeline()
