import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import copy
from dataset_utils import get_dataset_splits, MapDataset

# Baseline CNN architecture adjusted for 180x180 inputs
class DefectCNNAug(nn.Module):
    def __init__(self):
        super(DefectCNNAug, self).__init__()
        # Input shape: (Batch, 1, 180, 180)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # (Batch, 16, 180, 180)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)                         # (Batch, 16, 90, 90)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # (Batch, 32, 90, 90)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)                         # (Batch, 32, 45, 45)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 45 * 45, 6)                    # (Batch, 6)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 2. Hardened CNN variant
class HardenedDefectCNN(nn.Module):
    def __init__(self, input_size=180):
        super(HardenedDefectCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        
        # Calculate dynamic linear input dimension
        fc_input_dim = 32 * (input_size // 4) * (input_size // 4)
        
        # 3. Dropout right before the final linear layer
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(fc_input_dim, 6)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train_and_eval(model, train_loader, val_loader, epochs=15, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_accs = []
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        val_accs.append(acc)
        print(f"Epoch {epoch+1:02d}/{epochs} | Val Acc: {acc:.4f}")
    return val_accs

def run_hardening_experiments():
    os.makedirs("d:/Fusemachines_Fellowship/Week9_NeuralNetwork", exist_ok=True)
    train_sub, val_sub, _, _ = get_dataset_splits(seed=42)
    
    # 1. Define transforms
    # Augmentations on training set, ONLY resize & normalize on val set
    transform_augmented = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(15),
        transforms.RandomCrop(180, padding=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets and loaders
    train_dataset = MapDataset(train_sub, transform=transform_augmented)
    val_dataset = MapDataset(val_sub, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Validation accuracies from Part A baseline (200x200, no augmentation)
    val_accs_a = [0.7056, 0.8944, 0.8944, 0.9111, 0.8833, 0.9444, 0.8333, 0.8944, 0.9278, 0.9000, 0.9167, 0.9056, 0.9111, 0.9111, 0.9222]
    
    # Configuration 1: Part B with Augmentations Only (using DefectCNNAug)
    print("\n--- Training: Configuration 2 (Augmentations Only) ---")
    model_aug = DefectCNNAug()
    val_accs_b_aug = train_and_eval(model_aug, train_loader, val_loader, epochs=15)
    
    # Configuration 2: Part B with Augmentations + BatchNorm + Dropout (using HardenedDefectCNN)
    print("\n--- Training: Configuration 3 (Augmentations + BN + Dropout) ---")
    model_hardened = HardenedDefectCNN(input_size=180)
    val_accs_b_hardened = train_and_eval(model_hardened, train_loader, val_loader, epochs=15)
    
    # 4. Plot validation accuracy curves of all three configurations
    epochs_range = range(1, 16)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, val_accs_a, label='Config 1: Part A Baseline (No Aug, No BN/Dropout)', color='red', marker='o')
    plt.plot(epochs_range, val_accs_b_aug, label='Config 2: Part B Augmentations Only (No BN/Dropout)', color='blue', marker='s')
    plt.plot(epochs_range, val_accs_b_hardened, label='Config 3: Part B Augmentations + BN + Dropout (Hardened)', color='green', marker='^')
    
    plt.title('Validation Accuracy Comparison Across Model Configurations')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = "d:/Fusemachines_Fellowship/Week9_NeuralNetwork/partb_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved comparison plot to: {plot_path}")
    print(f"File exists after save: {os.path.exists(plot_path)}")
    print(f"Final Validation Accuracies:")
    print(f" - Baseline (Part A): {val_accs_a[-1]:.4f}")
    print(f" - Augmentations Only: {val_accs_b_aug[-1]:.4f}")
    print(f" - Hardened Model: {val_accs_b_hardened[-1]:.4f}")

if __name__ == "__main__":
    run_hardening_experiments()
