import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import optuna
import os
from dataset_utils import get_dataset_splits, MapDataset
from partb_hardening import HardenedDefectCNN

# Suppress Optuna logging to stdout to keep outputs clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_tuning():
    os.makedirs("d:/Fusemachines_Fellowship/Week9", exist_ok=True)
    train_sub, val_sub, test_sub, class_names = get_dataset_splits(seed=42)
    
    # Define transforms
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
    
    train_dataset = MapDataset(train_sub, transform=transform_augmented)
    val_dataset = MapDataset(val_sub, transform=transform_val)
    test_dataset = MapDataset(test_sub, transform=transform_val)
    
    # 1. Grid Search
    print("--- 1. Manual Grid Search ---")
    lrs = [0.001, 0.01]
    batch_sizes = [16, 32]
    
    grid_results = []
    best_grid_acc = 0.0
    best_grid_config = {}
    
    # Train for 10 epochs per combination to keep it efficient on CPU
    epochs_grid = 10
    
    for lr in lrs:
        for bs in batch_sizes:
            print(f"Training LR={lr}, Batch Size={bs}...")
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
            
            model = HardenedDefectCNN(input_size=180)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            peak_val_acc = 0.0
            for epoch in range(epochs_grid):
                model.train()
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                val_acc = correct / total
                if val_acc > peak_val_acc:
                    peak_val_acc = val_acc
                    
            print(f" -> Peak Val Acc: {peak_val_acc:.4f}")
            grid_results.append({'lr': lr, 'batch_size': bs, 'peak_acc': peak_val_acc})
            
            if peak_val_acc > best_grid_acc:
                best_grid_acc = peak_val_acc
                best_grid_config = {'lr': lr, 'batch_size': bs}
                
    # 2. Markdown Table Output for Grid Search
    print("\nGrid Search Results:")
    print(f"| Learning Rate | Batch Size | Peak Val Accuracy |")
    print(f"|---------------|------------|-------------------|")
    for res in grid_results:
        print(f"| {res['lr']:<13} | {res['batch_size']:<10} | {res['peak_acc']:<17.4f} |")
        
    print(f"\nBest Grid Search Configuration: LR={best_grid_config['lr']}, Batch Size={best_grid_config['batch_size']} (Peak Acc: {best_grid_acc:.4f})\n")
    
    # 3. Learning Rate Scheduler: StepLR
    print("--- 3. StepLR Scheduler on Optimal Configuration ---")
    optimal_lr = best_grid_config['lr']
    optimal_bs = best_grid_config['batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=optimal_bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=optimal_bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=optimal_bs, shuffle=False)
    
    model_scheduler = HardenedDefectCNN(input_size=180)
    optimizer = optim.Adam(model_scheduler.parameters(), lr=optimal_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    epochs_sched = 15
    for epoch in range(epochs_sched):
        model_scheduler.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_scheduler(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    # Evaluate final test set accuracy
    model_scheduler.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model_scheduler(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc_sched = correct / total
    print(f"StepLR Final Test Set Accuracy (15 Epochs): {test_acc_sched:.4f}\n")
    
    # 4. Bayesian Optimization with Optuna
    print("--- 4. Optuna Study (Bayesian Optimization, 10 Trials) ---")
    
    def objective(trial):
        # Suggest learning rate (1e-4 to 1e-1, log-uniform)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        # Suggest batch size (8 to 64 categorical)
        bs = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        
        # Loader with suggested batch size
        trial_train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        trial_val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        
        # Build hardened model
        model = HardenedDefectCNN(input_size=180)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        peak_val_acc = 0.0
        # Train for 5 epochs per trial to keep search fast on CPU
        for epoch in range(5):
            model.train()
            for images, labels in trial_train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in trial_val_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            if acc > peak_val_acc:
                peak_val_acc = acc
        return peak_val_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    print("Optuna Study Summary:")
    print(f"Best Trial Peak Val Accuracy: {study.best_value:.4f}")
    print(f"Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
if __name__ == "__main__":
    run_tuning()
