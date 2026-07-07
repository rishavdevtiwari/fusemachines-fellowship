import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from collections import Counter

# MapDataset wrapper to apply split-specific transforms on Subsets
class MapDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.subset)

def get_dataset_splits(seed=42):
    train_dir = r"C:\Users\risha\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET\train\images"
    val_dir = r"C:\Users\risha\.cache\kagglehub\datasets\kaustubhdikshit\neu-surface-defect-database\versions\1\NEU-DET\validation\images"
    
    # Load using ImageFolder without transforms (to keep raw PIL Images for random_split)
    ds_train = ImageFolder(train_dir)
    ds_val = ImageFolder(val_dir)
    full_ds = ConcatDataset([ds_train, ds_val])
    
    # Calculate 80/10/10 split sizes
    total_len = len(full_ds)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    torch.manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(full_ds, [train_len, val_len, test_len])
    
    class_names = ds_train.classes
    return train_sub, val_sub, test_sub, class_names

def print_dataset_diagnostics(train_sub, val_sub, test_sub, class_names):
    print("--- Dataset Diagnostics ---")
    print(f"Total samples: {len(train_sub) + len(val_sub) + len(test_sub)}")
    print(f"Train size: {len(train_sub)}")
    print(f"Val size: {len(val_sub)}")
    print(f"Test size: {len(test_sub)}")
    
    # Verify class balance across splits
    def count_classes(subset):
        labels = []
        for idx in subset.indices:
            # ConcatDataset indexes: need to map dataset index to inner dataset label
            # subset.dataset is the ConcatDataset
            # ConcatDataset has cumulative_sizes
            labels.append(subset.dataset[idx][1])
        return Counter(labels)
        
    train_counts = count_classes(train_sub)
    val_counts = count_classes(val_sub)
    test_counts = count_classes(test_sub)
    
    print("\nClass Balance:")
    print(f"{'Class Name':<20} | {'Train Count':<12} | {'Val Count':<10} | {'Test Count':<10}")
    print("-" * 62)
    for i, name in enumerate(class_names):
        print(f"{name:<20} | {train_counts[i]:<12} | {val_counts[i]:<10} | {test_counts[i]:<10}")
    print()
