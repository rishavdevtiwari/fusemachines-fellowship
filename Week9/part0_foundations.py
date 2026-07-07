import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import os

# 1. Custom 2-layer MLP without nn.Sequential
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', use_bn=False, dropout_prob=0.0):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_dim)
            
        self.activation_type = activation
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Activation type '{activation}' is not supported.")
            
        self.dropout_prob = dropout_prob
        if dropout_prob > 0.0:
            self.drop = nn.Dropout(dropout_prob)
            
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        
        # Layer 2 / Output
        if self.dropout_prob > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        return x

def get_simulated_data(num_train=300, num_val=100, input_dim=40000, num_classes=6):
    torch.manual_seed(42)
    X_train = torch.randn(num_train, input_dim)
    y_train = torch.randint(0, num_classes, (num_train,))
    X_val = torch.randn(num_val, input_dim)
    y_val = torch.randint(0, num_classes, (num_val,))
    return X_train, y_train, X_val, y_val

def train_mlp(model, X_train, y_train, optimizer, criterion, epochs=20, batch_size=32):
    epoch_losses = []
    num_samples = X_train.size(0)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(num_samples)
        running_loss = 0.0
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_losses.append(running_loss / num_samples)
    return epoch_losses

def evaluate_mlp(model, X_val, y_val, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        loss = criterion(outputs, y_val)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == y_val).float().mean().item()
    return loss.item(), accuracy

def run_experiments():
    # Make sure output directory exists
    os.makedirs("d:/Fusemachines_Fellowship/Week9", exist_ok=True)
    
    input_dim = 40000  # 200 * 200 grayscale image size
    hidden_dim = 128
    output_dim = 6
    criterion = nn.CrossEntropyLoss()
    
    print("--- Simulating Dataset ---")
    X_train, y_train, X_val, y_val = get_simulated_data(num_train=300, num_val=100, input_dim=input_dim, num_classes=output_dim)
    print(f"Train features shape: {X_train.shape}, Train labels shape: {y_train.shape}")
    print(f"Val features shape: {X_val.shape}, Val labels shape: {y_val.shape}\n")
    
    # 2. Compare ReLU vs Sigmoid over 20 epochs
    print("--- 2. Activation Function Comparison: ReLU vs Sigmoid ---")
    model_relu = SimpleMLP(input_dim, hidden_dim, output_dim, activation='relu')
    model_sig = SimpleMLP(input_dim, hidden_dim, output_dim, activation='sigmoid')
    
    opt_relu = optim.SGD(model_relu.parameters(), lr=0.01)
    opt_sig = optim.SGD(model_sig.parameters(), lr=0.01)
    
    losses_relu = train_mlp(model_relu, X_train, y_train, opt_relu, criterion, epochs=20)
    losses_sig = train_mlp(model_sig, X_train, y_train, opt_sig, criterion, epochs=20)
    
    print(f"ReLU Loss (Epoch 1): {losses_relu[0]:.4f} -> (Epoch 20): {losses_relu[-1]:.4f}")
    print(f"Sigmoid Loss (Epoch 1): {losses_sig[0]:.4f} -> (Epoch 20): {losses_sig[-1]:.4f}\n")
    
    # 4. Compare Optimizers: SGD, SGD (momentum=0.9), and Adam
    print("--- 4. Optimizer Comparison ---")
    model_base = SimpleMLP(input_dim, hidden_dim, output_dim, activation='relu')
    
    model_sgd = copy.deepcopy(model_base)
    model_mom = copy.deepcopy(model_base)
    model_adam = copy.deepcopy(model_base)
    
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
    opt_mom = optim.SGD(model_mom.parameters(), lr=0.01, momentum=0.9)
    opt_adam = optim.Adam(model_adam.parameters(), lr=0.001) # Adam typically uses smaller lr
    
    losses_sgd = train_mlp(model_sgd, X_train, y_train, opt_sgd, criterion, epochs=20)
    losses_mom = train_mlp(model_mom, X_train, y_train, opt_mom, criterion, epochs=20)
    losses_adam = train_mlp(model_adam, X_train, y_train, opt_adam, criterion, epochs=20)
    
    print(f"SGD Loss (Epoch 20): {losses_sgd[-1]:.4f}")
    print(f"SGD + Momentum Loss (Epoch 20): {losses_mom[-1]:.4f}")
    print(f"Adam Loss (Epoch 20): {losses_adam[-1]:.4f}\n")
    
    # Plotting optimizer curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 21), losses_sgd, label='SGD (lr=0.01)', color='red', linestyle='--')
    plt.plot(range(1, 21), losses_mom, label='SGD + Momentum (lr=0.01, mom=0.9)', color='blue', linestyle='-.')
    plt.plot(range(1, 21), losses_adam, label='Adam (lr=0.001)', color='green', linestyle='-')
    plt.title('Loss Convergence Comparison Across Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = "d:/Fusemachines_Fellowship/Week9/part0_optimizers.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved optimizer convergence plot to: {plot_path}\n")
    
    # 5. Training stability: BatchNorm1d vs Dropout(0.3)
    print("--- 5. Training Stability: BatchNorm1d vs Dropout ---")
    model_bn = SimpleMLP(input_dim, hidden_dim, output_dim, activation='relu', use_bn=True)
    model_dropout = SimpleMLP(input_dim, hidden_dim, output_dim, activation='relu', dropout_prob=0.3)
    
    opt_bn = optim.Adam(model_bn.parameters(), lr=0.001)
    opt_drop = optim.Adam(model_dropout.parameters(), lr=0.001)
    
    _ = train_mlp(model_bn, X_train, y_train, opt_bn, criterion, epochs=20)
    _ = train_mlp(model_dropout, X_train, y_train, opt_drop, criterion, epochs=20)
    
    val_loss_bn, val_acc_bn = evaluate_mlp(model_bn, X_val, y_val, criterion)
    val_loss_drop, val_acc_drop = evaluate_mlp(model_dropout, X_val, y_val, criterion)
    
    print(f"BatchNorm1d variant - Val Loss: {val_loss_bn:.4f}, Val Acc: {val_acc_bn:.4f}")
    print(f"Dropout(0.3) variant - Val Loss: {val_loss_drop:.4f}, Val Acc: {val_acc_drop:.4f}\n")
    
    total_params = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    param_size_mb = (total_params * 4) / (1024 * 1024)
    print(f"SimpleMLP Param Count: {total_params:,} parameters")
    print(f"Model Parameters Memory: {param_size_mb:.2f} MB")
    print(f"GPU Memory Overhead Estimate for Optimizer States:")
    print(f" - SGD: 0.00 MB")
    print(f" - SGD + Momentum: {param_size_mb:.2f} MB")
    print(f" - Adam: {param_size_mb * 2:.2f} MB")

if __name__ == "__main__":
    run_experiments()
