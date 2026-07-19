# Steel Defect Classifier - Technical & Analytical Report
**SmartForge Manufacturing - Week 9 Assignment**

This report contains the technical documentation, model architectures, mathematical proofs, experimental results, and reflections for the Steel Defect Classifier project.

---

## PART 0: NN FOUNDATIONS

### 1. Custom 2-Layer MLP Architecture (`SimpleMLP`)
The custom multi-layer perceptron was implemented without using `nn.Sequential` to show explicit layer definition and connection in the `forward` pass:
- **Input layer**: Flattens the input image tensor to 40,000 dimensions (200x200 grayscale pixels).
- **Hidden layer**: Dense linear transformation to 128 units, followed by optional Batch Normalization, Activation (ReLU/Sigmoid), and Dropout.
- **Output layer**: Dense linear mapping from 128 units to 6 logits (defect classes).

The code is saved in [`part0_foundations.py`](file:///d:/Fusemachines_Fellowship/Week9_NeuralNetwork/part0_foundations.py).

### 2. Hidden Activation Comparison: ReLU vs. Sigmoid
Over 20 training epochs on simulated data, we observed the following convergence behaviors:
- **ReLU hidden activation**: Loss decreased from **1.7992** (Epoch 1) to **0.0196** (Epoch 20).
- **Sigmoid hidden activation**: Loss decreased from **1.8384** (Epoch 1) to **0.2180** (Epoch 20).

#### Convergence Analysis
The faster convergence of the **ReLU** activation function is caused by the **vanishing gradient problem** inherent in the **Sigmoid** activation.
- **Sigmoid** maps input values into the range $[0, 1]$. For large positive or negative inputs, the gradient of the Sigmoid function ($\sigma'(x) = \sigma(x)(1 - \sigma(x)$) approaches $0$ (saturates). This causes gradients to shrink exponentially as they propagate backward through layers, slowing parameter updates.
- **ReLU** ($f(x) = \max(0, x)$) has a constant gradient of $1.0$ for all positive inputs. It does not saturate in the positive region, allowing gradients to flow back freely without decay, resulting in faster and more stable convergence.

### 3. Loss Function Choice: Cross-Entropy vs. MSE
For multi-class classification, `nn.CrossEntropyLoss` is mathematically preferred over `nn.MSELoss` (Mean Squared Error) for the following reasons:

#### Gradient Saturation & Convergence Speed
`nn.CrossEntropyLoss` combines log-softmax and negative log-likelihood loss:
$$L_{CE} = -\sum_{i} y_i \log(\hat{p}_i)$$
where $\hat{p}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ is the softmax probability for class $i$, and $z_i$ is the logit. The gradient of $L_{CE}$ with respect to the logits $z_i$ is:
$$\frac{\partial L_{CE}}{\partial z_i} = \hat{p}_i - y_i$$
This gradient is linear with respect to the prediction error ($\hat{p}_i - y_i$). If the model is completely confident but incorrect (e.g., $y_i = 1$ but $\hat{p}_i \approx 0$), the gradient is large ($\approx -1$), forcing rapid weight correction.

Under `nn.MSELoss`:
$$L_{MSE} = \frac{1}{2} \sum_i (y_i - \hat{p}_i)^2$$
The gradient with respect to logit $z_i$ is:
$$\frac{\partial L_{MSE}}{\partial z_i} = ( \hat{p}_i - y_i ) \cdot \hat{p}_i (1 - \hat{p}_i)$$
Here, if the prediction is completely incorrect (e.g., $y_i = 1$ but $\hat{p}_i \approx 0$), the term $\hat{p}_i (1 - \hat{p}_i)$ approaches $0$, causing the gradient to vanish. The model gets "stuck" and fails to learn, which is known as **gradient saturation**.

#### Probabilistic Interpretation
Cross-entropy minimizes the Kullback-Leibler (KL) divergence between the true distribution and the predicted distribution, maximizing the likelihood of the correct class. MSE assumes a Gaussian noise distribution, which is inappropriate for categorical targets.

### 4. Optimizer Convergence & Memory Complexity

#### Convergence Analysis
- **SGD** (lr=0.01) ended with a training loss of **0.0186** at Epoch 20.
- **SGD + Momentum** (lr=0.01, momentum=0.9) ended with a training loss of **0.0001**.
- **Adam** (lr=0.001) ended with a training loss of **0.0000**.

**Adam** converged fastest due to its adaptive learning rates for each parameter, using moving averages of both the gradients (first moment) and squared gradients (second moment). **Momentum SGD** outperformed vanilla SGD by accumulating velocity in the direction of persistent gradients, smoothing out oscillations and traversing flat loss areas.

#### GPU Parameter States Memory Overhead Breakdown
For `SimpleMLP` (Input = 40,000, Hidden = 128, Output = 6):
- **W1 (fc1 weights)**: $40000 \times 128 = 5,120,000$ params
- **b1 (fc1 biases)**: $128$ params
- **W2 (fc2 weights)**: $128 \times 6 = 768$ params
- **b2 (fc2 biases)**: $6$ params
- **Total parameters ($N$)**: $5,120,902$

Using 32-bit single-precision floating point (4 bytes per param), the model parameters occupy:
$$\text{Params Memory} = 5,120,902 \times 4 \text{ bytes} \approx 20,483,608 \text{ bytes} \approx 19.53 \text{ MB}$$

| Optimizer | State Vectors per Parameter | Formula | GPU Memory Overhead (MB) |
|---|---|---|---|
| **SGD** | 0 (No state tracking) | $0 \times N \times 4$ bytes | **0.00 MB** |
| **SGD + Momentum** | 1 (Velocity buffer) | $1 \times N \times 4$ bytes | **19.53 MB** |
| **Adam** | 2 (First moment $m$, Second moment $v$) | $2 \times N \times 4$ bytes | **39.07 MB** |

### 5. Training Stability: BatchNorm1d vs. Dropout (0.3)
- **BatchNorm1d variant**: Val Loss = **1.8472**, Val Acc = **0.1200**
- **Dropout(0.3) variant**: Val Loss = **2.3003**, Val Acc = **0.1600**

#### Architectural Mechanics
- **BatchNorm1d**: Normalizes the activations of the hidden layer across the batch dimension. It stabilizes training by maintaining zero mean and unit variance, reducing internal covariate shift. However, in small, simulated batches, the batch statistics can fluctuate widely, leading to noisy validation performance.
- **Dropout**: Randomly zeroes out $30\%$ of the hidden units during each forward pass. This prevents co-adaptation of features, forcing the remaining network units to learn robust, redundant representations. The high validation loss indicates overfitting on simulated noise, but it provides a slightly higher validation accuracy here as it restricts the network capacity and forces regularization.

---

## PART A: BASLINE CNN CLASSIFIER

### 1. Dataset Diagnostics & Class Balance
The NEU dataset consists of 1800 grayscale images of shape $200 \times 200$. Concatenated and randomly split into 80/10/10:
- **Train size**: 1440 images
- **Validation size**: 180 images
- **Test size**: 180 images

#### Class Balance Across Splits
| Class Name | Train Count | Val Count | Test Count |
|---|---|---|---|
| crazing | 236 | 28 | 36 |
| inclusion | 233 | 32 | 35 |
| patches | 252 | 22 | 26 |
| pitted_surface | 241 | 30 | 29 |
| rolled-in_scale | 232 | 39 | 29 |
| scratches | 246 | 29 | 25 |

The split is highly balanced, preserving the 300-samples-per-class distribution.

### 2. The Criticality of Input Normalization
Prior to training a CNN, pixel inputs are normalized (in our case to range $[-1, 1]$ via mean=0.5, std=0.5). This is critical because:
1. **Numerical Stability**: Normalizing values to a zero-centered small range (e.g., $[-1, 1]$) prevents numerical overflows during backpropagation and stabilizes gradients.
2. **Eliminating Scale Bias**: Unnormalized pixel values ($[0, 255]$) cause large input magnitudes to produce huge activations and gradients, resulting in unstable updates.
3. **Loss Landscape Conditioning**: Centering data makes the loss function's Hessian matrix more spherical (less elongated). This allows gradient descent to travel directly toward the minimum without oscillating, enabling larger learning rates and faster convergence.

### 3. Baseline CNN Spatial Transformations
Input tensor shape: $(B, 1, 200, 200)$

1. **`Conv2d` (in=1, out=16, k=3, p=1, s=1)**
   $$H_{out} = \frac{200 - 3 + 2(1)}{1} + 1 = 200$$
   *Shape*: $(B, 16, 200, 200)$
2. **`ReLU`**
   *Shape*: $(B, 16, 200, 200)$
3. **`MaxPool2d` (k=2, s=2)**
   $$H_{out} = \frac{200}{2} = 100$$
   *Shape*: $(B, 16, 100, 100)$
4. **`Conv2d` (in=16, out=32, k=3, p=1, s=1)**
   $$H_{out} = \frac{100 - 3 + 2(1)}{1} + 1 = 100$$
   *Shape*: $(B, 32, 100, 100)$
5. **`ReLU`**
   *Shape*: $(B, 32, 100, 100)$
6. **`MaxPool2d` (k=2, s=2)**
   $$H_{out} = \frac{100}{2} = 50$$
   *Shape*: $(B, 32, 50, 50)$
7. **`Flatten`**
   $$32 \times 50 \times 50 = 80,000$$
   *Shape*: $(B, 80000)$
8. **`Linear` (in=80000, out=6)**
   *Shape*: $(B, 6)$

### 4. Baseline CNN Training Logs
Saved in [`parta_baseline.py`](file:///d:/Fusemachines_Fellowship/Week9_NeuralNetwork/parta_baseline.py).

### 5. Overfitting Epoch Analysis
Based on the loss curves saved in `parta_baseline.png`:
- **Training Loss** steadily drops from **1.3097** to **0.0425** (Epoch 15).
- **Validation Loss** drops to **0.1694** at **Epoch 9**, but then begins to rise, reaching **0.2481** at Epoch 15.
- **Overfitting begins at Epoch 10**. Beyond this point, the validation loss begins to diverge upward while the training loss continues downward, indicating that the model is memorizing the training set noise rather than generalizing.

### 6. Per-Class Test Set F1-Scores & Visual Confusion Analysis
- **crazing**: 0.9351
- **inclusion**: 0.9254
- **patches**: 0.8936 *(Lowest F1-score)*
- **pitted_surface**: 0.9180
- **rolled-in_scale**: 0.9831
- **scratches**: 0.9796

#### Visual Confusion Analysis: Crazing vs. Patches
A model naturally confuses 'crazing' and 'patches' defect classes due to shared structural features:
- **Crazing** presents as fine, web-like network cracks on the steel surface.
- **Patches** appear as dark, irregular regions or localized surface scaling.
At local scales, both defects exhibit similar irregular edge boundaries, varying contrast, and high-frequency textures. When crazing is dense or when patches are fragmented, a baseline CNN lacking spatial context or rotation invariance struggles to distinguish the fine crack networks of crazing from the boundary edges of patches, leading to classification errors.

---

## PART B: MODEL HARDENING

### 1. Data Augmentation Rationale
- **Training Set**: RandomHorizontalFlip, RandomRotation(15), and RandomCrop(180, padding=10).
- **Validation/Test Set**: strictly non-augmented (only resized to 180x180 and normalized).
- **Rationale**: Augmentation synthesizes realistic variations (rotation, shifts, flips) to prevent the network from memorizing fixed spatial features, effectively acting as a regularizer. However, the validation and test sets must represent the real-world inference distribution. Applying random augmentations to them would introduce stochastic noise into the evaluation metrics, making validation metrics unstable and failing to represent the model's true performance on unseen, clean images.

### 2. Batch Normalization Mechanics
Adding `nn.BatchNorm2d` after every `Conv2d` layer stabilizes training:
- **Loss Landscape Smoothing**: BatchNorm smooths the optimization landscape. It reduces the Lipschitz constant of the loss function and makes the gradients more predictive and stable, allowing for higher learning rates.
- **Internal Covariate Shift**: By re-centering and re-scaling activations at each layer, it ensures that subsequent layers do not have to constantly adapt to shifting input distributions during optimization.

### 3. Dropout Mechanics
- **nn.Dropout(0.4)** before the final linear layer randomly drops 40% of activations.
- **During Training**: Dropout forces the network to learn redundant representations, preventing reliance on any single node. It acts as an ensemble of smaller sub-networks, reducing effective model capacity.
- **During Inference (`model.eval()`)**: Dropout is deactivated. All connections are active, and activations are scaled by $(1 - p) = 0.6$ (or PyTorch scales them up by $1/(1-p)$ during training so no scaling is needed at evaluation) to ensure the full capacity of the network is utilized for stable predictions.

### 4. Hardening Validation Accuracy Performance Comparison
- **Configuration 1: Baseline (Part A)**: Final Val Acc = **0.9222**
- **Configuration 2: Augmentations Only**: Final Val Acc = **0.8722**
- **Configuration 3: Augmentations + BN + Dropout (Hardened)**: Final Val Acc = **0.9389**

Adding augmentations alone makes the training task harder, which initially decreases validation accuracy on the small dataset without training support. However, when combined with **BatchNorm2d** (which stabilizes convergence) and **Dropout** (which prevents overfitting), the model achieves the highest validation accuracy (**93.89%**) and is robust against orientation and scale changes.

The comparative plot is saved at [`partb_comparison.png`](file:///d:/Fusemachines_Fellowship/Week9_NeuralNetwork/partb_comparison.png).

### 5. Prioritization Under Engineering Constraints
If allowed only a single technique to deploy on a new manufacturing line dataset, **Data Augmentation** should be prioritized. 
- **Reason**: CNNs are highly sensitive to spatial transformations. On a real manufacturing line, camera angles, lighting, and steel orientation vary. Standard convolutional layers are not rotation-invariant. Data augmentation directly encodes these variations into the training distribution, forcing the model to learn invariant representations. Without it, architectural techniques like BatchNorm and Dropout will simply overfit to the clean, centered dataset structure and fail under real-world shifts.

---

## PART C: HYPERPARAMETER TUNING & OPTIMIZATION

### 1. Manual Grid Search Results

| Learning Rate | Batch Size | Peak Val Accuracy |
|---|---|---|
| 0.001 | 16 | 0.9000 |
| 0.001 | 32 | 0.9056 |
| 0.01 | 16 | 0.9056 |
| 0.01 | 32 | **0.9278** |

*Best configuration found*: **LR = 0.01, Batch Size = 32**

### 2. Hyperparameter Leverage Analysis
The **Learning Rate** demonstrated greater leverage over model variance:
- Increasing LR from 0.001 to 0.01 resulted in a performance jump (up to +2.78% accuracy at BS=32).
- Batch size changes had a minor effect (+0.56% at LR=0.001).
A higher learning rate combined with Batch Normalization allows the model to escape local minima quickly in a complex augmented loss landscape, showing that the optimizer step size is the primary driver of convergence speed and quality.

### 3. Learning Rate Scheduler (StepLR) Impact
Integrating `StepLR` (step_size=5, gamma=0.5) to the optimal configuration (LR=0.01, Batch Size=32):
- **Final Test Set Accuracy**: **0.9167**
- **Impact**: StepLR decays the learning rate by half every 5 epochs. This helps the optimizer converge smoothly into narrow, deep minima during later training stages, preventing oscillations around the optimum and resulting in a highly stable test performance.

### 4. Bayesian Optimization (Optuna) Results
Optuna completed 10 trials searching over learning rate ($1e-4$ to $1e-1$ log-uniform) and batch size ($8$ to $64$ categorical):
- **Best Trial Peak Val Accuracy**: **0.9500** *(Exceeds manual grid search!)*
- **Optimal Hyperparameters**:
  - **Learning Rate**: `0.0042456`
  - **Batch Size**: `16`

Optuna's Tree-structured Parzen Estimator (TPE) successfully identified a better learning rate between $0.001$ and $0.01$, and paired it with a smaller batch size of 16, which increases the gradient noise frequency and helps escape flat saddle points in the loss landscape, yielding superior generalization.
