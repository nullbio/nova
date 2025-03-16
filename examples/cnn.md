# Convolutional Neural Network Example

This example demonstrates how to create, train, and evaluate a convolutional neural network (CNN) for image classification using Nova.

## Complete Nova Code

```nova
# Load data
load data collection cifar10 from torchvision.datasets with:
    apply image transformations:
        resize to 32x32
        convert to tensor
        normalize with mean [0.4914, 0.4822, 0.4465] and deviation [0.2470, 0.2435, 0.2616]

# Prepare data streams
prepare data stream train_stream from cifar10.train with batch size 128 and shuffle enabled
prepare data stream test_stream from cifar10.test with batch size 1000

# Create model
create image processing pipeline cifar_classifier:
    # First convolutional block
    add feature detector with 3 input channels, 32 output channels and 3x3 filter size and padding same
    apply batch normalization with 32 features
    apply relu activation
    add feature detector with 32 input channels, 32 output channels and 3x3 filter size and padding same
    apply batch normalization with 32 features
    apply relu activation
    add downsampling using max method with size 2x2
    add dropout with rate 0.2
    
    # Second convolutional block
    add feature detector with 32 input channels, 64 output channels and 3x3 filter size and padding same
    apply batch normalization with 64 features
    apply relu activation
    add feature detector with 64 input channels, 64 output channels and 3x3 filter size and padding same
    apply batch normalization with 64 features
    apply relu activation
    add downsampling using max method with size 2x2
    add dropout with rate 0.3
    
    # Third convolutional block
    add feature detector with 64 input channels, 128 output channels and 3x3 filter size and padding same
    apply batch normalization with 128 features
    apply relu activation
    add feature detector with 128 input channels, 128 output channels and 3x3 filter size and padding same
    apply batch normalization with 128 features
    apply relu activation
    add downsampling using max method with size 2x2
    add dropout with rate 0.4
    
    # Fully connected layers
    flatten features
    add transformation stage fully_connected with 2048 inputs and 128 outputs
    apply batch normalization with 128 features
    apply relu activation
    add dropout with rate 0.5
    add transformation stage fully_connected with 128 inputs and 10 outputs

# Train model
train cifar_classifier on train_stream:
    measure error using cross_entropy
    improve using adam with learning rate 0.001 and weight decay 0.0001
    reduce learning rate by factor 0.1 when plateau in validation loss for 5 cycles
    repeat for 50 learning cycles
    stop early if no improvement for 10 cycles
    print progress every 50 batches

# Evaluate model
evaluate cifar_classifier on test_stream:
    measure accuracy, precision, recall, and f1 score
    report detailed results

# Save model
save cifar_classifier to "models/cifar_classifier.pth"
```

## Equivalent PyTorch Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and preprocessing
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# CNN Model definition
class CifarClassifier(nn.Module):
    def __init__(self):
        super(CifarClassifier, self).__init__()
        # First convolutional block
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.2)
        
        # Second convolutional block
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.3)
        
        # Third convolutional block
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 128)  # 2048 = 128 * 4 * 4
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # First block
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)
        
        # Flatten features
        x = x.view(-1, 2048)
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

# Create model and define loss function and optimizer
model = CifarClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

# Training function
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
        if batch_idx % 50 == 49:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {running_loss / 50:.4f}')
            running_loss = 0.0
    
    return running_loss

# Validation function
def validate(model, loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return val_loss, accuracy

# Test function with detailed metrics
def test(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Collect predictions and targets for detailed metrics
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / len(loader.dataset)
    
    # Calculate additional metrics
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return accuracy, precision, recall, f1

# Training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0
best_model_path = "models/cifar_classifier_best.pth"

for epoch in range(50):
    train_loss = train(model, train_loader, criterion, optimizer, epoch)
    val_loss, val_accuracy = validate(model, test_loader, criterion)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")
        
    if patience_counter >= 10:
        print("Early stopping triggered")
        break

# Load the best model for final evaluation
model.load_state_dict(torch.load(best_model_path))

# Final evaluation
print("\nFinal Evaluation:")
test_accuracy, precision, recall, f1 = test(model, test_loader)

# Save the final model
torch.save(model.state_dict(), "models/cifar_classifier.pth")
print("Final model saved to models/cifar_classifier.pth")
```

## Step-by-Step Translation Explanation

### 1. Data Preparation

**Nova:**

```nova
load data collection cifar10 from torchvision.datasets with:
    apply image transformations:
        resize to 32x32
        convert to tensor
        normalize with mean [0.4914, 0.4822, 0.4465] and deviation [0.2470, 0.2435, 0.2616]

prepare data stream train_stream from cifar10.train with batch size 128 and shuffle enabled
prepare data stream test_stream from cifar10.test with batch size 1000
```

**PyTorch:**

```python
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
```

**Explanation:**

1. The Nova code loads CIFAR-10 dataset and applies transformations.
2. The PyTorch translation:
   - Creates composite transforms for preprocessing
   - Loads the datasets with transforms applied
   - Creates dataloaders with specified batch sizes and shuffling options

### 2. Model Architecture

**Nova:**

```nova
create image processing pipeline cifar_classifier:
    # First convolutional block
    add feature detector with 3 input channels, 32 output channels and 3x3 filter size and padding same
    apply batch normalization with 32 features
    apply relu activation
    ...
```

**PyTorch:**

```python
class CifarClassifier(nn.Module):
    def __init__(self):
        super(CifarClassifier, self).__init__()
        # First convolutional block
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        ...
    
    def forward(self, x):
        # First block
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        ...
```

**Explanation:**

1. The Nova code defines a CNN with multiple convolutional blocks, each containing:
   - Convolutional layers ("feature detectors")
   - Batch normalization
   - ReLU activation
   - Max pooling ("downsampling")
   - Dropout regularization
2. The PyTorch translation:
   - Creates a class that inherits from nn.Module
   - Defines each layer as an attribute in **init**
   - Implements the forward method to define the flow of data through the network
   - Uses functional interface (F.relu, F.max_pool2d) for some operations

### 3. Training Process with Advanced Features

**Nova:**

```nova
train cifar_classifier on train_stream:
    measure error using cross_entropy
    improve using adam with learning rate 0.001 and weight decay 0.0001
    reduce learning rate by factor 0.1 when plateau in validation loss for 5 cycles
    repeat for 50 learning cycles
    stop early if no improvement for 10 cycles
    print progress every 50 batches
```

**PyTorch:**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

# Training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0
best_model_path = "models/cifar_classifier_best.pth"

for epoch in range(50):
    train_loss = train(model, train_loader, criterion, optimizer, epoch)
    val_loss, val_accuracy = validate(model, test_loader, criterion)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs")
        
    if patience_counter >= 10:
        print("Early stopping triggered")
        break
```

**Explanation:**

1. The Nova code specifies:
   - Cross-entropy loss function
   - Adam optimizer with weight decay
   - Learning rate reduction on plateau
   - Maximum number of epochs
   - Early stopping condition
   - Progress printing frequency
2. The PyTorch translation:
   - Initializes the CrossEntropyLoss criterion
   - Creates Adam optimizer with specified parameters
   - Sets up ReduceLROnPlateau scheduler
   - Implements a training loop with early stopping logic
   - Includes model saving for the best validation performance

### 4. Evaluation with Multiple Metrics

**Nova:**

```nova
evaluate cifar_classifier on test_stream:
    measure accuracy, precision, recall, and f1 score
    report detailed results
```

**PyTorch:**

```python
def test(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Collect predictions and targets for detailed metrics
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / len(loader.dataset)
    
    # Calculate additional metrics
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return accuracy, precision, recall, f1

# Final evaluation
print("\nFinal Evaluation:")
test_accuracy, precision, recall, f1 = test(model, test_loader)
```

**Explanation:**

1. The Nova code specifies evaluation with multiple performance metrics.
2. The PyTorch translation:
   - Implements a test function that collects all predictions and targets
   - Calculates accuracy directly from correct predictions
   - Uses scikit-learn functions to calculate precision, recall, and F1 score
   - Prints detailed results and returns calculated metrics

### 5. Model Saving

**Nova:**

```nova
save cifar_classifier to "models/cifar_classifier.pth"
```

**PyTorch:**

```python
torch.save(model.state_dict(), "models/cifar_classifier.pth")
```

**Explanation:**

1. The Nova code specifies saving the trained model to a file.
2. The PyTorch translation saves the model's state dictionary using torch.save.

## Advanced Concepts Explained

### 1. Batch Normalization

**Nova:**

```nova
apply batch normalization with 32 features
```

**PyTorch:**

```python
self.bn1_1 = nn.BatchNorm2d(32)
```

**Explanation:** Batch normalization normalizes activations within a mini-batch, which helps with training stability and convergence. The number of features corresponds to the number of channels in the input tensor.

### 2. Learning Rate Scheduling

**Nova:**

```nova
reduce learning rate by factor 0.1 when plateau in validation loss for 5 cycles
```

**PyTorch:**

```python
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
scheduler.step(val_loss)
```

**Explanation:** This strategy reduces the learning rate when the validation loss stops improving. The "patience" value determines how many epochs to wait before reducing the learning rate.

### 3. Early Stopping

**Nova:**

```nova
stop early if no improvement for 10 cycles
```

**PyTorch:**

```python
if patience_counter >= 10:
    print("Early stopping triggered")
    break
```

**Explanation:** Early stopping prevents overfitting by stopping training when the validation performance stops improving. This implementation tracks the number of epochs without improvement and stops training when it reaches the patience threshold.

### 4. Model Checkpointing

**PyTorch:**

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), best_model_path)
```

**Explanation:** While not explicitly mentioned in the Nova code, the PyTorch implementation saves the model state whenever the validation loss improves. This ensures we keep the best performing model, not just the final one.

## Usage and Modifications

### 1. Data Augmentation

You can enhance the model's performance by adding data augmentation:

```nova
apply image transformations:
    resize to 32x32
    random horizontal flip with probability 0.5
    random crop to 32x32 with padding 4
    convert to tensor
    normalize with mean [0.4914, 0.4822, 0.4465] and deviation [0.2470, 0.2435, 0.2616]
```

### 2. Transfer Learning

For improved performance, you could use transfer learning:

```nova
create image processing pipeline transfer_classifier:
    load pretrained resnet18 model
    freeze all layers except final layer
    replace final layer with transformation stage fully_connected with 512 inputs and 10 outputs
```

### 3. Different Architecture

Try a different architecture like ResNet-style connections:

```nova
create image processing pipeline residual_classifier:
    # Residual block implementation
    add residual block with 3 input channels, 32 output channels
    add residual block with 32 input channels, 64 output channels
    add residual block with 64 input channels, 128 output channels
    flatten features
    add transformation stage fully_connected with 2048 inputs and 10 outputs
```

## Performance Considerations

1. **Batch Size**: Adjust batch size based on available memory. Larger batches can lead to more stable gradients but require more memory.

2. **Model Complexity**: The model has around 3 million parameters. For faster training or inference on limited hardware, consider reducing the number of filters or layers.

3. **GPU Acceleration**: The code automatically uses GPU if available. For multi-GPU training, consider using `nn.DataParallel` or `nn.DistributedDataParallel`.

4. **Mixed Precision Training**: For further speedup on supported GPUs, add mixed precision training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
