# Simple Neural Network Example

This example demonstrates how to create, train, and evaluate a simple fully-connected neural network for MNIST digit classification using Nova.

## Complete Nova Code

```nova
# Load data
load data collection mnist from torchvision.datasets with:
    apply normalization with mean 0.1307 and deviation 0.3081
    convert to feature grid

# Prepare data streams
prepare data stream train_stream from mnist.train with batch size 64 and shuffle enabled
prepare data stream test_stream from mnist.test with batch size 1000

# Create model
create processing pipeline digit_classifier:
    add transformation stage fully_connected with 784 inputs and 256 outputs
    apply relu activation
    add transformation stage fully_connected with 256 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs

# Train model
train digit_classifier on train_stream:
    measure error using cross_entropy
    improve using adam with learning rate 0.001
    repeat for 5 learning cycles
    print progress every 100 batches

# Evaluate model
evaluate digit_classifier on test_stream:
    measure accuracy
    report results

# Save model
save digit_classifier to "models/digit_classifier.pth"
```

## Equivalent PyTorch Code

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Create model
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = DigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}'
                  f' ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Evaluation function
def test(model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.0f}%)\n')
    
    return accuracy

# Train model
for epoch in range(1, 6):
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion)

# Save model
torch.save(model.state_dict(), "models/digit_classifier.pth")
```

## Step-by-Step Translation Explanation

### 1. Data Loading and Preparation

**Nova:**

```nova
load data collection mnist from torchvision.datasets with:
    apply normalization with mean 0.1307 and deviation 0.3081
    convert to feature grid

prepare data stream train_stream from mnist.train with batch size 64 and shuffle enabled
prepare data stream test_stream from mnist.test with batch size 1000
```

**PyTorch:**

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
```

**Explanation:**

1. The Nova code loads the MNIST dataset and applies normalization.
2. The PyTorch translation imports the necessary modules and creates transform composites.
3. The datasets are loaded from torchvision.datasets with appropriate transforms.
4. DataLoaders are created to serve as "data streams" with the specified batch sizes.

### 2. Model Definition

**Nova:**

```nova
create processing pipeline digit_classifier:
    add transformation stage fully_connected with 784 inputs and 256 outputs
    apply relu activation
    add transformation stage fully_connected with 256 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
```

**PyTorch:**

```python
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = DigitClassifier().to(device)
```

**Explanation:**

1. The Nova code defines a neural network with three fully-connected layers and ReLU activations.
2. The PyTorch translation creates a class that inherits from nn.Module.
3. The input size is 784 (28×28 pixels of MNIST images).
4. Each transformation stage is translated to a nn.Linear layer.
5. The ReLU activations are implemented as nn.ReLU.
6. The forward method defines how data flows through the network.

### 3. Training Setup and Execution

**Nova:**

```nova
train digit_classifier on train_stream:
    measure error using cross_entropy
    improve using adam with learning rate 0.001
    repeat for 5 learning cycles
    print progress every 100 batches
```

**PyTorch:**

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}'
                  f' ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Train model
for epoch in range(1, 6):
    train(model, train_loader, criterion, optimizer, epoch)
```

**Explanation:**

1. The Nova code specifies cross-entropy as the error measure (loss function).
2. It selects Adam as the improvement strategy (optimizer) with a learning rate of 0.001.
3. It specifies 5 learning cycles (epochs) and progress printing every 100 batches.
4. The PyTorch translation defines the criterion and optimizer accordingly.
5. It implements a training function that handles the training loop for one epoch.
6. It creates an outer loop to repeat the training for 5 epochs.

### 4. Evaluation

**Nova:**

```nova
evaluate digit_classifier on test_stream:
    measure accuracy
    report results
```

**PyTorch:**

```python
def test(model, loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.0f}%)\n')
    
    return accuracy

test(model, test_loader, criterion)
```

**Explanation:**

1. The Nova code specifies evaluation on the test data stream with accuracy measurement.
2. The PyTorch translation defines a test function that:
   - Sets the model to evaluation mode
   - Processes the test data without computing gradients
   - Calculates loss and accuracy
   - Prints the results

### 5. Model Saving

**Nova:**

```nova
save digit_classifier to "models/digit_classifier.pth"
```

**PyTorch:**

```python
torch.save(model.state_dict(), "models/digit_classifier.pth")
```

**Explanation:**

1. The Nova code specifies saving the model to a file.
2. The PyTorch translation uses torch.save to save the model's state dictionary.

## Usage and Modifications

This example can be modified in various ways:

1. **Change the model architecture**:

   ```nova
   # Add more layers or change layer sizes
   create processing pipeline digit_classifier:
       add transformation stage fully_connected with 784 inputs and 512 outputs
       apply relu activation
       add dropout with rate 0.2
       add transformation stage fully_connected with 512 inputs and 256 outputs
       apply relu activation
       add dropout with rate 0.2
       add transformation stage fully_connected with 256 inputs and 10 outputs
   ```

2. **Modify the training process**:

   ```nova
   # Use a different optimizer or learning rate schedule
   train digit_classifier on train_stream:
       measure error using cross_entropy
       improve using sgd with learning rate 0.01 and momentum 0.9
       reduce learning rate by factor 0.1 every 2 learning cycles
       repeat for 10 learning cycles
   ```

3. **Add regularization**:

   ```nova
   # Add weight decay for regularization
   improve using adam with learning rate 0.001 and weight decay 0.0001
   ```

## Common Pitfalls and Solutions

1. **Input reshaping**: Remember that MNIST images are 28×28 pixels, which need to be flattened to 784 inputs for fully-connected layers.

2. **Dimensions mismatch**: Make sure the dimensions of each layer match correctly. Output from one layer must match the input to the next.

3. **Device placement**: For efficient computation, ensure tensors are on the appropriate device (CPU/GPU) using `.to(device)`.

4. **Gradient accumulation**: Reset gradients before each backward pass using `optimizer.zero_grad()` to avoid accumulating gradients.

5. **Evaluation mode**: Remember to set `model.eval()` during evaluation to disable dropout and use running statistics for batch normalization.
