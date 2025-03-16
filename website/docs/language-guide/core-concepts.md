# Core Concepts

This guide explains the fundamental concepts of Nova and how they map to PyTorch constructs. Understanding these core concepts will help you effectively use Nova for deep learning tasks.

## Data Concepts

### Feature Grids (Tensors)

In Nova, we use the term "feature grid" to represent what PyTorch calls tensors. These are multi-dimensional arrays that store numerical data.

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
create feature grid image_data with shape 3x224x224
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
image_tensor = torch.zeros(3, 224, 224)
```

</div>
</div>
</div>

Feature grids have:
- **Dimensions**: The number of axes (1D, 2D, 3D, etc.)
- **Shape**: The size along each dimension
- **Data type**: The type of values stored (floating point, integer, etc.)

### Data Collections (Datasets)

A "data collection" in Nova represents a PyTorch Dataset. It contains multiple samples used for training or evaluation.

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
load data collection mnist from torchvision.datasets
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
from torchvision.datasets import MNIST
mnist_dataset = MNIST(root='./data', download=True, transform=transforms.ToTensor())
```

</div>
</div>
</div>

### Data Streams (DataLoaders)

A "data stream" represents a PyTorch DataLoader, which provides batches of data from a dataset.

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
prepare data stream from mnist with batch size 32 and shuffle enabled
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
mnist_dataloader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
```

</div>
</div>
</div>

## Model Architecture Concepts

### Processing Pipelines (Neural Networks)

A "processing pipeline" in Nova represents a complete neural network model (PyTorch's nn.Module).

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
create processing pipeline classifier:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

</div>
</div>
</div>

### Transformation Stages (Layers)

A "transformation stage" represents a layer in a neural network. Different types include:

#### 1. Fully Connected (Linear Layer)

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
add transformation stage fully_connected with 784 inputs and 128 outputs
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
nn.Linear(784, 128)
```

</div>
</div>
</div>

#### 2. Feature Detector (Convolutional Layer)

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
add feature detector with 3 input channels, 16 output channels and 3x3 filter size
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

</div>
</div>
</div>

#### 3. Downsampling (Pooling Layer)

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
add downsampling using max method with size 2x2
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

</div>
</div>
</div>

#### 4. Memory Cell (Recurrent Layer)

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
add memory cell lstm with 128 inputs and 256 hidden size
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
nn.LSTM(input_size=128, hidden_size=256)
```

</div>
</div>
</div>

### Activation Patterns (Activation Functions)

"Activation patterns" represent activation functions that introduce non-linearity:

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
apply relu activation
apply sigmoid activation
apply tanh activation
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
```

</div>
</div>
</div>

## Training Concepts

### Error Measures (Loss Functions)

An "error measure" quantifies how far the model's predictions are from the true values:

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
measure error using mean_squared_error
measure error using cross_entropy
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
nn.MSELoss()
nn.CrossEntropyLoss()
```

</div>
</div>
</div>

### Improvement Strategies (Optimizers)

An "improvement strategy" defines how to update model parameters:

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
improve using gradient_descent with learning rate 0.01
improve using adam with learning rate 0.001
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
optim.SGD(model.parameters(), lr=0.01)
optim.Adam(model.parameters(), lr=0.001)
```

</div>
</div>
</div>

### Learning Cycles (Epochs)

A "learning cycle" represents one complete pass through the training dataset:

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
repeat for 10 learning cycles
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
for epoch in range(10):
```

</div>
</div>
</div>

### Improvement Steps (Backpropagation & Update)

An "improvement step" represents the process of computing gradients and updating parameters:

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
# This happens implicitly in Nova's training syntax
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

</div>
</div>
</div>

## Regularization Concepts

### Weight Decay

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
improve using adam with learning rate 0.001 and weight decay 0.0001
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
```

</div>
</div>
</div>

### Dropout

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
add dropout with rate 0.5
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
nn.Dropout(0.5)
```

</div>
</div>
</div>

### Batch Normalization

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
add batch normalization with 128 features
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
nn.BatchNorm1d(128)
```

</div>
</div>
</div>

## Evaluation and Inference Concepts

### Performance Metrics

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
evaluate classifier on test_stream:
    measure accuracy, precision, and recall
    report results
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
def evaluate(model, test_loader):
    # Implementation of accuracy, precision, recall calculations
    pass
```

</div>
</div>
</div>

### Making Predictions

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
use classifier to process new_data
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
model(new_data)
```

</div>
</div>
</div>

## Model Persistence Concepts

### Saving Models

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
save classifier to "models/classifier.pth"
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
torch.save(model.state_dict(), "models/classifier.pth")
```

</div>
</div>
</div>

### Loading Models

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
load classifier from "models/classifier.pth"
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
model.load_state_dict(torch.load("models/classifier.pth"))
```

</div>
</div>
</div>

## Concept Mapping Summary

The table below summarizes how Nova concepts map to PyTorch:

| Domain | Nova Concept | PyTorch Concept |
|--------|-------------|----------------|
| **Data** | Feature Grid | Tensor |
|        | Data Collection | Dataset |
|        | Data Stream | DataLoader |
| **Model** | Processing Pipeline | nn.Module |
|         | Transformation Stage | Layer (Linear, Conv2d, etc.) |
|         | Feature Detector | Convolutional Layer |
|         | Downsampling | Pooling Layer |
|         | Memory Cell | RNN/LSTM/GRU |
|         | Connection Strengths | Weights & Biases |
| **Training** | Error Measure | Loss Function |
|           | Improvement Strategy | Optimizer |
|           | Learning Cycle | Epoch |
|           | Improvement Step | Backpropagation & Update |
| **Evaluation** | Performance Metric | Accuracy, Precision, etc. |

## Next Steps

Now that you understand the core concepts of Nova, proceed to the [Syntax](syntax.md) guide to learn how to combine these concepts into complete programs.