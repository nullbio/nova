# Nova Language Guide

Nova is a natural language interface for expressing deep learning concepts and operations. This guide defines the vocabulary, syntax, and semantic mappings that make up the Nova language.

## Core Concepts and Terminology

Nova reimagines neural networks as data transformation pipelines - a concept familiar to most programmers who have worked with ETL processes, Unix pipes, or functional programming chains.

### Data Concepts

| Nova Term | PyTorch Equivalent | Description |
|-----------|-------------------|-------------|
| `data collection` | Dataset | A collection of samples used for training or evaluation |
| `sample` | Individual data point | A single instance in a dataset |
| `feature grid` | Tensor | Multi-dimensional array of values |
| `data stream` | DataLoader | Iterable that provides batches of data |
| `data transformation` | Transform | Operations applied to preprocess data |

### Model Concepts

| Nova Term | PyTorch Equivalent | Description |
|-----------|-------------------|-------------|
| `processing pipeline` | nn.Module | Complete neural network model |
| `transformation stage` | Layer (nn.Linear, etc.) | Individual layers in a neural network |
| `connection strengths` | Weights and biases | Learnable parameters |
| `activation pattern` | Activation function output | Result after applying activation function |
| `feature detector` | Convolutional layer | Extracts spatial features from inputs |
| `feature map` | Conv layer output | Output from convolutional operations |
| `downsampling` | Pooling layers | Reduces spatial dimensions |
| `memory cell` | RNN/LSTM/GRU cell | Processes sequential data with memory |

### Learning Concepts

| Nova Term | PyTorch Equivalent | Description |
|-----------|-------------------|-------------|
| `error measure` | Loss function | Quantifies model prediction error |
| `improvement strategy` | Optimizer | Algorithm for updating model parameters |
| `learning cycle` | Epoch | Complete pass through training dataset |
| `improvement step` | Backpropagation & update | Parameter update based on gradients |
| `performance metric` | Evaluation metric | Measures model quality (accuracy, etc.) |
| `learning rate` | Learning rate | Controls size of parameter updates |
| `parameter tuner` | Optimizer | Adjusts model parameters systematically |

## Syntax Rules

### Data Operations

```
# Creating tensors
create feature grid [name] with shape [dimensions]
create feature grid [name] from [data]

# Loading data
load data collection [name] from [source]
create data collection [name] from [arrays/files/etc]

# Data preparation
split collection into [training_percent]% training and [testing_percent]% testing
prepare data stream from [collection] with batch size [number]

# Data transformations
define transformations for [collection]:
    resize images to [dimensions]
    normalize values using mean [mean_values] and deviation [std_values]
    [additional transformations...]
```

### Model Definition

```
# Basic model creation
create processing pipeline [name]:
    add transformation stage [type] with [inputs] inputs and [outputs] outputs
    apply [activation_function] activation
    [additional stages...]
    
# Convolutional networks
create image processing pipeline [name]:
    add feature detector with [channels] channels and [filter_size] filter size
    apply [activation_function] activation
    add downsampling using max method with size [dimensions]
    [additional stages...]
    flatten features
    add transformation stage fully_connected with [inputs] inputs and [outputs] outputs
    
# Recurrent networks
create sequence processing pipeline [name]:
    add memory cell [type] with [inputs] inputs and [hidden_size] hidden size
    [additional stages...]
```

### Training Process

```
# Basic training
train [pipeline] on [data_stream]:
    measure error using [error_measure]
    improve using [strategy] with learning rate [rate]
    repeat for [cycles] learning cycles
    
# Advanced training with validation
train [pipeline] on [training_stream]:
    measure error using [error_measure]
    improve using [strategy] with learning rate [rate]
    evaluate on [validation_stream] every [interval] steps
    save best model based on [metric]
    stop early if no improvement for [patience] cycles
    repeat for [cycles] learning cycles
```

### Evaluation and Inference

```
# Evaluation
evaluate [pipeline] on [test_stream]:
    measure [metrics]
    report results

# Prediction
use [pipeline] to process [input_data]
get predictions from [pipeline] for [input_data]
```

### Model Saving and Loading

```
# Saving
save [pipeline] to [filepath]
save [pipeline] parameters to [filepath]

# Loading
load [pipeline] from [filepath]
load parameters into [pipeline] from [filepath]
```

## Translation Examples

Below are examples showing Nova language and their PyTorch translations:

### Example 1: Basic Neural Network

**Nova:**
```
create processing pipeline digit_recognizer:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
    apply softmax activation

train digit_recognizer on mnist_data_stream:
    measure error using cross_entropy
    improve using gradient_descent with learning rate 0.01
    repeat for 10 learning cycles
```

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

model = DigitRecognizer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for data, target in mnist_data_stream:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### Example 2: Convolutional Neural Network

**Nova:**
```
create image processing pipeline image_classifier:
    add feature detector with 3 input channels, 16 output channels and 3x3 filter size
    apply relu activation
    add downsampling using max method with size 2x2
    add feature detector with 16 input channels, 32 output channels and 3x3 filter size
    apply relu activation
    add downsampling using max method with size 2x2
    flatten features
    add transformation stage fully_connected with 2048 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs

train image_classifier on cifar_data_stream:
    measure error using cross_entropy
    improve using adam with learning rate 0.001
    repeat for 20 learning cycles
```

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Assuming input image is 32x32
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

model = ImageClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    for data, target in cifar_data_stream:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Advanced Concepts

Nova also supports more advanced deep learning concepts including:

- Transfer learning
- Model ensembles
- Hyperparameter tuning
- Custom loss functions
- Learning rate scheduling
- Regularization techniques
- Advanced architectures (Transformers, GANs, etc.)

These advanced concepts will be covered in more detail in the [advanced guide](advanced_concepts.md).