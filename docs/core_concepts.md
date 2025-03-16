# Core Concepts in Nova

This guide explains fundamental machine learning concepts using Nova's intuitive terminology, helping bridge the gap between natural language understanding and technical implementation.

## 1. Data Representation

### Feature Grids (Tensors)

In Nova, we use the term "feature grid" to represent what PyTorch calls tensors. These are multi-dimensional arrays that store numerical data.

```
# Creating a feature grid in Nova
create feature grid image_data with shape 3x224x224

# PyTorch equivalent
image_tensor = torch.zeros(3, 224, 224)
```

Feature grids have:

- **Dimensions**: The number of axes (1D, 2D, 3D, etc.)
- **Shape**: The size along each dimension
- **Data type**: The type of values stored (floating point, integer, etc.)

### Data Collections (Datasets)

A "data collection" in Nova represents a PyTorch Dataset. It contains multiple samples used for training or evaluation.

```
# Loading a data collection in Nova
load data collection mnist from torchvision.datasets

# PyTorch equivalent
from torchvision.datasets import MNIST
mnist_dataset = MNIST(root='./data', download=True, transform=transforms.ToTensor())
```

### Data Streams (DataLoaders)

A "data stream" represents a PyTorch DataLoader, which provides batches of data from a dataset.

```
# Creating a data stream in Nova
prepare data stream from mnist with batch size 32 and shuffle enabled

# PyTorch equivalent
mnist_dataloader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
```

## 2. Model Architecture

### Processing Pipelines (Neural Networks)

A "processing pipeline" in Nova represents a complete neural network model (PyTorch's nn.Module).

```
# Creating a processing pipeline in Nova
create processing pipeline classifier:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs

# PyTorch equivalent
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

### Transformation Stages (Layers)

A "transformation stage" represents a layer in a neural network. Different types include:

1. **Fully Connected** (Linear Layer)

   ```
   add transformation stage fully_connected with 784 inputs and 128 outputs
   # PyTorch: nn.Linear(784, 128)
   ```

2. **Feature Detector** (Convolutional Layer)

   ```
   add feature detector with 3 input channels, 16 output channels and 3x3 filter size
   # PyTorch: nn.Conv2d(3, 16, kernel_size=3, padding=1)
   ```

3. **Downsampling** (Pooling Layer)

   ```
   add downsampling using max method with size 2x2
   # PyTorch: nn.MaxPool2d(kernel_size=2, stride=2)
   ```

4. **Memory Cell** (Recurrent Layer)

   ```
   add memory cell lstm with 128 inputs and 256 hidden size
   # PyTorch: nn.LSTM(input_size=128, hidden_size=256)
   ```

### Activation Patterns (Activation Functions)

"Activation patterns" represent activation functions that introduce non-linearity:

```
apply relu activation
# PyTorch: nn.ReLU()

apply sigmoid activation
# PyTorch: nn.Sigmoid()

apply tanh activation
# PyTorch: nn.Tanh()
```

## 3. Training Process

### Error Measures (Loss Functions)

An "error measure" quantifies how far the model's predictions are from the true values:

```
measure error using mean_squared_error
# PyTorch: nn.MSELoss()

measure error using cross_entropy
# PyTorch: nn.CrossEntropyLoss()
```

### Improvement Strategies (Optimizers)

An "improvement strategy" defines how to update model parameters:

```
improve using gradient_descent with learning rate 0.01
# PyTorch: optim.SGD(model.parameters(), lr=0.01)

improve using adam with learning rate 0.001
# PyTorch: optim.Adam(model.parameters(), lr=0.001)
```

### Learning Cycles (Epochs)

A "learning cycle" represents one complete pass through the training dataset:

```
repeat for 10 learning cycles
# PyTorch: for epoch in range(10):
```

### Improvement Steps (Backpropagation & Update)

An "improvement step" represents the process of computing gradients and updating parameters:

```
# This happens implicitly in Nova's training syntax, but equates to:
# PyTorch:
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 4. Regularization Techniques

### Weight Decay

```
improve using adam with learning rate 0.001 and weight decay 0.0001
# PyTorch: optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
```

### Dropout

```
add dropout with rate 0.5
# PyTorch: nn.Dropout(0.5)
```

### Batch Normalization

```
add batch normalization with 128 features
# PyTorch: nn.BatchNorm1d(128)
```

## 5. Evaluation and Inference

### Performance Metrics

```
evaluate classifier on test_stream:
    measure accuracy, precision, and recall
    report results
```

### Making Predictions

```
use classifier to process new_data
# PyTorch: model(new_data)
```

## 6. Model Persistence

### Saving Models

```
save classifier to "models/classifier.pth"
# PyTorch: torch.save(model.state_dict(), "models/classifier.pth")
```

### Loading Models

```
load classifier from "models/classifier.pth"
# PyTorch: model.load_state_dict(torch.load("models/classifier.pth"))
```

## Conceptual Mappings

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

This conceptual mapping forms the foundation of Nova, making deep learning more intuitive and approachable while maintaining a clear path to the underlying PyTorch implementation.
