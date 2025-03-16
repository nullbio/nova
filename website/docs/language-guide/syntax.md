# Nova Syntax Guide

This guide provides a comprehensive reference for Nova's syntax, covering all the major operations and constructs available in the language.

## Basic Syntax Rules

Nova's syntax follows a few general principles:

1. **Natural language constructs**: Instructions and operations are written in a way that reads like natural language
2. **Hierarchical structure**: Operations can be nested under higher-level constructs
3. **Explicit parameters**: Parameter names are explicitly stated for clarity
4. **Consistent patterns**: Similar operations follow similar syntax patterns

## Data Operations

### Creating Tensors

```
# Create an empty tensor with a specified shape
create feature grid [name] with shape [dimensions]

# Create a tensor from existing data
create feature grid [name] from [data]

# Examples
create feature grid input_data with shape 3x224x224
create feature grid weights from numpy_array
```

### Loading Datasets

```
# Load a dataset from a standard source
load data collection [name] from [source]

# Create a dataset from files or arrays
create data collection [name] from [arrays/files/etc]

# Examples
load data collection mnist from torchvision.datasets
create data collection custom_dataset from "data/images/"
```

### Data Preparation

```
# Split a dataset for training and testing
split collection into [training_percent]% training and [testing_percent]% testing

# Create a data loader from a dataset
prepare data stream from [collection] with batch size [number]

# Optional parameters for data streams
prepare data stream from [collection] with batch size [number] and shuffle enabled

# Examples
split mnist into 80% training and 20% testing
prepare data stream train_stream from mnist.train with batch size 64 and shuffle enabled
```

### Data Transformations

```
# Define transformations for data
define transformations for [collection]:
    [transformation operations...]

# Common transformation operations
resize images to [dimensions]
normalize values using mean [mean_values] and deviation [std_values]
random horizontal flip with probability [probability]

# Examples
define transformations for cifar10:
    resize images to 32x32
    normalize values using mean [0.4914, 0.4822, 0.4465] and deviation [0.2470, 0.2435, 0.2616]
    random horizontal flip with probability 0.5
```

## Model Definition

### Basic Model Creation

```
# Create a basic neural network model
create processing pipeline [name]:
    [layer definitions...]
    
# Examples
create processing pipeline digit_recognizer:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
```

### Fully Connected Layers

```
# Add a fully connected (linear) layer
add transformation stage fully_connected with [inputs] inputs and [outputs] outputs

# Examples
add transformation stage fully_connected with 784 inputs and 128 outputs
add transformation stage fully_connected with 128 inputs and 10 outputs
```

### Convolutional Layers

```
# Add a convolutional layer
add feature detector with [in_channels] input channels, [out_channels] output channels and [size]x[size] filter size

# With optional parameters
add feature detector with [in_channels] input channels, [out_channels] output channels, [size]x[size] filter size and padding [padding]

# Examples
add feature detector with 3 input channels, 16 output channels and 3x3 filter size
add feature detector with 16 input channels, 32 output channels, 3x3 filter size and padding same
```

### Pooling Layers

```
# Add a pooling layer
add downsampling using [method] method with size [size]x[size]

# Examples
add downsampling using max method with size 2x2
add downsampling using average method with size 3x3
```

### Recurrent Layers

```
# Add a recurrent layer
add memory cell [type] with [inputs] inputs and [hidden_size] hidden size

# Examples
add memory cell lstm with 50 inputs and 100 hidden size
add memory cell gru with 64 inputs and 128 hidden size
```

### Activation Functions

```
# Apply an activation function
apply [function_name] activation

# Examples
apply relu activation
apply sigmoid activation
apply tanh activation
apply leaky_relu activation
```

### Regularization Layers

```
# Add dropout
add dropout with rate [rate]

# Add batch normalization
add batch normalization with [features] features

# Examples
add dropout with rate 0.5
add batch normalization with 128 features
```

### Special Operations

```
# Flatten a tensor (e.g., after convolutional layers)
flatten features

# Reshape a tensor
reshape features to [dimensions]

# Examples
flatten features
reshape features to 1x28x28
```

## Training Process

### Basic Training

```
# Train a model on a dataset
train [model_name] on [data_stream]:
    [training parameters...]
    
# Examples
train digit_recognizer on mnist_train_stream:
    measure error using cross_entropy
    improve using adam with learning rate 0.001
    repeat for 10 learning cycles
```

### Error Measures (Loss Functions)

```
# Specify a loss function
measure error using [error_function]

# Examples
measure error using mean_squared_error
measure error using cross_entropy
measure error using binary_cross_entropy
```

### Improvement Strategies (Optimizers)

```
# Specify an optimizer
improve using [strategy] with learning rate [rate]

# With additional parameters
improve using [strategy] with learning rate [rate] and [parameter] [value]

# Examples
improve using gradient_descent with learning rate 0.01
improve using adam with learning rate 0.001
improve using adam with learning rate 0.001 and weight decay 0.0001
```

### Learning Cycles (Epochs)

```
# Specify the number of training epochs
repeat for [cycles] learning cycles

# Examples
repeat for 10 learning cycles
```

### Advanced Training Options

```
# Learning rate scheduling
reduce learning rate by factor [factor] when plateau in validation loss for [patience] cycles

# Early stopping
stop early if no improvement for [patience] cycles

# Gradient clipping
clip gradients to maximum [max_norm]

# Progress reporting
print progress every [interval] batches

# Examples
reduce learning rate by factor 0.1 when plateau in validation loss for 5 cycles
stop early if no improvement for 10 cycles
clip gradients to maximum 5.0
print progress every 100 batches
```

## Evaluation and Inference

### Model Evaluation

```
# Evaluate a model on a test dataset
evaluate [model_name] on [test_stream]:
    [evaluation parameters...]
    
# Examples
evaluate digit_recognizer on mnist_test_stream:
    measure accuracy
    report results
```

### Performance Metrics

```
# Specify which metrics to calculate
measure [metrics...]

# Examples
measure accuracy
measure accuracy, precision, and recall
measure accuracy, f1 score
```

### Making Predictions

```
# Use a model for inference
use [model_name] to process [input_data]

# Get predictions from a model
get predictions from [model_name] for [input_data]

# Examples
use digit_recognizer to process test_image
get predictions from classifier for batch_data
```

## Model Persistence

### Saving Models

```
# Save a model to a file
save [model_name] to [filepath]

# Save only model parameters
save [model_name] parameters to [filepath]

# Examples
save digit_recognizer to "models/digit_recognizer.pth"
save classifier parameters to "checkpoints/weights.pth"
```

### Loading Models

```
# Load a model from a file
load [model_name] from [filepath]

# Load parameters into a model
load parameters into [model_name] from [filepath]

# Examples
load digit_recognizer from "models/digit_recognizer.pth"
load parameters into classifier from "checkpoints/weights.pth"
```

## Advanced Model Architectures

### Residual Connections

```
# Add a residual block
add residual block with [in_channels] input channels, [out_channels] output channels

# Examples
add residual block with 64 input channels, 64 output channels
```

### Attention Mechanisms

```
# Add an attention mechanism
add attention mechanism with [query_dim] query dimension and [key_dim] key dimension

# Examples
add attention mechanism with 512 query dimension and 512 key dimension
```

### Transfer Learning

```
# Load a pretrained model
load pretrained [model_type] model

# Freeze layers
freeze all layers except [layer_names]

# Replace layers
replace final layer with transformation stage fully_connected with [inputs] inputs and [outputs] outputs

# Examples
load pretrained resnet18 model
freeze all layers except final layer
replace final layer with transformation stage fully_connected with 512 inputs and 10 outputs
```

## Comments and Documentation

```
# Single line comments start with a hash symbol

# Multiple line comments
# can be written like
# this
```

## Next Steps

Now that you understand Nova's syntax, proceed to the [Translation Process](translation-process.md) guide to learn how Nova code is translated to PyTorch.