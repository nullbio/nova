# Quick Start Guide

This guide will help you get started with Nova quickly. You'll learn how to create a simple neural network, translate it to PyTorch code, and run it.

## Basic Usage

The simplest way to use Nova is through its Python API:

```python
from nova import NovaInterpreter

# Create an interpreter instance
interpreter = NovaInterpreter()

# Define a model using Nova language
nova_code = """
create processing pipeline simple_model:
    add transformation stage fully_connected with 10 inputs and 5 outputs
    apply relu activation
    add transformation stage fully_connected with 5 inputs and 1 outputs
    apply sigmoid activation
"""

# Translate Nova code to PyTorch code
pytorch_code = interpreter.translate(nova_code)

# Print the generated code
print(pytorch_code)

# Execute the generated code
exec(pytorch_code)

# Now you can use the model
print(model)
```

## Complete Example: MNIST Digit Classification

Let's create a simple digit classifier for the MNIST dataset using Nova:

```python
from nova import NovaInterpreter
import torch

# Create an interpreter instance
interpreter = NovaInterpreter()

# Define a complete model with data loading and training
nova_code = """
# Load MNIST dataset
load data collection mnist from torchvision.datasets with:
    apply normalization with mean 0.1307 and deviation 0.3081
    convert to feature grid

# Prepare data streams
prepare data stream train_stream from mnist.train with batch size 64 and shuffle enabled
prepare data stream test_stream from mnist.test with batch size 1000

# Create a neural network for digit classification
create processing pipeline digit_classifier:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs

# Train the model
train digit_classifier on train_stream:
    measure error using cross_entropy
    improve using adam with learning rate 0.001
    repeat for 3 learning cycles

# Evaluate the model
evaluate digit_classifier on test_stream:
    measure accuracy
    report results
"""

# Translate and execute
pytorch_code = interpreter.translate(nova_code)
exec(pytorch_code)
```

## Step-by-Step Guide

### 1. Creating a Model

In Nova, models are called "processing pipelines" and are defined with a clear, natural syntax:

```
create processing pipeline model_name:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
```

This creates a simple neural network with two fully connected layers and ReLU activation.

### 2. Loading Data

Nova provides intuitive syntax for loading and preparing data:

```
load data collection mnist from torchvision.datasets with:
    apply normalization with mean 0.1307 and deviation 0.3081

prepare data stream train_stream from mnist.train with batch size 64 and shuffle enabled
```

### 3. Training the Model

Training is expressed in terms of "error measures" (loss functions) and "improvement strategies" (optimizers):

```
train model_name on train_stream:
    measure error using cross_entropy
    improve using adam with learning rate 0.001
    repeat for 5 learning cycles
```

### 4. Evaluating the Model

Evaluation is similarly expressed in intuitive terms:

```
evaluate model_name on test_stream:
    measure accuracy
    report results
```

## Understanding the Translation

Nova translates natural language instructions into proper PyTorch code. You can examine the translation with:

```python
# Get an explanation of the translation
explanation = interpreter.explain_translation(nova_code, pytorch_code)
print(explanation)
```

This helps you understand how your Nova code maps to PyTorch constructs, aiding the learning process.

## Common Patterns

### Convolutional Neural Networks

```
create image processing pipeline cnn_model:
    add feature detector with 1 input channels, 32 output channels and 3x3 filter size
    apply relu activation
    add downsampling using max method with size 2x2
    add feature detector with 32 input channels, 64 output channels and 3x3 filter size
    apply relu activation
    add downsampling using max method with size 2x2
    flatten features
    add transformation stage fully_connected with 1600 inputs and 10 outputs
```

### Recurrent Neural Networks

```
create sequence processing pipeline rnn_model:
    add memory cell lstm with 50 inputs and 100 hidden size
    add transformation stage fully_connected with 100 inputs and 10 outputs
```

## Next Steps

Now that you've created your first model with Nova, you can explore more complex architectures and features:

- [Language Guide](../language-guide/overview.md): Learn the full Nova syntax
- [Examples](../examples/basic-models.md): Explore more practical examples
- [Core Concepts](../language-guide/core-concepts.md): Understand the concepts behind Nova