# Testing Code Tabs

This page demonstrates the tabbed code comparison feature.

## Example with Tabs

=== "Nova"
    ```python
    # Load MNIST dataset
    load data collection mnist from torchvision.datasets with:
        apply normalization with mean 0.1307 and deviation 0.3081
        convert to feature grid

    # Create a neural network for digit classification
    create processing pipeline digit_classifier:
        add transformation stage fully_connected with 784 inputs and 256 outputs
        apply relu activation
        add transformation stage fully_connected with 256 inputs and 128 outputs
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_dataset = datasets.MNIST('./data', download=True, transform=transform)
    
    # Create model
    class DigitClassifier(nn.Module):
        def __init__(self):
            super(DigitClassifier, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
    ```

## Another Example

=== "Nova"
    ```python
    # This is Nova code
    create processing pipeline simple_model:
        add transformation stage fully_connected with 10 inputs and 5 outputs
        apply relu activation
    ```

=== "PyTorch"
    ```python
    # This is PyTorch code
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.relu = nn.ReLU()
    ```