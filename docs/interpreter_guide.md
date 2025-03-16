# Nova Interpreter Guide

The Nova Interpreter is an AI-powered system that translates natural language Nova code into executable PyTorch code. This guide explains how to use the interpreter effectively.

## How the Interpreter Works

The Nova Interpreter:
1. Parses natural language Nova code
2. Identifies key components and operations
3. Maps them to corresponding PyTorch constructs
4. Generates executable PyTorch code
5. Provides explanations of the translation process

## Using the Nova Interpreter

### Basic Usage

1. Write your Nova code using the syntax defined in the [Language Guide](language_guide.md)
2. Submit the code to the Nova Interpreter
3. Review the generated PyTorch code
4. Execute the PyTorch code in your environment

### Interactive Mode

In interactive mode, you can:
1. Write Nova code incrementally
2. Ask questions about the translation
3. Request explanations of specific parts
4. Refine your code based on feedback

### Example Session

**User Input (Nova):**
```
create processing pipeline image_classifier:
    add feature detector with 3 input channels, 16 output channels and 3x3 filter size
    apply relu activation
    add downsampling using max method with size 2x2
    flatten features
    add transformation stage fully_connected with 1024 inputs and 10 outputs
```

**Interpreter Output (PyTorch):**
```python
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(1024, 10)  # Assumes input size gives 1024 features after convolution and pooling
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

model = ImageClassifier()
```

**Explanation:**
```
Translation Explanation:
1. Created a new PyTorch model class called 'ImageClassifier' that inherits from nn.Module
2. Implemented a convolutional layer with 3 input channels, 16 output channels, and 3x3 kernel
3. Added ReLU activation function
4. Added max pooling layer with 2x2 kernel
5. Added a flattening operation in the forward method
6. Implemented a fully connected layer with 1024 inputs and 10 outputs
7. Instantiated the model

Note: The 1024 input size for the fully connected layer assumes the input dimensions will result in this size after convolution and pooling. You may need to adjust this value based on your input image dimensions.
```

## Advanced Features

### Asking Questions

You can ask the interpreter questions about the translation:

**User Question:**
```
Why did you add padding=1 to the convolutional layer?
```

**Interpreter Response:**
```
I added padding=1 to the convolutional layer to preserve the spatial dimensions of the input. Without padding, a 3x3 convolution would reduce each dimension by 2 pixels (one on each side). 

By adding padding=1, the output feature map will have the same height and width as the input, which is often desirable to maintain spatial information, especially in the early layers of a convolutional network.

If you prefer not to use padding, you can specify this in Nova:

add feature detector with 3 input channels, 16 output channels and 3x3 filter size with no padding

Which would translate to:
nn.Conv2d(3, 16, kernel_size=3, padding=0)
```

### Requesting Modifications

You can ask the interpreter to modify the generated code:

**User Request:**
```
Modify the model to add batch normalization after the convolutional layer
```

**Interpreter Update:**
```python
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Added batch normalization
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Applied batch normalization
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ImageClassifier()
```

## Best Practices

### 1. Be Specific with Dimensions

Clearly specify input and output dimensions to ensure accurate translation:

```
# Good - clear dimensions
add transformation stage fully_connected with 784 inputs and 128 outputs

# Less clear - may cause issues
add transformation stage fully_connected
```

### 2. Use Consistent Terminology

Stick to Nova's terminology for clarity:

```
# Good - uses Nova terminology
add feature detector with 3 input channels, 16 output channels

# Less clear - mixes terminology
add convolution layer with 3 in channels, 16 out channels
```

### 3. Structure Your Code Logically

Organize your code in a logical flow:

```
# Good structure
create processing pipeline classifier:
    # Define all layers
    ...

train classifier on data_stream:
    # Define training process
    ...

evaluate classifier on test_stream:
    # Define evaluation
    ...
```

### 4. Ask for Explanations

When in doubt, ask the interpreter to explain its translation:

```
Please explain why you chose this implementation for the memory cell
```

## Troubleshooting

### Common Issues and Solutions

1. **Unclear Dimensions**
   - Issue: `flatten features` operation generates incorrect dimensions
   - Solution: Specify the expected output dimension or input image size

2. **Missing Dependencies**
   - Issue: Generated code imports missing dependencies
   - Solution: Ask the interpreter to include all necessary imports

3. **Complex Operations**
   - Issue: Advanced operations may not translate directly
   - Solution: Break complex operations into simpler steps or ask for guidance

4. **Performance Concerns**
   - Issue: Generated code may not be optimized
   - Solution: Ask the interpreter for optimization suggestions

## Extending Nova

The Nova language and interpreter can be extended to support:

1. **Custom Components**: Define your own reusable components
2. **Domain-Specific Extensions**: Add terminology for specific domains like NLP or computer vision
3. **Pipeline Integration**: Integrate with existing data processing pipelines
4. **Alternative Backends**: Support for other frameworks beyond PyTorch

To suggest extensions or improvements, contribute to the Nova repository.