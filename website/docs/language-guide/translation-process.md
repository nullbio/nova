# Translation Process

This guide explains how Nova code is translated into executable PyTorch code. Understanding this process helps you bridge the gap between Nova's intuitive syntax and PyTorch's implementation details.

## Overview of the Translation Process

The Nova interpreter follows these steps to translate Nova code to PyTorch:

1. **Parsing**: Break down Nova code into components and operations
2. **Analysis**: Identify the relationships between components
3. **Code Generation**: Create equivalent PyTorch code
4. **Explanation**: Generate explanations of the translation

## Step 1: Parsing

During parsing, the Nova interpreter identifies key constructs in your code:

- Model definitions
- Layer specifications
- Training parameters
- Data operations

For example, given this Nova code:

```
create processing pipeline simple_model:
    add transformation stage fully_connected with 10 inputs and 5 outputs
    apply relu activation
```

The parser identifies:
- A model named "simple_model"
- A fully connected layer with 10 inputs and 5 outputs
- A ReLU activation function

## Step 2: Analysis

During analysis, the interpreter:

- Resolves dependencies between components
- Validates the coherence of the model architecture
- Determines the proper sequence of operations
- Identifies implied operations that need to be made explicit in PyTorch

For instance, it validates that the activation function follows a layer that produces compatible outputs.

## Step 3: Code Generation

The code generation process creates equivalent PyTorch code for each Nova component:

### Model Generation

Nova models ("processing pipelines") are translated to PyTorch `nn.Module` classes:

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
create processing pipeline simple_model:
    add transformation stage fully_connected with 10 inputs and 5 outputs
    apply relu activation
    add transformation stage fully_connected with 5 inputs and 1 outputs
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleModel()
```

</div>
</div>
</div>

The translation process:

1. Creates a class that inherits from `nn.Module`
2. Initializes layers in `__init__`
3. Defines data flow in the `forward` method
4. Instantiates the model

### Training Code Generation

Nova training instructions translate to PyTorch training loops:

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
train simple_model on data_stream:
    measure error using mean_squared_error
    improve using adam with learning rate 0.001
    repeat for 5 learning cycles
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    for inputs, targets in data_stream:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
```

</div>
</div>
</div>

The translation process:

1. Sets up the loss function based on the specified error measure
2. Creates an optimizer with the specified parameters
3. Generates a training loop with the specified number of epochs
4. Adds the necessary gradient calculation and parameter update steps

### Data Processing Generation

Nova data operations translate to PyTorch data handling code:

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
load data collection mnist from torchvision.datasets with:
    apply normalization with mean 0.1307 and deviation 0.3081

prepare data stream from mnist with batch size 32 and shuffle enabled
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_dataset = datasets.MNIST('./data', download=True, transform=transform)
data_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)
```

</div>
</div>
</div>

## Step 4: Explanation Generation

The Nova interpreter can generate explanations of the translation to help you understand the mapping:

```python
from nova import NovaInterpreter

interpreter = NovaInterpreter()
pytorch_code = interpreter.translate(nova_code)
explanation = interpreter.explain_translation(nova_code, pytorch_code)

print(explanation)
```

Example explanation output:

```
Translation Explanation:
1. Created a PyTorch model class called 'SimpleModel' that inherits from nn.Module
2. Added a fully connected layer (nn.Linear) with 10 inputs and 5 outputs
3. Added a ReLU activation function
4. Added another fully connected layer with 5 inputs and 1 output
5. Defined the forward method to connect these layers in sequence
6. Created loss function (MSELoss) based on the 'mean_squared_error' error measure
7. Created Adam optimizer with learning rate 0.001
8. Generated a training loop with 5 epochs
```

## Translation Examples

### CNN Translation

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
create image processing pipeline cnn_model:
    add feature detector with 1 input channels, 32 output channels and 3x3 filter size
    apply relu activation
    add downsampling using max method with size 2x2
    flatten features
    add transformation stage fully_connected with 6272 inputs and 10 outputs
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(6272, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

model = CnnModel()
```

</div>
</div>
</div>

### RNN Translation

<div class="code-comparison">
<div>
<div class="title">Nova</div>
<div class="nova-code">

```
create sequence processing pipeline rnn_model:
    add memory cell lstm with 50 inputs and 100 hidden size
    add transformation stage fully_connected with 100 inputs and 10 outputs
```

</div>
</div>
<div>
<div class="title">PyTorch</div>
<div class="pytorch-code">

```python
class RnnModel(nn.Module):
    def __init__(self):
        super(RnnModel, self).__init__()
        self.lstm = nn.LSTM(input_size=50, hidden_size=100, batch_first=True)
        self.fc = nn.Linear(100, 10)
        
    def forward(self, x):
        # LSTM returns output and hidden state
        output, _ = self.lstm(x)
        # Use the output from the last time step
        x = output[:, -1, :]
        x = self.fc(x)
        return x

model = RnnModel()
```

</div>
</div>
</div>

## Advanced Translation Considerations

### Handling Complex Architectures

For more complex architectures, the Nova interpreter:

1. **Analyzes model connectivity**: Determines how components connect to each other
2. **Resolves dimension conflicts**: Inserts necessary reshaping operations
3. **Generates appropriate forward logic**: Creates the proper sequence in the forward method

### Translating Custom Behavior

For cases where the direct mapping is not obvious, the interpreter makes intelligent decisions:

1. **Implied operations**: Adds necessary operations that are implied but not explicit
2. **Default values**: Provides reasonable defaults for optional parameters
3. **Best practices**: Follows PyTorch best practices in the generated code

## Debugging Translation Issues

If you encounter issues with translation:

1. **Check the explanation**: Review the explanation to understand the translator's decisions
2. **Examine the generated code**: Look for any unexpected elements
3. **Simplify the model**: Try a simpler version to isolate the problem
4. **Verify dimensions**: Ensure that layer dimensions are compatible

## Next Steps

Now that you understand how Nova code is translated to PyTorch, explore the [Examples](../examples/basic-models.md) section to see Nova in action with practical applications.