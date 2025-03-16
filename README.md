# Nova: Natural Language Interface for Deep Learning

Nova is a natural language interpreter that bridges the gap between intuitive human understanding and PyTorch machine learning operations. It allows programmers and non-experts to express machine learning concepts and operations using natural language, which is then translated into functional PyTorch code.

## Quick Start

```python
# Nova code
create processing pipeline digit_recognizer:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs

# Translated PyTorch
import torch
import torch.nn as nn

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = DigitRecognizer()
```

## Why Nova?

Machine learning and deep learning come with steep learning curves due to complex terminology, mathematical concepts, and programming patterns. Nova addresses these challenges by:

1. **Simplifying Terminology**: Replacing jargon with intuitive terms
2. **Conceptualizing Models as Pipelines**: Using the familiar data transformation pipeline paradigm
3. **Abstracting Implementation Details**: Focusing on intent rather than syntax
4. **Providing Educational Bridges**: Connecting intuitive understanding with technical implementation

## Documentation

- [Language Guide](docs/language_guide.md) - Learn Nova vocabulary and syntax
- [Core Concepts](docs/core_concepts.md) - Essential machine learning concepts in Nova terms
- [Interpreter Guide](docs/interpreter_guide.md) - How to use the Nova interpreter
- [Examples](examples/README.md) - See Nova in action with code examples

## Features

- **Intuitive Terminology**: Makes ML concepts more approachable
- **Pipeline-based Approach**: Aligns with familiar programming paradigms
- **Educational Value**: Bridges conceptual understanding and implementation
- **Flexibility**: Can express a wide range of deep learning operations
- **Extensibility**: Can be extended for specific domains or use cases

## Requirements

- Python 3.8+
- PyTorch 2.0+

## Installation

```bash
# Coming soon
pip install nova-dl
```

## Usage

```python
# Coming soon
from nova import interpreter

# Create an interpreter instance
nova = interpreter.NovaInterpreter()

# Process Nova code
pytorch_code = nova.translate("""
create processing pipeline digit_recognizer:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
""")

# The generated pytorch_code would contain:
"""
import torch
import torch.nn as nn

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = DigitRecognizer()
"""

# Execute the generated code
exec(pytorch_code)

# Now 'model' is available in the namespace
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
