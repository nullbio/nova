# Nova: Natural Language Interface for Deep Learning

<p class="hero-subtitle">Bridging the gap between intuitive language and PyTorch code</p>

<div class="grid-container">
  <div class="feature-card">
    <h3>🔤 Natural Language</h3>
    <p>Express machine learning concepts in intuitive, human-readable language</p>
  </div>
  <div class="feature-card">
    <h3>🔄 Pipeline Approach</h3>
    <p>Conceptualize neural networks as familiar data transformation pipelines</p>
  </div>
  <div class="feature-card">
    <h3>🧠 Education-Focused</h3>
    <p>Learn PyTorch concepts through accessible terminology and examples</p>
  </div>
  <div class="feature-card">
    <h3>⚡ Productivity</h3>
    <p>Rapidly prototype machine learning models with minimal boilerplate</p>
  </div>
</div>

## What is Nova?

Nova is a natural language interface that transforms how we approach deep learning. It allows developers, researchers, and students to express machine learning concepts using intuitive language, which gets translated into executable PyTorch code.

Instead of wrestling with complex PyTorch syntax and concepts, Nova lets you focus on what your model should do, not how to code it.

```
# Nova code
create processing pipeline digit_recognizer:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
```

translates to:

```python
# PyTorch code
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
```

## Why Nova?

<div class="grid-container">
  <div class="feature-card">
    <h3>🚫 No Jargon</h3>
    <p>Replace complex machine learning terminology with intuitive concepts that make sense to everyone</p>
  </div>
  <div class="feature-card">
    <h3>🔍 Transparency</h3>
    <p>See exactly how your natural language translates to PyTorch code</p>
  </div>
  <div class="feature-card">
    <h3>🌱 Gentle Learning Curve</h3>
    <p>Gradually learn PyTorch concepts through natural language and translations</p>
  </div>
  <div class="feature-card">
    <h3>🧩 Familiar Patterns</h3>
    <p>Use the pipeline metaphor familiar to most programmers</p>
  </div>
</div>

Machine learning and deep learning have steep learning curves due to:

1. **Complex Terminology**: Terms like tensors, gradients, and backpropagation can be intimidating
2. **Mathematical Foundations**: Understanding the math can be challenging for many developers
3. **Framework-Specific Patterns**: Learning PyTorch's specific implementation patterns takes time

Nova addresses these challenges by providing an intuitive bridge between human understanding and technical implementation.

## Quick Example

=== "Nova Code"
    ```
    # Load data
    load data collection mnist from torchvision.datasets with:
        apply normalization with mean 0.1307 and deviation 0.3081
        convert to feature grid

    # Create model
    create processing pipeline digit_classifier:
        add transformation stage fully_connected with 784 inputs and 256 outputs
        apply relu activation
        add transformation stage fully_connected with 256 inputs and 10 outputs

    # Train model
    train digit_classifier on mnist_dataloader:
        measure error using cross_entropy
        improve using adam with learning rate 0.001
        repeat for 5 learning cycles
    ```

=== "PyTorch Code"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_dataset = datasets.MNIST('./data', download=True, transform=transform)
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

    # Create model
    class DigitClassifier(nn.Module):
        def __init__(self):
            super(DigitClassifier, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 10)
        
        def forward(self, x):
            x = x.view(-1, 784)  # Flatten the input
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = DigitClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    for epoch in range(5):
        for data, target in mnist_dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    ```

## Get Started

Ready to try Nova? Follow our getting started guide to begin using natural language for deep learning.

<div class="grid-container">
  <a href="getting-started/installation/" class="feature-card">
    <h3>📥 Installation</h3>
    <p>Install Nova and set up your environment</p>
  </a>
  <a href="getting-started/quick-start/" class="feature-card">
    <h3>🚀 Quick Start</h3>
    <p>Create your first model using Nova</p>
  </a>
  <a href="language-guide/overview/" class="feature-card">
    <h3>📚 Language Guide</h3>
    <p>Learn the Nova language and syntax</p>
  </a>
  <a href="examples/basic-models/" class="feature-card">
    <h3>💡 Examples</h3>
    <p>Explore practical examples using Nova</p>
  </a>
</div>