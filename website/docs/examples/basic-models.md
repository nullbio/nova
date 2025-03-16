# Basic Models

This page demonstrates how to create and train simple neural network models using Nova. These examples are ideal for beginners getting started with Nova and deep learning.

## Simple Neural Network

Let's start with a basic fully-connected neural network for classifying MNIST digits.

=== "Nova"
    ```
    # Load MNIST dataset
    load data collection mnist from torchvision.datasets with:
        apply normalization with mean 0.1307 and deviation 0.3081
        convert to feature grid

    # Prepare data streams
    prepare data stream train_stream from mnist.train with batch size 64 and shuffle enabled
    prepare data stream test_stream from mnist.test with batch size 1000

    # Create a neural network for digit classification
    create processing pipeline digit_classifier:
        add transformation stage fully_connected with 784 inputs and 256 outputs
        apply relu activation
        add transformation stage fully_connected with 256 inputs and 128 outputs
        apply relu activation
        add transformation stage fully_connected with 128 inputs and 10 outputs

    # Train the model
    train digit_classifier on train_stream:
        measure error using cross_entropy
        improve using adam with learning rate 0.001
        repeat for 5 learning cycles
        print progress every 100 batches

    # Evaluate the model
    evaluate digit_classifier on test_stream:
        measure accuracy
        report results

    # Save the model
    save digit_classifier to "models/digit_classifier.pth"
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Create model
    class DigitClassifier(nn.Module):
        def __init__(self):
            super(DigitClassifier, self).__init__()
            self.fc1 = nn.Linear(28*28, 256)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = x.view(-1, 28*28)  # Flatten the input
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    model = DigitClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training function
    def train(model, loader, criterion, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}'
                    f' ({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Evaluation function
    def test(model, loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
            f'Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.0f}%)\n')
        
        return accuracy

    # Train model
    for epoch in range(1, 6):
        train(model, train_loader, criterion, optimizer, epoch)
        test(model, test_loader, criterion)

    # Save model
    torch.save(model.state_dict(), "models/digit_classifier.pth")
    ```

### Key Components Explained

1. **Data Loading**: Load the MNIST dataset with normalization
2. **Data Preparation**: Create data loaders for training and testing
3. **Model Definition**: Create a neural network with three fully-connected layers
4. **Training Setup**: Define the loss function and optimizer
5. **Training Loop**: Train the model for 5 epochs
6. **Evaluation**: Evaluate the model on the test set
7. **Model Saving**: Save the trained model

## Logistic Regression

Logistic regression is one of the simplest models for binary classification. Here's how to implement it in Nova:

=== "Nova"
    ```
    # Create binary classification dataset
    load data collection binary_dataset from sklearn.datasets.make_classification with:
        100 samples
        2 features
        2 classes
        
    # Split dataset
    split binary_dataset into 70% training and 30% testing

    # Create a logistic regression model
    create processing pipeline logistic_model:
        add transformation stage fully_connected with 2 inputs and 1 outputs
        apply sigmoid activation

    # Train the model
    train logistic_model on train_stream:
        measure error using binary_cross_entropy
        improve using gradient_descent with learning rate 0.05
        repeat for 100 learning cycles

    # Evaluate the model
    evaluate logistic_model on test_stream:
        measure accuracy
        report results
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader

    # Create dataset
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Create model
    class LogisticModel(nn.Module):
        def __init__(self):
            super(LogisticModel, self).__init__()
            self.linear = nn.Linear(2, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            x = self.linear(x)
            x = self.sigmoid(x)
            return x

    model = LogisticModel()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # Training loop
    for epoch in range(100):
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/100, Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
    ```

### Key Components Explained

1. **Dataset Creation**: Generate a synthetic binary classification dataset
2. **Data Preparation**: Split the dataset and create data loaders
3. **Model Definition**: Create a logistic regression model (a single neuron with sigmoid activation)
4. **Training Setup**: Define the binary cross-entropy loss and gradient descent optimizer
5. **Training Loop**: Train the model for 100 epochs
6. **Evaluation**: Evaluate the model accuracy on the test set

## Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron (MLP) is a fully-connected neural network with multiple hidden layers.

=== "Nova"
    ```
    # Load iris dataset
    load data collection iris from sklearn.datasets.load_iris
    split iris into 80% training and 20% testing

    # Prepare data streams
    prepare data stream train_stream from iris.train with batch size 16 and shuffle enabled
    prepare data stream test_stream from iris.test with batch size 32

    # Create a multi-layer perceptron for classification
    create processing pipeline iris_classifier:
        add transformation stage fully_connected with 4 inputs and 64 outputs
        apply relu activation
        add dropout with rate 0.2
        add transformation stage fully_connected with 64 inputs and 32 outputs
        apply relu activation
        add dropout with rate 0.2
        add transformation stage fully_connected with 32 inputs and 3 outputs
        apply softmax activation

    # Train the model
    train iris_classifier on train_stream:
        measure error using cross_entropy
        improve using adam with learning rate 0.001 and weight decay 0.0001
        repeat for 100 learning cycles
        
    # Evaluate the model
    evaluate iris_classifier on test_stream:
        measure accuracy, precision, recall, and f1 score
        report results
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score
    from torch.utils.data import TensorDataset, DataLoader

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create model
    class IrisClassifier(nn.Module):
        def __init__(self):
            super(IrisClassifier, self).__init__()
            self.fc1 = nn.Linear(4, 64)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(32, 3)
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            x = self.softmax(x)
            return x

    model = IrisClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Training loop
    for epoch in range(100):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/100, Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
    ```

### Key Components Explained

1. **Dataset Loading**: Load the Iris dataset for multi-class classification
2. **Data Preparation**: Split the dataset and create data loaders
3. **Model Definition**: Create an MLP with two hidden layers, ReLU activations, and dropout regularization
4. **Training Setup**: Define the cross-entropy loss and Adam optimizer with weight decay
5. **Training Loop**: Train the model for 100 epochs
6. **Evaluation**: Evaluate the model using multiple metrics (accuracy, precision, recall, F1 score)

## Next Steps

Now that you've learned about basic models, you can explore more complex architectures:

- Computer Vision: Creating CNNs for image tasks
- NLP: Building RNNs and transformers for text
- Advanced Topics: Exploring advanced techniques like transfer learning and generative models