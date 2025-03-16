# Computer Vision Examples

This page demonstrates how to use Nova for computer vision tasks using convolutional neural networks (CNNs) and related architectures.

## Basic CNN for MNIST

Let's start with a simple CNN for digit classification on the MNIST dataset.

=== "Nova"
    ```
    # Load MNIST dataset
    load data collection mnist from torchvision.datasets with:
        apply normalization with mean 0.1307 and deviation 0.3081
        convert to feature grid

    # Prepare data streams
    prepare data stream train_stream from mnist.train with batch size 64 and shuffle enabled
    prepare data stream test_stream from mnist.test with batch size 1000

    # Create a CNN for digit classification
    create image processing pipeline mnist_cnn:
        # First convolutional block
        add feature detector with 1 input channels, 32 output channels and 3x3 filter size
        apply relu activation
        add feature detector with 32 input channels, 32 output channels and 3x3 filter size
        apply relu activation
        add downsampling using max method with size 2x2
        
        # Second convolutional block
        add feature detector with 32 input channels, 64 output channels and 3x3 filter size
        apply relu activation
        add feature detector with 64 input channels, 64 output channels and 3x3 filter size
        apply relu activation
        add downsampling using max method with size 2x2
        
        # Fully connected layers
        flatten features
        add transformation stage fully_connected with 1600 inputs and 128 outputs
        apply relu activation
        add dropout with rate 0.5
        add transformation stage fully_connected with 128 inputs and 10 outputs

    # Train the model
    train mnist_cnn on train_stream:
        measure error using cross_entropy
        improve using adam with learning rate 0.001
        repeat for 10 learning cycles
        print progress every 100 batches

    # Evaluate the model
    evaluate mnist_cnn on test_stream:
        measure accuracy
        report results
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
    class MnistCNN(nn.Module):
        def __init__(self):
            super(MnistCNN, self).__init__()
            # First convolutional block
            self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.relu1_1 = nn.ReLU()
            self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.relu1_2 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Second convolutional block
            self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.relu2_1 = nn.ReLU()
            self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.relu2_2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Fully connected layers
            self.fc1 = nn.Linear(1600, 128)  # 7x7x64 = 3136
            self.relu3 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            # First convolutional block
            x = self.relu1_1(self.conv1_1(x))
            x = self.relu1_2(self.conv1_2(x))
            x = self.pool1(x)
            
            # Second convolutional block
            x = self.relu2_1(self.conv2_1(x))
            x = self.relu2_2(self.conv2_2(x))
            x = self.pool2(x)
            
            # Flatten and fully connected layers
            x = x.view(x.size(0), -1)  # Flatten
            x = self.relu3(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x

    model = MnistCNN().to(device)
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
    def test(model, loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(loader.dataset)
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        return accuracy

    # Train and evaluate model
    for epoch in range(1, 11):
        train(model, train_loader, criterion, optimizer, epoch)
        accuracy = test(model, test_loader)
    ```

### Key Components Explained

1. **Convolutional Layers**: Used to extract spatial features from images
2. **Pooling Layers**: Reduce the spatial dimensions while preserving important features
3. **Fully Connected Layers**: Connect all extracted features for final classification
4. **Dropout**: Prevents overfitting by randomly deactivating neurons during training

## CNN for CIFAR-10

Let's build a more complex CNN for the CIFAR-10 dataset, which consists of 10 classes of color images.

=== "Nova"
    ```
    # Load CIFAR-10 dataset
    load data collection cifar10 from torchvision.datasets with:
        apply image transformations:
            resize to 32x32
            random horizontal flip with probability 0.5
            random crop to 32x32 with padding 4
            convert to tensor
            normalize with mean [0.4914, 0.4822, 0.4465] and deviation [0.2470, 0.2435, 0.2616]

    # Prepare data streams
    prepare data stream train_stream from cifar10.train with batch size 128 and shuffle enabled
    prepare data stream test_stream from cifar10.test with batch size 1000

    # Create a CNN for CIFAR-10 classification
    create image processing pipeline cifar_cnn:
        # First convolutional block
        add feature detector with 3 input channels, 64 output channels and 3x3 filter size and padding same
        apply batch normalization with 64 features
        apply relu activation
        add feature detector with 64 input channels, 64 output channels and 3x3 filter size and padding same
        apply batch normalization with 64 features
        apply relu activation
        add downsampling using max method with size 2x2
        add dropout with rate 0.25
        
        # Second convolutional block
        add feature detector with 64 input channels, 128 output channels and 3x3 filter size and padding same
        apply batch normalization with 128 features
        apply relu activation
        add feature detector with 128 input channels, 128 output channels and 3x3 filter size and padding same
        apply batch normalization with 128 features
        apply relu activation
        add downsampling using max method with size 2x2
        add dropout with rate 0.25
        
        # Third convolutional block
        add feature detector with 128 input channels, 256 output channels and 3x3 filter size and padding same
        apply batch normalization with 256 features
        apply relu activation
        add feature detector with 256 input channels, 256 output channels and 3x3 filter size and padding same
        apply batch normalization with 256 features
        apply relu activation
        add downsampling using max method with size 2x2
        add dropout with rate 0.25
        
        # Fully connected layers
        flatten features
        add transformation stage fully_connected with 4096 inputs and 512 outputs
        apply batch normalization with 512 features
        apply relu activation
        add dropout with rate 0.5
        add transformation stage fully_connected with 512 inputs and 10 outputs

    # Train the model
    train cifar_cnn on train_stream:
        measure error using cross_entropy
        improve using adam with learning rate 0.001 and weight decay 0.0001
        reduce learning rate by factor 0.1 when plateau in validation loss for 5 cycles
        repeat for 50 learning cycles
        stop early if no improvement for 10 cycles
        print progress every 100 batches
        
    # Evaluate the model
    evaluate cifar_cnn on test_stream:
        measure accuracy
        report results
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Define the model
    class CifarCNN(nn.Module):
        def __init__(self):
            super(CifarCNN, self).__init__()
            # First convolutional block
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1_1 = nn.BatchNorm2d(64)
            self.relu1_1 = nn.ReLU()
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn1_2 = nn.BatchNorm2d(64)
            self.relu1_2 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout1 = nn.Dropout(0.25)
            
            # Second convolutional block
            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2_1 = nn.BatchNorm2d(128)
            self.relu2_1 = nn.ReLU()
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn2_2 = nn.BatchNorm2d(128)
            self.relu2_2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout2 = nn.Dropout(0.25)
            
            # Third convolutional block
            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3_1 = nn.BatchNorm2d(256)
            self.relu3_1 = nn.ReLU()
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn3_2 = nn.BatchNorm2d(256)
            self.relu3_2 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout3 = nn.Dropout(0.25)
            
            # Fully connected layers
            self.fc1 = nn.Linear(4096, 512)
            self.bn4 = nn.BatchNorm1d(512)
            self.relu4 = nn.ReLU()
            self.dropout4 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 10)
        
        def forward(self, x):
            # First block
            x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
            x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
            x = self.pool1(x)
            x = self.dropout1(x)
            
            # Second block
            x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
            x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
            x = self.pool2(x)
            x = self.dropout2(x)
            
            # Third block
            x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
            x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
            x = self.pool3(x)
            x = self.dropout3(x)
            
            # Flatten and fully connected layers
            x = x.view(x.size(0), -1)  # Flatten
            x = self.relu4(self.bn4(self.fc1(x)))
            x = self.dropout4(x)
            x = self.fc2(x)
            
            return x

    model = CifarCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    # Training function
    def train(model, loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: Loss: {running_loss/(batch_idx+1):.4f} | '
                      f'Acc: {100.*correct/total:.2f}%')
        
        return running_loss / len(loader)

    # Evaluation function
    def test(model, loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        
        return accuracy

    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(50):
        print(f'Epoch {epoch+1}/50')
        train_loss = train(model, train_loader, criterion, optimizer)
        accuracy = test(model, test_loader)
        
        # Learning rate scheduling
        scheduler.step(train_loss)
        
        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'cifar_cnn_best.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= 10:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    ```

### Advanced Techniques Explained

1. **Data Augmentation**: Random horizontal flips and crops increase the effective training set size
2. **Batch Normalization**: Stabilizes and accelerates training by normalizing layer inputs
3. **Multiple Convolutional Blocks**: Deeper networks can learn more complex features
4. **Learning Rate Scheduling**: Reduces learning rate when improvement plateaus
5. **Early Stopping**: Prevents overfitting by stopping training when validation loss stops improving

## Next Steps

Now that you've learned about computer vision models in Nova, you can explore:

- NLP: Learn how to build models for text data
- Advanced Topics: Discover techniques like transfer learning and GANs