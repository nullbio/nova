# Advanced Techniques

This page demonstrates more advanced deep learning techniques using Nova, including transfer learning, model fine-tuning, and custom training loops.

## Transfer Learning with Pre-trained Models

Transfer learning allows you to leverage pre-trained models and adapt them to your specific task. This example shows how to use a pre-trained ResNet model for image classification.

=== "Nova"
    ```
    # Load a pre-trained ResNet model
    load pretrained model resnet18 from torchvision.models
    
    # Modify the final layer for our task (10 classes)
    customize resnet18:
        replace final layer with fully_connected with 512 inputs and 10 outputs
    
    # Freeze earlier layers to preserve learned features
    freeze resnet18 layers except final layer
    
    # Load and prepare custom dataset
    load data collection custom_dataset from folder "data/custom_images" with:
        apply image resizing to 224x224
        apply normalization with means [0.485, 0.456, 0.406] and deviations [0.229, 0.224, 0.225]
        split into 80% training and 20% validation
    
    # Prepare data streams
    prepare data stream train_stream from custom_dataset.train with batch size 32 and shuffle enabled
    prepare data stream val_stream from custom_dataset.validation with batch size 64
    
    # Fine-tune the model
    train resnet18 on train_stream:
        measure error using cross_entropy
        improve using adam with learning rate 0.0001
        repeat for 10 learning cycles
        evaluate on val_stream every 1 cycle
        save best model based on validation accuracy
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models, transforms, datasets
    from torch.utils.data import DataLoader
    import os
    
    # Load pre-trained model
    model = models.resnet18(pretrained=True)
    
    # Replace final fully connected layer
    num_classes = 10
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Freeze all layers except the final layer
    for name, param in model.named_parameters():
        if "fc" not in name:  # fc = final fully connected layer
            param.requires_grad = False
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root="data/custom_images", transform=transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
    
    # Training function
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
            
            # Validation phase
            model.eval()
            running_corrects = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)
            
            epoch_acc = running_corrects.double() / len(val_loader.dataset)
            print(f"Validation Accuracy: {epoch_acc:.4f}")
            
            # Save best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Saved new best model with accuracy: {best_acc:.4f}")
    
    # Run training
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
    ```

## Implementing Custom Training Loops

This example shows how to create a custom training loop with more fine-grained control over the training process.

=== "Nova"
    ```
    # Load dataset
    load data collection cifar10 from torchvision.datasets with:
        apply standard image augmentation
        apply normalization with means [0.4914, 0.4822, 0.4465] and deviations [0.2023, 0.1994, 0.2010]
    
    # Create a CNN model
    create processing pipeline custom_cnn:
        add transformation stage convolution with 3 input channels, 32 output channels, kernel size 3
        apply batch normalization
        apply relu activation
        add max pooling with kernel size 2
        add transformation stage convolution with 32 input channels, 64 output channels, kernel size 3
        apply batch normalization
        apply relu activation
        add max pooling with kernel size 2
        add flatten operation
        add transformation stage fully_connected with 2304 inputs and 512 outputs
        apply relu activation
        add dropout with rate 0.5
        add transformation stage fully_connected with 512 inputs and 10 outputs
    
    # Set up optimizer with learning rate scheduler
    configure training:
        use optimizer adam with learning rate 0.001 and weight decay 0.0001
        use learning rate scheduler step with step size 7 and gamma 0.1
        use loss function cross_entropy
    
    # Define custom training loop
    define custom training loop for custom_cnn:
        for each epoch in range 20:
            set model to training mode
            track training loss
            for each batch in cifar10.train with batch size 128:
                compute forward pass
                calculate loss
                compute backward pass and update weights
                update training statistics
            
            set model to evaluation mode
            track evaluation metrics
            for each batch in cifar10.test with batch size 256:
                compute forward pass
                calculate loss and accuracy
                update evaluation statistics
            
            print statistics for epoch
            update learning rate scheduler
            
            if this is the best model so far:
                save model checkpoint
    
    # Execute the custom training loop
    run custom training loop
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preparation with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR10
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # Define CNN model
    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2)
            
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 64 channels, 8x8 after pooling
            self.relu3 = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 10)
        
        def forward(self, x):
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.flatten(x)
            x = self.dropout(self.relu3(self.fc1(x)))
            x = self.fc2(x)
            return x
    
    # Initialize model, loss, optimizer
    model = CustomCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Custom training loop
    def train_and_evaluate():
        best_acc = 0.0
        
        for epoch in range(20):
            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {train_loss/(batch_idx+1):.3f}, '
                          f'Acc: {100.0*correct/total:.3f}%')
            
            # Evaluation phase
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_acc = 100.0 * correct / total
            print(f'Epoch: {epoch+1}, Test Loss: {test_loss/len(test_loader):.3f}, '
                  f'Test Acc: {test_acc:.3f}%')
            
            # Update scheduler
            scheduler.step()
            
            # Save checkpoint if it's the best so far
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': best_acc,
                }, 'best_model_checkpoint.pt')
                print(f'Saved new best model with accuracy: {best_acc:.3f}%')
    
    # Run training
    train_and_evaluate()
    ```

## Multi-GPU Training

This example demonstrates how to efficiently train on multiple GPUs using data parallelism.

=== "Nova"
    ```
    # Load dataset
    load data collection imagenet from folder "data/imagenet" with:
        apply standard image augmentation
        apply normalization with standard imagenet values
    
    # Create a deep CNN model (ResNet50)
    load pretrained model resnet50 from torchvision.models
    
    # Customize for our task
    customize resnet50:
        replace final layer with fully_connected with 2048 inputs and 1000 outputs
    
    # Configure distributed training
    configure distributed training:
        use all available GPUs
        wrap model with data parallel
    
    # Prepare data loaders with appropriate workers 
    prepare data stream train_stream from imagenet.train with:
        batch size 256
        shuffle enabled
        num workers 16
        pin memory enabled
    
    prepare data stream val_stream from imagenet.validation with:
        batch size 256
        num workers 16
        pin memory enabled
    
    # Train the model with distribution awareness
    train resnet50 on train_stream:
        measure error using cross_entropy
        improve using sgd with:
            learning rate 0.1
            momentum 0.9
            weight decay 0.0001
        adjust learning rate using cosine annealing
        repeat for 90 learning cycles
        evaluate on val_stream every 1 cycle
        save checkpoint every 10 cycles
    ```

=== "PyTorch"
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torchvision import models, datasets, transforms
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preparation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder('data/imagenet/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('data/imagenet/val', transform=val_transform)
    
    # Multi-GPU setup
    def setup_for_distributed(is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print
    
        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
    
        __builtin__.print = print
    
    def main_worker(gpu, ngpus_per_node, args):
        args.gpu = gpu
        args.rank = args.rank * ngpus_per_node + gpu
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl', 
            init_method='tcp://127.0.0.1:23456',
            world_size=args.world_size, 
            rank=args.rank
        )
        
        # Suppress print statements for non-master processes
        setup_for_distributed(args.rank == 0)
        
        # Create model
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1000)
        model = model.cuda(args.gpu)
        
        # Use DistributedDataParallel
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
        
        # Create dataloaders with DistributedSampler
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=256 // ngpus_per_node, shuffle=False,
            num_workers=16, pin_memory=True, sampler=train_sampler
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=256 // ngpus_per_node, shuffle=False,
            num_workers=16, pin_memory=True
        )
        
        # Training loop
        for epoch in range(90):
            train_sampler.set_epoch(epoch)
            
            # Train one epoch
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # Print statistics
            if args.rank == 0:
                print(f'Epoch: {epoch+1}, Train Loss: {train_loss/len(train_loader):.3f}, '
                      f'Train Acc: {100.0*correct/total:.3f}%')
            
            # Evaluate
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.cuda(args.gpu), targets.cuda(args.gpu)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            # Print statistics and save checkpoint
            if args.rank == 0:
                val_acc = 100.0 * correct / total
                print(f'Validation Loss: {val_loss/len(val_loader):.3f}, '
                      f'Validation Acc: {val_acc:.3f}%')
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f'checkpoint_epoch_{epoch+1}.pt')
            
            # Update scheduler
            scheduler.step()
    
    def main():
        # Set up multiprocessing environment
        args = type('', (), {})()
        args.world_size = torch.cuda.device_count()
        args.rank = 0  # Global rank
        
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    
    if __name__ == "__main__":
        main()
    ```

These examples demonstrate advanced techniques that are commonly used in deep learning workflows, showing how Nova simplifies complex operations while maintaining the full power of PyTorch.