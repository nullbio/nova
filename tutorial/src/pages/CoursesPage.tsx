import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Grid, 
  Card, 
  CardContent, 
  CardMedia, 
  CardActions, 
  Button, 
  Divider, 
  Chip, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon,
  Paper,
  LinearProgress,
  Breadcrumbs,
  Link
} from '@mui/material';
import { useParams, useNavigate, Link as RouterLink } from 'react-router-dom';
import { useProgress } from '../context/ProgressContext';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import TimerIcon from '@mui/icons-material/Timer';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import SchoolIcon from '@mui/icons-material/School';
import { Course, Lesson } from '../types';

// Mock courses data - in a real app, this would come from an API
const coursesData: Course[] = [
  {
    id: 'intro-to-nova',
    title: 'Introduction to Nova',
    description: 'Learn the basics of the Nova language and how to create simple neural networks.',
    image: '/images/course-intro.jpg',
    tags: ['beginner', 'fundamentals', 'neural networks'],
    lessons: [
      {
        id: 'getting-started',
        title: 'Getting Started with Nova',
        description: 'Learn about Nova language syntax and its core concepts.',
        complexity: 'beginner',
        estimatedTime: 15,
        content: [
          {
            type: 'text',
            content: '# Welcome to Nova\n\nNova is a natural language interface for creating deep learning models with PyTorch. In this lesson, you will learn the basics of Nova syntax and how it maps to PyTorch code.\n\n## What is Nova?\n\nNova bridges the gap between natural language and deep learning code by providing a more intuitive syntax for defining and training neural networks.'
          },
          {
            type: 'image',
            content: '/images/nova-overview.png',
            explanation: 'Nova transforms natural language descriptions into PyTorch code'
          },
          {
            type: 'text',
            content: '## Nova Core Concepts\n\n- **Processing Pipeline**: A neural network model\n- **Transformation Stage**: A layer in the network\n- **Connection Strengths**: Weights and biases\n- **Activation Pattern**: Results of activation functions'
          },
          {
            type: 'quiz',
            content: {
              question: 'What does a "Processing Pipeline" represent in Nova?',
              options: [
                'A data preprocessing function',
                'A neural network model',
                'A hyperparameter tuning process',
                'A visualization tool'
              ],
              correctAnswer: 1,
              explanation: 'In Nova, a "Processing Pipeline" represents a neural network model. This terminology helps make the concept more intuitive for those new to deep learning.'
            }
          }
        ]
      },
      {
        id: 'first-neural-network',
        title: 'Creating Your First Neural Network',
        description: 'Build and understand a simple neural network in Nova.',
        complexity: 'beginner',
        estimatedTime: 20,
        content: [
          {
            type: 'text',
            content: '# Building Your First Neural Network\n\nIn this lesson, we\'ll create a simple neural network using Nova syntax. You\'ll see how Nova makes it intuitive to define network architecture.'
          },
          {
            type: 'code',
            content: {
              nova: 'create processing pipeline simple_classifier:\n    add transformation stage fully_connected with 784 inputs and 128 outputs\n    apply relu activation\n    add transformation stage fully_connected with 128 inputs and 10 outputs\n    apply softmax activation',
              pytorch: 'import torch\nimport torch.nn as nn\n\nclass SimpleClassifier(nn.Module):\n    def __init__(self):\n        super(SimpleClassifier, self).__init__()\n        self.fc1 = nn.Linear(784, 128)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(128, 10)\n        self.softmax = nn.Softmax(dim=1)\n        \n    def forward(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        x = self.softmax(x)\n        return x'
            },
            explanation: 'This code creates a simple classifier with two fully connected layers and ReLU activation.'
          },
          {
            type: 'interactive-code',
            content: {
              nova: '# Complete the code below to add a hidden layer with 64 neurons\n\ncreate processing pipeline digit_recognizer:\n    add transformation stage fully_connected with 784 inputs and 128 outputs\n    apply relu activation\n    # Add another transformation stage here\n    \n    add transformation stage fully_connected with 64 inputs and 10 outputs\n    apply softmax activation',
              pytorch: '',
              solution: 'create processing pipeline digit_recognizer:\n    add transformation stage fully_connected with 784 inputs and 128 outputs\n    apply relu activation\n    add transformation stage fully_connected with 128 inputs and 64 outputs\n    apply relu activation\n    add transformation stage fully_connected with 64 inputs and 10 outputs\n    apply softmax activation',
              instructions: 'Add a fully connected layer with 128 inputs and 64 outputs, followed by ReLU activation.',
              hints: [
                'The missing layer should take the output of the previous layer (128) as input',
                'Don\'t forget to add an activation function after your new layer',
                'The syntax is: add transformation stage fully_connected with X inputs and Y outputs'
              ]
            }
          }
        ]
      },
      {
        id: 'understanding-data',
        title: 'Understanding Data in Nova',
        description: 'Learn how to work with data collections, samples, and feature grids in Nova.',
        complexity: 'beginner',
        estimatedTime: 25,
        content: [
          {
            type: 'text',
            content: '# Understanding Data in Nova\n\nBefore we can train a neural network, we need to understand how to work with data in Nova. In this lesson, we\'ll explore Nova\'s intuitive data concepts and how they map to PyTorch\'s data structures.\n\n## Data Concepts in Nova\n\nNova uses familiar terminology to describe data structures:\n\n- **Data Collection**: A dataset (e.g., MNIST, CIFAR-10)\n- **Sample**: An individual data point\n- **Feature Grid**: A tensor (with dimensions and types)\n- **Data Stream**: A DataLoader that provides batches of samples'
          },
          {
            type: 'code',
            content: {
              nova: '# Loading a data collection in Nova\nload data collection mnist from "torchvision.datasets"\n\n# Splitting the data collection\nsplit collection mnist into 80% training and 20% testing\n\n# Creating a data stream with batches\nprepare data stream from mnist_training with batch size 32',
              pytorch: 'import torch\nfrom torchvision import datasets, transforms\nfrom torch.utils.data import DataLoader, random_split\n\n# Load the MNIST dataset\ntransform = transforms.Compose([\n    transforms.ToTensor(),\n    transforms.Normalize((0.1307,), (0.3081,))\n])\n\nmnist_full = datasets.MNIST(\'./data\', train=True, download=True, transform=transform)\n\n# Split the dataset\ntrain_size = int(0.8 * len(mnist_full))\ntest_size = len(mnist_full) - train_size\nmnist_training, mnist_testing = random_split(mnist_full, [train_size, test_size])\n\n# Create data loaders (streams)\nmnist_training_stream = DataLoader(mnist_training, batch_size=32, shuffle=True)\nmnist_testing_stream = DataLoader(mnist_testing, batch_size=32)'
            },
            explanation: 'This code demonstrates how to load, split, and create data streams in Nova and its PyTorch equivalent.'
          },
          {
            type: 'text',
            content: '## Feature Grids (Tensors)\n\nIn deep learning, we represent data as tensors - multi-dimensional arrays of numbers. Nova calls these "feature grids" to make them more intuitive:\n\n- A 1D feature grid is like a list of numbers (a vector)\n- A 2D feature grid is like a table (a matrix)\n- A 3D feature grid could represent an image (width × height × channels)\n- A 4D feature grid could represent a batch of images'
          },
          {
            type: 'code',
            content: {
              nova: '# Creating feature grids in Nova\ncreate feature grid sample_vector with shape [10]\ncreate feature grid sample_matrix with shape [3, 4]\ncreate feature grid sample_image with shape [28, 28, 1]\ncreate feature grid sample_batch with shape [32, 28, 28, 1]',
              pytorch: 'import torch\n\n# Creating tensors in PyTorch\nsample_vector = torch.zeros(10)\nsample_matrix = torch.zeros(3, 4)\nsample_image = torch.zeros(28, 28, 1)\nsample_batch = torch.zeros(32, 28, 28, 1)'
            },
            explanation: 'This code shows how to create feature grids (tensors) of different dimensions in Nova and PyTorch.'
          },
          {
            type: 'quiz',
            content: {
              question: 'What Nova concept corresponds to a PyTorch DataLoader?',
              options: [
                'Feature Grid',
                'Sample',
                'Data Collection',
                'Data Stream'
              ],
              correctAnswer: 3,
              explanation: 'In Nova, a "Data Stream" corresponds to a PyTorch DataLoader, which provides batches of samples for training or evaluation.'
            }
          }
        ]
      },
      {
        id: 'training-models',
        title: 'Training Neural Networks',
        description: 'Learn how to train your neural networks using Nova\'s intuitive training syntax.',
        complexity: 'beginner',
        estimatedTime: 30,
        content: [
          {
            type: 'text',
            content: '# Training Neural Networks in Nova\n\nAfter defining a neural network and preparing your data, the next step is to train the model. In this lesson, we\'ll explore Nova\'s intuitive training syntax and understand the key components of the training process.\n\n## Training Concepts in Nova\n\nNova uses clear terminology to describe the training process:\n\n- **Error Measure**: Loss function that evaluates model performance\n- **Improvement Strategy**: Optimizer that updates model parameters\n- **Learning Cycle**: Epoch (one complete pass through the training data)\n- **Improvement Step**: Backpropagation and parameter updates'
          },
          {
            type: 'code',
            content: {
              nova: '# Training a neural network in Nova\ntrain digit_recognizer on mnist_data_stream:\n    measure error using cross_entropy\n    improve using gradient_descent with learning rate 0.01\n    repeat for 10 learning cycles',
              pytorch: 'import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Assuming digit_recognizer and mnist_data_stream are defined\nmodel = digit_recognizer\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.SGD(model.parameters(), lr=0.01)\n\n# Training loop\nfor epoch in range(10):  # 10 learning cycles (epochs)\n    for data, target in mnist_data_stream:\n        # Zero the parameter gradients\n        optimizer.zero_grad()\n        \n        # Forward pass\n        output = model(data)\n        \n        # Calculate loss\n        loss = criterion(output, target)\n        \n        # Backward pass and optimize\n        loss.backward()\n        optimizer.step()'
            },
            explanation: 'This code demonstrates the training process in Nova and its PyTorch equivalent. Note how Nova\'s syntax is more concise and uses intuitive terminology.'
          },
          {
            type: 'text',
            content: '## Common Error Measures (Loss Functions)\n\nDifferent tasks require different error measures:\n\n- **Cross Entropy**: For classification tasks\n- **Mean Squared Error**: For regression tasks\n- **Binary Cross Entropy**: For binary classification\n\n## Common Improvement Strategies (Optimizers)\n\n- **Gradient Descent**: Standard optimization algorithm\n- **Adam**: Adaptive optimization with momentum\n- **RMSProp**: Adaptive learning rates for each parameter'
          },
          {
            type: 'interactive-code',
            content: {
              nova: '# Complete the training code for a regression model\n\ncreate processing pipeline house_price_predictor:\n    add transformation stage fully_connected with 10 inputs and 32 outputs\n    apply relu activation\n    add transformation stage fully_connected with 32 inputs and 1 outputs\n\n# Add your training code below\ntrain house_price_predictor on housing_data_stream:\n    # Add error measure\n    \n    # Add improvement strategy\n    \n    # Add learning cycles',
              pytorch: '',
              solution: 'create processing pipeline house_price_predictor:\n    add transformation stage fully_connected with 10 inputs and 32 outputs\n    apply relu activation\n    add transformation stage fully_connected with 32 inputs and 1 outputs\n\ntrain house_price_predictor on housing_data_stream:\n    measure error using mean_squared_error\n    improve using adam with learning rate 0.001\n    repeat for 50 learning cycles',
              instructions: 'Complete the training code by adding the appropriate error measure (mean_squared_error), improvement strategy (adam with learning rate 0.001), and number of learning cycles (50).',
              hints: [
                'For regression tasks, mean_squared_error is a common error measure',
                'Adam is an effective optimizer for many tasks',
                'The syntax for the improvement strategy is: improve using [strategy] with learning rate [rate]',
                'Specify learning cycles with: repeat for [number] learning cycles'
              ]
            }
          },
          {
            type: 'quiz',
            content: {
              question: 'What is the purpose of an "Improvement Strategy" in Nova?',
              options: [
                'To visualize the model\'s performance',
                'To update the model\'s parameters based on the calculated error',
                'To evaluate the model\'s performance on test data',
                'To preprocess the input data'
              ],
              correctAnswer: 1,
              explanation: 'In Nova, an "Improvement Strategy" (optimizer in PyTorch) is responsible for updating the model\'s parameters based on the calculated error, helping the model improve over time.'
            }
          }
        ]
      },
      {
        id: 'evaluating-models',
        title: 'Evaluating Model Performance',
        description: 'Learn how to evaluate your models and interpret the results.',
        complexity: 'beginner',
        estimatedTime: 25,
        content: [
          {
            type: 'text',
            content: '# Evaluating Model Performance\n\nAfter training a model, it\'s crucial to evaluate its performance to understand how well it generalizes to new data. In this lesson, we\'ll learn how to evaluate models using Nova and interpret the results.\n\n## Evaluation Concepts\n\n- **Test Data**: Data not seen during training\n- **Predictions**: Model outputs for given inputs\n- **Performance Metrics**: Measures of model quality (accuracy, precision, recall, etc.)\n- **Confusion Matrix**: Table showing correct and incorrect predictions'
          },
          {
            type: 'code',
            content: {
              nova: '# Evaluating a model in Nova\nevaluate digit_recognizer on mnist_test_stream:\n    calculate accuracy\n    calculate confusion_matrix',
              pytorch: 'import torch\nimport numpy as np\nfrom sklearn.metrics import accuracy_score, confusion_matrix\n\n# Assuming digit_recognizer and mnist_test_stream are defined\nmodel = digit_recognizer\nmodel.eval()  # Set the model to evaluation mode\n\nall_predictions = []\nall_targets = []\n\nwith torch.no_grad():  # Disable gradient computation\n    for data, target in mnist_test_stream:\n        # Forward pass\n        output = model(data)\n        \n        # Get the predicted class\n        _, predicted = torch.max(output, 1)\n        \n        # Store predictions and targets\n        all_predictions.extend(predicted.numpy())\n        all_targets.extend(target.numpy())\n\n# Calculate accuracy\naccuracy = accuracy_score(all_targets, all_predictions)\nprint(f"Accuracy: {accuracy:.4f}")\n\n# Calculate confusion matrix\ncm = confusion_matrix(all_targets, all_predictions)\nprint("Confusion Matrix:")\nprint(cm)'
            },
            explanation: 'This code shows how to evaluate a model in Nova and PyTorch. The Nova code is much more concise.'
          },
          {
            type: 'text',
            content: '## Common Performance Metrics\n\n- **Accuracy**: Proportion of correct predictions (correct / total)\n- **Precision**: Proportion of true positives among positive predictions (TP / (TP + FP))\n- **Recall**: Proportion of true positives among actual positives (TP / (TP + FN))\n- **F1 Score**: Harmonic mean of precision and recall\n- **ROC Curve**: Plots true positive rate vs. false positive rate at different thresholds'
          },
          {
            type: 'interactive-code',
            content: {
              nova: '# Complete the evaluation code\n\ncreate processing pipeline sentiment_classifier:\n    add transformation stage fully_connected with 100 inputs and 64 outputs\n    apply relu activation\n    add transformation stage fully_connected with 64 inputs and 2 outputs\n    apply softmax activation\n\n# Train the model (assume it\'s already trained)\n\n# Add your evaluation code below\nevaluate sentiment_classifier on review_test_stream:\n    # Add metrics to calculate',
              pytorch: '',
              solution: 'create processing pipeline sentiment_classifier:\n    add transformation stage fully_connected with 100 inputs and 64 outputs\n    apply relu activation\n    add transformation stage fully_connected with 64 inputs and 2 outputs\n    apply softmax activation\n\n# Train the model (assume it\'s already trained)\n\nevaluate sentiment_classifier on review_test_stream:\n    calculate accuracy\n    calculate precision\n    calculate recall\n    calculate f1_score',
              instructions: 'Complete the evaluation code by adding appropriate metrics: accuracy, precision, recall, and f1_score.',
              hints: [
                'Start with the most common metric: accuracy',
                'For binary classification, precision and recall are important',
                'F1 score combines precision and recall',
                'The syntax is: calculate [metric_name]'
              ]
            }
          },
          {
            type: 'quiz',
            content: {
              question: 'Why is it important to evaluate a model on test data rather than training data?',
              options: [
                'Because test data is easier to process',
                'To measure how well the model generalizes to new, unseen data',
                'Because training data is usually corrupted',
                'To make the evaluation process faster'
              ],
              correctAnswer: 1,
              explanation: 'Evaluating on test data (data not seen during training) helps measure how well the model generalizes to new, unseen data. This is crucial because the ultimate goal is to make accurate predictions on new data in real-world scenarios.'
            }
          }
        ]
      },
      {
        id: 'saving-loading-models',
        title: 'Saving and Loading Models',
        description: 'Learn how to save your trained models and load them for later use.',
        complexity: 'beginner',
        estimatedTime: 15,
        content: [
          {
            type: 'text',
            content: '# Saving and Loading Models\n\nAfter training a valuable model, you\'ll want to save it for future use without retraining. In this lesson, we\'ll learn how to save and load models in Nova.\n\n## Why Save Models?\n\n- **Reuse**: Use the trained model for predictions without retraining\n- **Sharing**: Share the model with others\n- **Deployment**: Use the model in production applications\n- **Checkpointing**: Save model progress during long training processes'
          },
          {
            type: 'code',
            content: {
              nova: '# Saving a model in Nova\nsave model digit_recognizer to "models/digit_recognizer.nova"\n\n# Loading a model in Nova\nload model from "models/digit_recognizer.nova" as loaded_recognizer',
              pytorch: 'import torch\nimport torch.nn as nn\n\n# Assuming digit_recognizer is defined and trained\n\n# Save the model\ntorch.save(digit_recognizer.state_dict(), "models/digit_recognizer.pth")\n\n# Load the model\n# First, create a model instance with the same architecture\nloaded_recognizer = DigitRecognizer()  # Assuming this class is defined elsewhere\n\n# Then load the state dictionary\nloaded_recognizer.load_state_dict(torch.load("models/digit_recognizer.pth"))\n\n# Put the model in evaluation mode\nloaded_recognizer.eval()'
            },
            explanation: 'This code demonstrates how to save and load models in Nova and PyTorch. Notice how Nova simplifies the process.'
          },
          {
            type: 'text',
            content: '## What Gets Saved?\n\nWhen you save a model, various components are stored:\n\n- **Model Architecture**: The structure of the network\n- **Parameter Values**: Weights and biases learned during training\n- **Optimizer State** (optional): State of the optimizer, useful for resuming training\n- **Training History** (optional): Loss values and metrics during training'
          },
          {
            type: 'interactive-code',
            content: {
              nova: '# Complete the code to save and load a model\n\ncreate processing pipeline image_classifier:\n    add transformation stage convolutional with 3 channels, 16 filters, and 3x3 kernel\n    apply relu activation\n    add pooling stage max_pooling with 2x2 size and 2x2 stride\n    add transformation stage fully_connected with 16*13*13 inputs and 10 outputs\n    apply softmax activation\n\ntrain image_classifier on cifar10_data_stream:\n    measure error using cross_entropy\n    improve using adam with learning rate 0.001\n    repeat for 5 learning cycles\n\n# Add code to save the model\n\n\n# Add code to load the model with a different name',
              pytorch: '',
              solution: 'create processing pipeline image_classifier:\n    add transformation stage convolutional with 3 channels, 16 filters, and 3x3 kernel\n    apply relu activation\n    add pooling stage max_pooling with 2x2 size and 2x2 stride\n    add transformation stage fully_connected with 16*13*13 inputs and 10 outputs\n    apply softmax activation\n\ntrain image_classifier on cifar10_data_stream:\n    measure error using cross_entropy\n    improve using adam with learning rate 0.001\n    repeat for 5 learning cycles\n\n# Save the model\nsave model image_classifier to "models/cifar_classifier.nova"\n\n# Load the model with a different name\nload model from "models/cifar_classifier.nova" as loaded_classifier',
              instructions: 'Complete the code by adding commands to save the trained image_classifier and then load it with a new name (loaded_classifier).',
              hints: [
                'Use save model [model_name] to [file_path]',
                'Use load model from [file_path] as [new_model_name]',
                'Choose appropriate file names with the .nova extension',
                'Make sure the file paths are in quotes'
              ]
            }
          },
          {
            type: 'quiz',
            content: {
              question: 'What is the primary benefit of saving a trained model?',
              options: [
                'It makes the model more accurate',
                'It allows you to use the model without retraining it every time',
                'It automatically improves model performance',
                'It increases the model\'s learning rate'
              ],
              correctAnswer: 1,
              explanation: 'The primary benefit of saving a trained model is that it allows you to use the model for predictions without having to retrain it every time. This saves time and computational resources, especially for large models that take a long time to train.'
            }
          }
        ]
      },
      {
        id: 'making-predictions',
        title: 'Making Predictions with Nova',
        description: 'Learn how to use your trained models to make predictions on new data.',
        complexity: 'beginner',
        estimatedTime: 20,
        content: [
          {
            type: 'text',
            content: '# Making Predictions with Nova\n\nAfter training and evaluating a model, the ultimate goal is to use it to make predictions on new, unseen data. In this lesson, we\'ll learn how to use Nova models to make predictions.\n\n## Prediction Process\n\n1. **Prepare Input Data**: Format new data to match the model\'s expected input\n2. **Run the Model**: Pass the data through the model\n3. **Process Output**: Interpret the model\'s predictions\n4. **Apply Decision Logic**: Take actions based on predictions'
          },
          {
            type: 'code',
            content: {
              nova: '# Making predictions with a trained model in Nova\ncreate feature grid input_image with shape [1, 28, 28, 1] from "sample_digit.png"\n\n# Get prediction\npredict using digit_recognizer on input_image\n\n# Get prediction with probability\npredict using digit_recognizer on input_image with probabilities',
              pytorch: 'import torch\nfrom PIL import Image\nimport torchvision.transforms as transforms\nimport numpy as np\n\n# Load and preprocess the image\nimage = Image.open("sample_digit.png").convert("L")  # Convert to grayscale\ntransform = transforms.Compose([\n    transforms.Resize((28, 28)),\n    transforms.ToTensor(),\n    transforms.Normalize((0.1307,), (0.3081,))\n])\ninput_image = transform(image).unsqueeze(0)  # Add batch dimension\n\n# Make prediction\nmodel = digit_recognizer\nmodel.eval()\nwith torch.no_grad():\n    output = model(input_image)\n    \n    # Get the predicted class\n    _, predicted_class = torch.max(output, 1)\n    print(f"Predicted digit: {predicted_class.item()}")\n    \n    # Get probabilities\n    probabilities = torch.nn.functional.softmax(output, dim=1)\n    print(f"Probabilities: {probabilities.squeeze().numpy()}")'
            },
            explanation: 'This code shows how to make predictions with a trained model in Nova and PyTorch. Nova\'s syntax is much simpler and more intuitive.'
          },
          {
            type: 'text',
            content: '## Types of Predictions\n\n- **Classification**: Predicting a category or class (e.g., digit recognition)\n- **Regression**: Predicting a continuous value (e.g., house price prediction)\n- **Probability**: Predicting the likelihood of each possible outcome\n- **Sequence**: Predicting a sequence of values (e.g., text generation)\n- **Structured**: Predicting structured outputs (e.g., image segmentation)'
          },
          {
            type: 'interactive-code',
            content: {
              nova: '# Complete the code to make predictions with a sentiment classifier\n\n# Assume the model is already trained\nload model from "models/sentiment_classifier.nova" as sentiment_model\n\n# Create input features for two sample reviews\ncreate feature grid positive_review_features with shape [1, 100] from "Great product, highly recommend it!"\ncreate feature grid negative_review_features with shape [1, 100] from "Terrible quality, don\'t waste your money."\n\n# Make predictions\n# Add code to predict sentiment for the positive review\n\n\n# Add code to predict sentiment for the negative review with probabilities',
              pytorch: '',
              solution: 'load model from "models/sentiment_classifier.nova" as sentiment_model\n\ncreate feature grid positive_review_features with shape [1, 100] from "Great product, highly recommend it!"\ncreate feature grid negative_review_features with shape [1, 100] from "Terrible quality, don\'t waste your money."\n\n# Make prediction for positive review\npredict using sentiment_model on positive_review_features\n\n# Make prediction for negative review with probabilities\npredict using sentiment_model on negative_review_features with probabilities',
              instructions: 'Complete the code to make predictions on the positive and negative reviews. For the negative review, also show the probabilities.',
              hints: [
                'Use predict using [model_name] on [input_data]',
                'To get probabilities, add "with probabilities" at the end',
                'The syntax is similar for both predictions, but with different inputs',
                'Make sure to use the correct model and input names'
              ]
            }
          },
          {
            type: 'quiz',
            content: {
              question: 'Why might you want to get prediction probabilities rather than just the predicted class?',
              options: [
                'Because probabilities are always more accurate',
                'To understand the model\'s confidence in its prediction',
                'Because it\'s faster to compute',
                'Probabilities are required for model evaluation'
              ],
              correctAnswer: 1,
              explanation: 'Getting prediction probabilities allows you to understand the model\'s confidence in its prediction. A high probability (e.g., 0.98) suggests high confidence, while a lower probability (e.g., 0.51) indicates the model is less certain. This information can be crucial for decision-making, especially in critical applications.'
            }
          }
        ]
      }
    ]
  },
  {
    id: 'computer-vision',
    title: 'Computer Vision with Nova',
    description: 'Create image classification and object detection models with Nova\'s simplified syntax.',
    image: '/images/course-cv.jpg',
    tags: ['intermediate', 'computer vision', 'CNNs'],
    lessons: [
      {
        id: 'intro-to-cnns',
        title: 'Introduction to Convolutional Networks',
        description: 'Learn how to build and train convolutional neural networks for image processing.',
        complexity: 'intermediate',
        estimatedTime: 30,
        content: [
          {
            type: 'text',
            content: '# Introduction to Convolutional Networks\n\nConvolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. In this lesson, you will learn how to define CNNs using Nova syntax.'
          },
          {
            type: 'code',
            content: {
              nova: 'create processing pipeline image_classifier:\n    add transformation stage convolutional with 3 channels, 16 filters, and 3x3 kernel\n    apply relu activation\n    add pooling stage max_pooling with 2x2 size and 2x2 stride\n    add transformation stage fully_connected with 16*13*13 inputs and 10 outputs\n    apply softmax activation',
              pytorch: 'import torch\nimport torch.nn as nn\n\nclass ImageClassifier(nn.Module):\n    def __init__(self):\n        super(ImageClassifier, self).__init__()\n        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)\n        self.relu = nn.ReLU()\n        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)\n        self.fc = nn.Linear(16*13*13, 10)\n        self.softmax = nn.Softmax(dim=1)\n        \n    def forward(self, x):\n        x = self.conv1(x)\n        x = self.relu(x)\n        x = self.pool(x)\n        x = x.view(-1, 16*13*13)\n        x = self.fc(x)\n        x = self.softmax(x)\n        return x'
            },
            explanation: 'This code creates a simple CNN for image classification with one convolutional layer followed by pooling and a fully connected layer.'
          }
        ]
      }
    ]
  },
  {
    id: 'advanced-techniques',
    title: 'Advanced ML Techniques',
    description: 'Learn transfer learning, fine-tuning, and other advanced techniques using Nova.',
    image: '/images/course-advanced.jpg',
    tags: ['advanced', 'transfer learning', 'fine-tuning'],
    lessons: [
      {
        id: 'transfer-learning',
        title: 'Transfer Learning with Pre-trained Models',
        description: 'Learn how to leverage pre-trained models for your specific tasks.',
        complexity: 'advanced',
        estimatedTime: 35,
        content: [
          {
            type: 'text',
            content: '# Transfer Learning with Nova\n\nTransfer learning allows you to use knowledge from pre-trained models on new tasks. This lesson will show you how to use transfer learning with Nova.'
          },
          {
            type: 'code',
            content: {
              nova: 'load pretrained model "resnet18"\n\n# Freeze the early layers\nmark early_layers as fixed from layer 1 to layer 4\n\n# Replace the classifier\nreplace classifier with:\n    add transformation stage fully_connected with 512 inputs and 128 outputs\n    apply relu activation\n    add transformation stage fully_connected with 128 inputs and 5 outputs\n    apply softmax activation',
              pytorch: 'import torch\nimport torchvision.models as models\nimport torch.nn as nn\n\n# Load pre-trained ResNet18\nmodel = models.resnet18(pretrained=True)\n\n# Freeze early layers\nfor idx, param in enumerate(list(model.parameters())[:4]):\n    param.requires_grad = False\n\n# Replace the classifier\nmodel.fc = nn.Sequential(\n    nn.Linear(512, 128),\n    nn.ReLU(),\n    nn.Linear(128, 5),\n    nn.Softmax(dim=1)\n)'
            },
            explanation: 'This code demonstrates how to load a pre-trained ResNet18 model, freeze the early layers, and replace the classifier with a custom one for a specific task.'
          }
        ]
      }
    ]
  }
];

const CoursesPage: React.FC = () => {
  const { courseId } = useParams<{ courseId: string }>();
  const navigate = useNavigate();
  const { progress, isLessonCompleted } = useProgress();
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null);

  useEffect(() => {
    if (courseId) {
      const course = coursesData.find(c => c.id === courseId);
      setSelectedCourse(course || null);
      
      if (!course) {
        navigate('/courses', { replace: true });
      }
    } else {
      setSelectedCourse(null);
    }
  }, [courseId, navigate]);

  const handleLessonClick = (lessonId: string) => {
    if (selectedCourse) {
      navigate(`/courses/${selectedCourse.id}/lessons/${lessonId}`);
    }
  };

  const handleBackToCourses = () => {
    navigate('/courses');
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'beginner':
        return 'success';
      case 'intermediate':
        return 'warning';
      case 'advanced':
        return 'error';
      default:
        return 'default';
    }
  };

  const calculateCourseProgress = (course: Course): number => {
    if (!course.lessons.length) return 0;
    
    const completedCount = course.lessons.filter(lesson => 
      isLessonCompleted(lesson.id)
    ).length;
    
    return Math.round((completedCount / course.lessons.length) * 100);
  };

  // Render the list of courses
  const renderCoursesList = () => (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Available Courses
      </Typography>
      
      <Typography variant="body1" paragraph color="text.secondary">
        Select a course to begin your learning journey with Nova.
      </Typography>
      
      <Grid container spacing={4}>
        {coursesData.map((course) => {
          const progress = calculateCourseProgress(course);
          
          return (
            <Grid item xs={12} md={4} key={course.id}>
              <Card 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                  }
                }}
                onClick={() => navigate(`/courses/${course.id}`)}
              >
                <CardMedia
                  component="img"
                  height="140"
                  image={course.image || "https://via.placeholder.com/300x140?text=Nova+Course"}
                  alt={course.title}
                />
                
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h5" component="h2" gutterBottom>
                    {course.title}
                  </Typography>
                  
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {course.description}
                  </Typography>
                  
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                    {course.tags.map((tag) => (
                      <Chip 
                        key={tag} 
                        label={tag} 
                        size="small" 
                        color={
                          tag === 'beginner' ? 'success' : 
                          tag === 'intermediate' ? 'warning' : 
                          tag === 'advanced' ? 'error' : 'default'
                        }
                        variant="outlined"
                      />
                    ))}
                  </Box>
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <SchoolIcon fontSize="small" sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="body2" color="text.secondary">
                      {course.lessons.length} {course.lessons.length === 1 ? 'Lesson' : 'Lessons'}
                    </Typography>
                  </Box>
                </CardContent>
                
                <Box sx={{ p: 2, pt: 0 }}>
                  <Box sx={{ width: '100%', mb: 1 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2" color="text.secondary">
                        Progress
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {progress}%
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={progress} 
                      sx={{ height: 6, borderRadius: 3 }}
                    />
                  </Box>
                </Box>
                
                <Divider />
                
                <CardActions>
                  <Button 
                    size="small" 
                    color="primary"
                    endIcon={<PlayArrowIcon />}
                    fullWidth
                  >
                    {progress > 0 ? 'Continue Learning' : 'Start Course'}
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          );
        })}
      </Grid>
    </Container>
  );

  // Render the course details with lessons
  const renderCourseDetails = () => {
    if (!selectedCourse) return null;
    
    const courseProgress = calculateCourseProgress(selectedCourse);
    
    return (
      <Container maxWidth="lg">
        <Box sx={{ mb: 4 }}>
          <Breadcrumbs separator={<NavigateNextIcon fontSize="small" />}>
            <Link 
              component={RouterLink} 
              to="/courses" 
              underline="hover" 
              color="inherit"
            >
              Courses
            </Link>
            <Typography color="text.primary">{selectedCourse.title}</Typography>
          </Breadcrumbs>
        </Box>
        
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={handleBackToCourses}
          sx={{ mb: 2 }}
        >
          Back to Courses
        </Button>
        
        <Card sx={{ mb: 4 }}>
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' } }}>
            <CardMedia
              component="img"
              sx={{ 
                width: { xs: '100%', md: 340 },
                height: { xs: 200, md: 'auto' }
              }}
              image={selectedCourse.image || "https://via.placeholder.com/600x400?text=Nova+Course"}
              alt={selectedCourse.title}
            />
            
            <CardContent sx={{ flex: '1 0 auto', p: 3 }}>
              <Typography variant="h4" component="h1" gutterBottom>
                {selectedCourse.title}
              </Typography>
              
              <Typography variant="body1" paragraph>
                {selectedCourse.description}
              </Typography>
              
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                {selectedCourse.tags.map((tag) => (
                  <Chip 
                    key={tag} 
                    label={tag} 
                    size="small" 
                    color={
                      tag === 'beginner' ? 'success' : 
                      tag === 'intermediate' ? 'warning' : 
                      tag === 'advanced' ? 'error' : 'default'
                    }
                  />
                ))}
              </Box>
              
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <SchoolIcon fontSize="small" sx={{ mr: 1 }} />
                <Typography variant="body2" color="text.secondary">
                  {selectedCourse.lessons.length} {selectedCourse.lessons.length === 1 ? 'Lesson' : 'Lessons'}
                </Typography>
              </Box>
              
              <Box sx={{ width: '100%', mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                  <Typography variant="body2">Course Progress</Typography>
                  <Typography variant="body2" color="text.secondary">
                    {courseProgress}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={courseProgress} 
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </CardContent>
          </Box>
        </Card>
        
        <Typography variant="h5" gutterBottom>
          Course Lessons
        </Typography>
        
        <Paper elevation={2} sx={{ mb: 4 }}>
          <List sx={{ width: '100%' }}>
            {selectedCourse.lessons.map((lesson, index) => {
              const isCompleted = isLessonCompleted(lesson.id);
              
              return (
                <React.Fragment key={lesson.id}>
                  {index > 0 && <Divider component="li" />}
                  <ListItem 
                    alignItems="flex-start" 
                    sx={{ 
                      py: 2,
                      cursor: 'pointer',
                      '&:hover': {
                        bgcolor: 'action.hover',
                      }
                    }}
                    onClick={() => handleLessonClick(lesson.id)}
                  >
                    <ListItemIcon sx={{ pt: 0.5 }}>
                      {isCompleted ? (
                        <CheckCircleIcon color="success" />
                      ) : (
                        <MenuBookIcon color="primary" />
                      )}
                    </ListItemIcon>
                    
                    <ListItemText
                      primary={
                        <Typography variant="h6" component="div">
                          {lesson.title}
                        </Typography>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary" paragraph>
                            {lesson.description}
                          </Typography>
                          
                          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                            <Chip 
                              size="small" 
                              label={lesson.complexity} 
                              color={getComplexityColor(lesson.complexity) as any}
                              sx={{ mr: 2 }}
                            />
                            
                            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                              <TimerIcon fontSize="small" sx={{ mr: 0.5 }} />
                              <Typography variant="body2" color="text.secondary">
                                {lesson.estimatedTime} min
                              </Typography>
                            </Box>
                          </Box>
                        </Box>
                      }
                    />
                    
                    <Button 
                      variant="outlined" 
                      color="primary"
                      endIcon={<PlayArrowIcon />}
                      sx={{ mt: 2 }}
                    >
                      {isCompleted ? 'Review' : 'Start'}
                    </Button>
                  </ListItem>
                </React.Fragment>
              );
            })}
          </List>
        </Paper>
      </Container>
    );
  };

  return courseId ? renderCourseDetails() : renderCoursesList();
};

export default CoursesPage;