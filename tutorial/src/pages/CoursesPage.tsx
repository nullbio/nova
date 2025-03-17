import React, { useState, useEffect } from "react";
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
  Link,
} from "@mui/material";
import { useParams, useNavigate, Link as RouterLink } from "react-router-dom";
import { useProgress } from "../context/ProgressContext";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import MenuBookIcon from "@mui/icons-material/MenuBook";
import TimerIcon from "@mui/icons-material/Timer";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import NavigateNextIcon from "@mui/icons-material/NavigateNext";
import SchoolIcon from "@mui/icons-material/School";
import { Course, Lesson } from "../types";

// Mock courses data - in a real app, this would come from an API
export const coursesData: Course[] = [
  {
    id: "intro-to-nova",
    title: "Introduction to Nova",
    description:
      "Learn the basics of the Nova language and how to create simple neural networks.",
    image: "/images/course-intro.jpg",
    tags: ["beginner", "fundamentals", "neural networks"],
    lessons: [
      {
        id: "getting-started",
        title: "Getting Started with Nova",
        description: "Learn about Nova language syntax and its core concepts.",
        complexity: "beginner",
        estimatedTime: 15,
        content: [
          {
            type: "text",
            content:
              "# Welcome to Nova\n\nNova is a natural language interface for creating deep learning models with PyTorch. In this lesson, you will learn the basics of Nova syntax and how it maps to PyTorch code.\n\n## What is Nova?\n\nNova bridges the gap between natural language and deep learning code by providing a more intuitive syntax for defining and training neural networks. Instead of requiring you to understand complex PyTorch concepts immediately, Nova allows you to express your models using familiar, everyday language.\n\nThink of it as a translator that converts human-readable instructions into the technical language that machines understand. This makes machine learning more accessible to beginners while still generating proper PyTorch code that you can learn from.",
          },
          {
            type: "text",
            content:
              "## Nova at a Glance\n\nNova transforms natural language descriptions into PyTorch code. This means you can write code that looks like this:\n\n```\ncreate processing pipeline digit_classifier:\n    add transformation stage fully_connected with 784 inputs and 128 outputs\n    apply relu activation\n```\n\nAnd it will be translated to proper PyTorch code. This makes machine learning more accessible without sacrificing the power and flexibility of PyTorch.",
          },
          {
            type: "text",
            content:
              "## The Challenge with Traditional Deep Learning\n\nTraditional deep learning frameworks like PyTorch require you to understand many technical concepts before you can build even simple models:\n\n- **Complex Terminology**: Terms like tensors, gradients, and backpropagation\n- **Mathematical Foundations**: Linear algebra, calculus, and statistics\n- **Framework-Specific Patterns**: Understanding PyTorch's specific coding patterns\n- **Boilerplate Code**: Writing repetitive scaffolding code\n\nThis steep learning curve can be intimidating and discouraging for beginners. Nova aims to solve this problem by providing an intuitive bridge to machine learning.",
          },
          {
            type: "text",
            content:
              "## Nova's Pipeline Metaphor\n\nNova reimagines neural networks as **data transformation pipelines** - a concept familiar to most programmers who have worked with data processing, ETL processes, Unix pipes, or functional programming chains.\n\nIn this mental model:\n\n1. **Data flows** through a series of transformation steps\n2. Each step modifies or extracts information from the data\n3. The final output represents the model's prediction or result\n\nThis pipeline metaphor makes it easier to conceptualize how neural networks process information, without immediately having to understand all the mathematical details.",
          },
          {
            type: "code",
            content: {
              nova: "# A simple Nova pipeline\ncreate processing pipeline digit_classifier:\n    add transformation stage fully_connected with 784 inputs and 128 outputs\n    apply relu activation\n    add transformation stage fully_connected with 128 inputs and 10 outputs",
              pytorch:
                "import torch\nimport torch.nn as nn\n\nclass DigitClassifier(nn.Module):\n    def __init__(self):\n        super(DigitClassifier, self).__init__()\n        self.fc1 = nn.Linear(784, 128)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(128, 10)\n        \n    def forward(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        return x",
            },
            explanation:
              "The Nova code (left) is more readable and intuitive, while the PyTorch code (right) requires understanding of classes, inheritance, and PyTorch-specific patterns.",
          },
          {
            type: "text",
            content:
              "## Nova Core Concepts\n\nNova uses intuitive terminology to describe the components of a neural network:\n\n### Model Components\n\n- **Processing Pipeline**: A complete neural network model (equivalent to PyTorch's `nn.Module`)\n- **Transformation Stage**: A layer in the network that transforms input data (like `nn.Linear` in PyTorch)\n- **Connection Strengths**: The weights and biases that are learned during training\n- **Activation Pattern**: The result after applying an activation function like ReLU\n\n### Data Concepts\n\n- **Feature Grid**: A multi-dimensional array of values (PyTorch's tensor)\n- **Data Collection**: A dataset containing samples for training or evaluation\n- **Data Stream**: An iterable that provides batches of data (like DataLoader in PyTorch)\n\n### Training Concepts\n\n- **Error Measure**: A function that quantifies prediction errors (loss function)\n- **Improvement Strategy**: An algorithm for updating model parameters (optimizer)\n- **Learning Cycle**: One complete pass through the training dataset (epoch)",
          },
          {
            type: "quiz",
            content: {
              question: 'What does a "Processing Pipeline" represent in Nova?',
              options: [
                "A data preprocessing function",
                "A neural network model",
                "A hyperparameter tuning process",
                "A visualization tool",
              ],
              correctAnswer: 1,
              explanation:
                'In Nova, a "Processing Pipeline" represents a neural network model. This terminology helps make the concept more intuitive for those new to deep learning.',
            },
          },
          {
            type: "text",
            content:
              '## Why Nova Uses Different Terminology\n\nYou might wonder: "Why not just use the standard PyTorch terminology?"\n\nNova\'s alternative terminology serves several important purposes:\n\n1. **Intuitive Understanding**: Terms like "Processing Pipeline" and "Transformation Stage" are more descriptive and easier to grasp than "Module" or "Layer"\n\n2. **Mental Model**: The pipeline metaphor provides a conceptual framework that matches how most people think about data processing\n\n3. **Reduced Intimidation**: Traditional ML terminology can be intimidating to newcomers and create unnecessary barriers to entry\n\n4. **Learning Bridge**: Nova terminology acts as a bridge, allowing you to focus on concepts first, then gradually learn the corresponding PyTorch terms',
          },
          {
            type: "interactive-code",
            content: {
              nova: "# Try to complete this Nova code to create a simple classifier\n\ncreate processing pipeline simple_classifier:\n    # Add a transformation stage with 10 inputs and 5 outputs\n    \n    # Apply the relu activation function\n    \n    # Add another transformation stage with 5 inputs and 2 outputs",
              pytorch: "",
              solution:
                "create processing pipeline simple_classifier:\n    add transformation stage fully_connected with 10 inputs and 5 outputs\n    apply relu activation\n    add transformation stage fully_connected with 5 inputs and 2 outputs",
              instructions:
                "Complete the code by adding a fully connected layer with 10 inputs and 5 outputs, followed by a ReLU activation, and then another fully connected layer with 5 inputs and 2 outputs.",
              hints: [
                "For the transformation stage, use: add transformation stage fully_connected with X inputs and Y outputs",
                "For activation functions, use: apply relu activation",
                "Make sure the input size of the second layer matches the output size of the first layer",
              ],
            },
          },
          {
            type: "text",
            content:
              "## Learning Path with Nova\n\nAs you progress through Nova tutorials, you'll follow this learning path:\n\n1. **Nova Basics**: Learn the intuitive Nova syntax and terminology\n2. **Side-by-Side Comparison**: See Nova code alongside equivalent PyTorch code\n3. **PyTorch Mapping**: Understand how Nova concepts map to PyTorch components\n4. **Advanced Concepts**: Gradually learn more advanced ML concepts using Nova's intuitive framework\n\nBy the end of this course, you'll understand both Nova's intuitive syntax AND the underlying PyTorch code it generates, giving you the best of both worlds.",
          },
          {
            type: "quiz",
            content: {
              question: "Which of the following best describes Nova's purpose?",
              options: [
                "To replace PyTorch completely",
                "To provide an alternative syntax that compiles to a different framework than PyTorch",
                "To serve as a bridge between intuitive language and PyTorch code",
                "To simplify PyTorch by removing advanced features",
              ],
              correctAnswer: 2,
              explanation:
                "Nova is designed to be a bridge or translator between intuitive natural language and PyTorch code. It doesn't replace PyTorch but rather makes it more accessible while still generating proper PyTorch code that you can learn from.",
            },
          },
        ],
      },
      {
        id: "first-neural-network",
        title: "Creating Your First Neural Network",
        description: "Build and understand a simple neural network in Nova.",
        complexity: "beginner",
        estimatedTime: 20,
        content: [
          {
            type: "text",
            content:
              "# Building Your First Neural Network\n\nIn this lesson, we'll create a simple neural network using Nova syntax. You'll see how Nova makes it intuitive to define network architecture and understand the flow of data through a machine learning model.\n\n## What is a Neural Network?\n\nBefore we dive into code, let's understand what a neural network actually is:\n\nA neural network is a computational model inspired by the human brain. It consists of interconnected nodes (neurons) organized in layers that process information. These networks can learn patterns from data and make predictions on new, unseen examples.\n\nIn Nova, we view neural networks as **processing pipelines** that transform input data through multiple stages to produce a desired output.",
          },
          {
            type: "text",
            content:
              "## Neural Network Visualization\n\nA neural network typically consists of:\n\n1. **Input Layer**: Receives the raw data (e.g., flattened pixels of an image)\n2. **Hidden Layers**: Process and transform the data through weighted connections\n3. **Output Layer**: Produces the final prediction (e.g., digit classification)\n\nData flows from input to output, with each layer applying transformations to extract meaningful patterns.",
          },
          {
            type: "text",
            content:
              "## Understanding the MNIST Digit Recognition Task\n\nFor our first neural network, we'll tackle a classic machine learning problem: handwritten digit recognition using the MNIST dataset.\n\n**The Task**: Given a 28×28 pixel grayscale image of a handwritten digit (0-9), correctly identify which digit it represents.\n\n**The Data**: Each image has 784 pixels (28×28), with each pixel having a grayscale value between 0-255. We'll flatten these 2D images into a 1D array of 784 values as input to our model.\n\n**The Output**: Our model will produce 10 output values representing the probability that the input image belongs to each of the 10 possible digit classes (0-9).",
          },
          {
            type: "text",
            content:
              "## Designing Our Neural Network\n\nNow let's design a simple neural network for this task. We'll create a model with:\n\n1. An input layer accepting 784 features (one per pixel)\n2. A hidden layer with 128 neurons to learn patterns\n3. An output layer with 10 neurons (one per digit class)\n4. Activation functions to introduce non-linearity\n\nHere's how we express this neural network in Nova syntax:",
          },
          {
            type: "code",
            content: {
              nova: "create processing pipeline digit_classifier:\n    add transformation stage fully_connected with 784 inputs and 128 outputs\n    apply relu activation\n    add transformation stage fully_connected with 128 inputs and 10 outputs\n    apply softmax activation",
              pytorch:
                "import torch\nimport torch.nn as nn\n\nclass DigitClassifier(nn.Module):\n    def __init__(self):\n        super(DigitClassifier, self).__init__()\n        self.fc1 = nn.Linear(784, 128)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(128, 10)\n        self.softmax = nn.Softmax(dim=1)\n        \n    def forward(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        x = self.softmax(x)\n        return x",
            },
            explanation:
              "This code creates a neural network for digit classification with an input layer (784 neurons), one hidden layer (128 neurons), and an output layer (10 neurons). The ReLU activation adds non-linearity, while the softmax activation converts the output to probabilities that sum to 1.",
          },
          {
            type: "text",
            content:
              '## Understanding Each Component\n\nLet\'s break down the Nova code line by line:\n\n1. `create processing pipeline digit_classifier:` - Defines a new neural network model named "digit_classifier"\n\n2. `add transformation stage fully_connected with 784 inputs and 128 outputs` - Creates the first layer, which takes 784 input features (the flattened image pixels) and produces 128 output values\n\n3. `apply relu activation` - Applies the ReLU (Rectified Linear Unit) activation function, which replaces negative values with zero while keeping positive values unchanged. This introduces non-linearity into the model.\n\n4. `add transformation stage fully_connected with 128 inputs and 10 outputs` - Creates the output layer, which takes the 128 values from the hidden layer and produces 10 outputs (one for each digit class)\n\n5. `apply softmax activation` - Applies the softmax function to convert the raw output values into probabilities that sum to 1, making them easier to interpret as confidence scores',
          },
          {
            type: "text",
            content:
              "## The Data Flow Through Our Network\n\nLet's visualize how data flows through this neural network:\n\n1. **Input**: A 28×28 pixel image is flattened into a vector of 784 values\n\n2. **First Transformation Stage**: Each of the 128 neurons in the hidden layer computes a weighted sum of all 784 input values, plus a bias term\n\n3. **ReLU Activation**: Negative values are replaced with zeros, adding non-linearity\n\n4. **Second Transformation Stage**: Each of the 10 neurons in the output layer computes a weighted sum of all 128 hidden values, plus a bias term\n\n5. **Softmax Activation**: Raw outputs are converted to probabilities that sum to 1\n\n6. **Prediction**: The digit with the highest probability is chosen as the prediction\n\nThis entire process is what makes neural networks so powerful - they can learn complex patterns from data through the adjustment of weights and biases during training.",
          },
          {
            type: "interactive-code",
            content: {
              nova: "# Complete the code below to add a hidden layer with 64 neurons\n\ncreate processing pipeline digit_recognizer:\n    add transformation stage fully_connected with 784 inputs and 128 outputs\n    apply relu activation\n    # Add another transformation stage here\n    \n    add transformation stage fully_connected with 64 inputs and 10 outputs\n    apply softmax activation",
              pytorch: "",
              solution:
                "create processing pipeline digit_recognizer:\n    add transformation stage fully_connected with 784 inputs and 128 outputs\n    apply relu activation\n    add transformation stage fully_connected with 128 inputs and 64 outputs\n    apply relu activation\n    add transformation stage fully_connected with 64 inputs and 10 outputs\n    apply softmax activation",
              instructions:
                "Add a fully connected layer with 128 inputs and 64 outputs, followed by ReLU activation. This will create a model with two hidden layers instead of just one, potentially improving its ability to learn complex patterns.",
              hints: [
                "The missing layer should take the output of the previous layer (128) as input",
                "Don't forget to add an activation function after your new layer",
                "The syntax is: add transformation stage fully_connected with X inputs and Y outputs",
              ],
            },
          },
          {
            type: "text",
            content:
              '## Why Add More Layers?\n\nYou just added a second hidden layer to the neural network. But why would we want to do this?\n\nNeural networks with multiple hidden layers (known as "deep" neural networks) can learn more complex patterns than those with only one hidden layer. Each layer can learn increasingly abstract representations of the data:\n\n- The first hidden layer might learn to detect simple patterns like edges and curves\n- The second hidden layer might combine these to recognize parts of digits\n- The output layer combines these parts to identify complete digits\n\nThis hierarchical learning is what makes deep learning so powerful for complex tasks like image recognition, natural language processing, and more.',
          },
          {
            type: "text",
            content:
              "## The Importance of Activation Functions\n\nYou've seen the ReLU activation function used in our model, but why do we need activation functions at all?\n\nWithout activation functions, neural networks would only be able to learn linear relationships, regardless of how many layers they have. This is because combining linear transformations just results in another linear transformation.\n\nActivation functions introduce non-linearity, allowing neural networks to learn complex, non-linear patterns in the data. Some common activation functions include:\n\n- **ReLU** (Rectified Linear Unit): f(x) = max(0, x)\n- **Sigmoid**: f(x) = 1 / (1 + e^(-x))\n- **Tanh**: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))\n- **Softmax**: Converts values to probabilities that sum to 1, used for classification",
          },
          {
            type: "quiz",
            content: {
              question:
                "Why do we need activation functions in neural networks?",
              options: [
                "To make the network run faster",
                "To introduce non-linearity, allowing the network to learn complex patterns",
                "To reduce the number of parameters that need to be learned",
                "To convert the output to a specific data type",
              ],
              correctAnswer: 1,
              explanation:
                "Activation functions introduce non-linearity into neural networks. Without them, no matter how many layers we add, the network would only be able to learn linear relationships between inputs and outputs. Non-linearity allows the network to learn complex patterns and relationships in the data.",
            },
          },
          {
            type: "text",
            content:
              "## Next Steps\n\nCongratulations! You've created your first neural network using Nova's intuitive syntax. In the next lesson, we'll learn how to work with data in Nova, preparing it for use with our neural network models.\n\nAs you continue with this course, you'll learn how to:\n\n1. Load and preprocess data\n2. Train your neural network\n3. Evaluate its performance\n4. Make predictions on new data\n5. Save and load your trained models\n\nAll using Nova's intuitive, natural language syntax that bridges the gap between human understanding and machine learning code.",
          },
        ],
      },
      {
        id: "understanding-data",
        title: "Understanding Data in Nova",
        description:
          "Learn how to work with data collections, samples, and feature grids in Nova.",
        complexity: "beginner",
        estimatedTime: 25,
        content: [
          {
            type: "text",
            content:
              "# Understanding Data in Nova\n\nBefore we can train a neural network, we need to understand how to work with data in Nova. In this lesson, we'll explore Nova's intuitive data concepts and how they map to PyTorch's data structures.\n\n## The Importance of Data in Machine Learning\n\nThe foundation of any machine learning project is data. As the saying goes: \"garbage in, garbage out.\" No matter how sophisticated your model is, it can't learn effectively from poor-quality or improperly formatted data.\n\nFor neural networks specifically, we need to:\n\n1. **Collect** relevant data for our task\n2. **Prepare** the data in a format suitable for neural networks\n3. **Organize** the data for efficient training and evaluation\n4. **Feed** the data to our models in appropriate batches\n\nNova provides intuitive terminology and syntax for handling all these data-related tasks.",
          },
          {
            type: "text",
            content:
              "## Data Concepts in Nova\n\nNova uses familiar terminology to describe data structures, making them more intuitive than their PyTorch counterparts:\n\n| Nova Term | PyTorch Equivalent | Description |\n|-----------|-------------------|-------------|\n| **Data Collection** | Dataset | A collection of samples used for training or evaluation |\n| **Sample** | Individual data point | A single instance in a dataset (e.g., one image) |\n| **Feature Grid** | Tensor | Multi-dimensional array of numerical values |\n| **Data Stream** | DataLoader | Iterable that provides batches of data for training |\n| **Data Transformation** | Transform | Operations applied to preprocess data |\n\nThese intuitive terms make it easier to understand the data pipeline in machine learning projects.",
          },
          {
            type: "code",
            content: {
              nova: '# Loading a data collection in Nova\nload data collection mnist from "torchvision.datasets"\n\n# Applying transformations\napply transformations to mnist:\n    convert to feature grid\n    normalize with mean 0.1307 and deviation 0.3081\n\n# Splitting the data collection\nsplit collection mnist into 80% training and 20% testing\n\n# Creating data streams with batches\nprepare data stream from mnist_training with batch size 32 and shuffle enabled',
              pytorch:
                "import torch\nfrom torchvision import datasets, transforms\nfrom torch.utils.data import DataLoader, random_split\n\n# Define transformations\ntransform = transforms.Compose([\n    transforms.ToTensor(),\n    transforms.Normalize((0.1307,), (0.3081,))\n])\n\n# Load the MNIST dataset with transformations\nmnist_full = datasets.MNIST('./data', train=True, download=True, transform=transform)\n\n# Split the dataset\ntrain_size = int(0.8 * len(mnist_full))\ntest_size = len(mnist_full) - train_size\nmnist_training, mnist_testing = random_split(mnist_full, [train_size, test_size])\n\n# Create data loaders (streams)\nmnist_training_stream = DataLoader(mnist_training, batch_size=32, shuffle=True)\nmnist_testing_stream = DataLoader(mnist_testing, batch_size=32)",
            },
            explanation:
              "This code demonstrates how to load a dataset, apply transformations, split it into training and testing sets, and create data streams. Notice how Nova's syntax is more readable and intuitive compared to the equivalent PyTorch code.",
          },
          {
            type: "text",
            content:
              "## Data Collections: Organizing Your Samples\n\nA **data collection** in Nova is equivalent to a dataset in PyTorch. It's a structured collection of samples (data points) used for training or evaluation.\n\nCommon operations on data collections include:\n\n1. **Loading** pre-existing collections (like MNIST or CIFAR-10)\n2. **Creating** custom collections from your own data\n3. **Splitting** collections into training, validation, and test sets\n4. **Applying** transformations to preprocess the data\n\nData collections are crucial because they organize your data and provide a consistent interface for models to access it.",
          },
          {
            type: "text",
            content:
              '## Feature Grids: The Building Blocks of Data\n\nIn deep learning, we represent data as multi-dimensional arrays of numbers. PyTorch calls these "tensors," but Nova uses the more intuitive term "feature grids" to help visualize what they actually represent.\n\nFeature grids can have different dimensions, each serving a different purpose:\n\n- **1D feature grid**: A list of numbers (a vector) - e.g., a single feature vector\n- **2D feature grid**: A table of numbers (a matrix) - e.g., a grayscale image without channel dimension\n- **3D feature grid**: A volume of numbers - e.g., an image with channels (width × height × channels)\n- **4D feature grid**: A batch of 3D grids - e.g., multiple images in a training batch\n\nUnderstanding these dimensions is crucial for designing and debugging neural networks.',
          },
          {
            type: "code",
            content: {
              nova: "# Creating feature grids in Nova\n\n# 1D feature grid (vector)\ncreate feature grid sample_vector with shape [10]\n\n# 2D feature grid (matrix)\ncreate feature grid sample_matrix with shape [3, 4]\n\n# 3D feature grid (single image with channels)\ncreate feature grid sample_image with shape [28, 28, 3]\n\n# 4D feature grid (batch of images with channels)\ncreate feature grid sample_batch with shape [32, 28, 28, 3]",
              pytorch:
                "import torch\n\n# Creating tensors in PyTorch\n\n# 1D tensor (vector)\nsample_vector = torch.zeros(10)\n\n# 2D tensor (matrix)\nsample_matrix = torch.zeros(3, 4)\n\n# 3D tensor (single image with channels)\nsample_image = torch.zeros(28, 28, 3)\n\n# 4D tensor (batch of images with channels)\nsample_batch = torch.zeros(32, 28, 28, 3)",
            },
            explanation:
              "This code shows how to create feature grids (tensors) of different dimensions in Nova and PyTorch. Feature grids are the fundamental data structure in deep learning, representing everything from simple vectors to complex multi-dimensional data like images or videos.",
          },
          {
            type: "text",
            content:
              "## Visualizing Feature Grids\n\nTo better understand feature grids, let's visualize them:\n\n**1D Feature Grid (Vector)**:\n```\n[5.2, 3.1, 7.6, 2.9, 8.1]\n```\nThis could represent a single sample with 5 features, like measurements of a flower.\n\n**2D Feature Grid (Matrix)**:\n```\n[[1, 2, 3],\n [4, 5, 6],\n [7, 8, 9]]\n```\nThis could represent a tiny 3×3 grayscale image or a table of data.\n\n**3D Feature Grid**:\n```\n[[[R, G, B], [R, G, B], ...],\n [[R, G, B], [R, G, B], ...],\n ...]\n```\nThis could represent a color image where each pixel has red, green, and blue channel values.\n\nFeature grids are the language of neural networks - both the inputs they process and the internal representations they learn.",
          },
          {
            type: "text",
            content:
              "## Data Streams: Feeding Data to Your Model\n\nA **data stream** in Nova (equivalent to a DataLoader in PyTorch) is responsible for providing batches of data to your model during training or evaluation.\n\nData streams handle important tasks like:\n\n1. **Batching**: Grouping samples together for more efficient processing\n2. **Shuffling**: Randomizing the order of samples to improve learning\n3. **Iteration**: Providing an easy way to loop through your entire dataset\n4. **Efficiency**: Loading data as needed rather than all at once\n\nThis batched approach is crucial for training neural networks on large datasets that might not fit in memory all at once.",
          },
          {
            type: "interactive-code",
            content: {
              nova: '# Complete the code to prepare data for training an image classifier\n\n# Load the CIFAR-10 dataset\nload data collection cifar10 from "torchvision.datasets"\n\n# Apply transformations (fill in the missing parts)\napply transformations to cifar10:\n    # Convert images to feature grids\n    \n    # Normalize with mean [0.5, 0.5, 0.5] and deviation [0.5, 0.5, 0.5]\n    \n\n# Split into training and testing sets\nsplit collection cifar10 into 80% training and 20% testing\n\n# Create data streams (fill in the missing parts)\n# Create a training stream with batch size 64 and shuffling enabled\n',
              pytorch: "",
              solution:
                'load data collection cifar10 from "torchvision.datasets"\n\napply transformations to cifar10:\n    convert to feature grid\n    normalize with mean [0.5, 0.5, 0.5] and deviation [0.5, 0.5, 0.5]\n\nsplit collection cifar10 into 80% training and 20% testing\n\nprepare data stream from cifar10_training with batch size 64 and shuffle enabled',
              instructions:
                "Complete the data preparation code by adding the missing transformations and creating a data stream with batch size 64 and shuffling enabled.",
              hints: [
                "For the conversion to feature grid, use: convert to feature grid",
                "For normalization, use: normalize with mean [values] and deviation [values]",
                "For the data stream, use: prepare data stream from [collection] with batch size [number] and shuffle enabled",
              ],
            },
          },
          {
            type: "text",
            content:
              "## Why Data Preparation Matters\n\nProper data preparation is crucial for successful machine learning. Let's understand why each step matters:\n\n1. **Converting to Feature Grids**: Neural networks operate on numerical data, so images, text, or other data must be converted to tensors (feature grids).\n\n2. **Normalization**: Neural networks learn better when input values are scaled to similar ranges (typically between -1 and 1 or 0 and 1). Without normalization, networks can be unstable and difficult to train.\n\n3. **Splitting Data**: Separate training and testing sets are essential to evaluate how well your model generalizes to new, unseen data.\n\n4. **Batching and Shuffling**: Batching improves computational efficiency, while shuffling prevents the model from learning patterns based on the order of samples.\n\nProperly prepared data can make the difference between a model that fails to learn and one that performs exceptionally well.",
          },
          {
            type: "quiz",
            content: {
              question:
                "What Nova concept corresponds to a PyTorch DataLoader?",
              options: [
                "Feature Grid",
                "Sample",
                "Data Collection",
                "Data Stream",
              ],
              correctAnswer: 3,
              explanation:
                'In Nova, a "Data Stream" corresponds to a PyTorch DataLoader. Both are responsible for providing batches of data to the model during training or evaluation, handling tasks like batching, shuffling, and efficient iteration.',
            },
          },
          {
            type: "text",
            content:
              "## Common Data Transformations\n\nData rarely comes in a format immediately suitable for neural networks. Transformations prepare raw data for effective learning:\n\n**Image Data Transformations**:\n- Resizing: Ensuring all images have consistent dimensions\n- Cropping: Focusing on the relevant parts of an image\n- Normalization: Scaling pixel values to suitable ranges\n- Data Augmentation: Creating variations to improve robustness (flips, rotations, etc.)\n\n**Text Data Transformations**:\n- Tokenization: Breaking text into words or subwords\n- Embedding: Converting tokens to numerical vectors\n- Padding: Making sequences uniform in length\n\n**Numerical Data Transformations**:\n- Normalization: Scaling to a standard range\n- Standardization: Converting to have mean 0 and standard deviation 1\n- Missing Value Handling: Imputing or encoding missing values\n\nIn the next lesson, we'll learn how to train neural networks using these prepared data streams.",
          },
          {
            type: "quiz",
            content: {
              question:
                "Why is normalization an important data transformation?",
              options: [
                "It makes the data more colorful and visually appealing",
                "It helps neural networks learn better by scaling values to similar ranges",
                "It reduces the file size of the dataset",
                "It increases the number of training samples",
              ],
              correctAnswer: 1,
              explanation:
                "Normalization scales the data values to a similar range (typically between -1 and 1 or 0 and 1), which helps neural networks learn more effectively and stably. Without normalization, features with larger scales might dominate the learning process, making it difficult for the network to learn from features with smaller scales.",
            },
          },
        ],
      },
      {
        id: "training-models",
        title: "Training Neural Networks",
        description:
          "Learn how to train your neural networks using Nova's intuitive training syntax.",
        complexity: "beginner",
        estimatedTime: 30,
        content: [
          {
            type: "text",
            content:
              "# Training Neural Networks in Nova\n\nAfter defining a neural network and preparing your data, the next step is to train the model. In this lesson, we'll explore Nova's intuitive training syntax and understand the key components of the training process.\n\n## Training Concepts in Nova\n\nNova uses clear terminology to describe the training process:\n\n- **Error Measure**: Loss function that evaluates model performance\n- **Improvement Strategy**: Optimizer that updates model parameters\n- **Learning Cycle**: Epoch (one complete pass through the training data)\n- **Improvement Step**: Backpropagation and parameter updates",
          },
          {
            type: "code",
            content: {
              nova: "# Training a neural network in Nova\ntrain digit_recognizer on mnist_data_stream:\n    measure error using cross_entropy\n    improve using gradient_descent with learning rate 0.01\n    repeat for 10 learning cycles",
              pytorch:
                "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Assuming digit_recognizer and mnist_data_stream are defined\nmodel = digit_recognizer\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.SGD(model.parameters(), lr=0.01)\n\n# Training loop\nfor epoch in range(10):  # 10 learning cycles (epochs)\n    for data, target in mnist_data_stream:\n        # Zero the parameter gradients\n        optimizer.zero_grad()\n        \n        # Forward pass\n        output = model(data)\n        \n        # Calculate loss\n        loss = criterion(output, target)\n        \n        # Backward pass and optimize\n        loss.backward()\n        optimizer.step()",
            },
            explanation:
              "This code demonstrates the training process in Nova and its PyTorch equivalent. Note how Nova's syntax is more concise and uses intuitive terminology.",
          },
          {
            type: "text",
            content:
              "## Common Error Measures (Loss Functions)\n\nDifferent tasks require different error measures:\n\n- **Cross Entropy**: For classification tasks\n- **Mean Squared Error**: For regression tasks\n- **Binary Cross Entropy**: For binary classification\n\n## Common Improvement Strategies (Optimizers)\n\n- **Gradient Descent**: Standard optimization algorithm\n- **Adam**: Adaptive optimization with momentum\n- **RMSProp**: Adaptive learning rates for each parameter",
          },
          {
            type: "interactive-code",
            content: {
              nova: "# Complete the training code for a regression model\n\ncreate processing pipeline house_price_predictor:\n    add transformation stage fully_connected with 10 inputs and 32 outputs\n    apply relu activation\n    add transformation stage fully_connected with 32 inputs and 1 outputs\n\n# Add your training code below\ntrain house_price_predictor on housing_data_stream:\n    # Add error measure\n    \n    # Add improvement strategy\n    \n    # Add learning cycles",
              pytorch: "",
              solution:
                "create processing pipeline house_price_predictor:\n    add transformation stage fully_connected with 10 inputs and 32 outputs\n    apply relu activation\n    add transformation stage fully_connected with 32 inputs and 1 outputs\n\ntrain house_price_predictor on housing_data_stream:\n    measure error using mean_squared_error\n    improve using adam with learning rate 0.001\n    repeat for 50 learning cycles",
              instructions:
                "Complete the training code by adding the appropriate error measure (mean_squared_error), improvement strategy (adam with learning rate 0.001), and number of learning cycles (50).",
              hints: [
                "For regression tasks, mean_squared_error is a common error measure",
                "Adam is an effective optimizer for many tasks",
                "The syntax for the improvement strategy is: improve using [strategy] with learning rate [rate]",
                "Specify learning cycles with: repeat for [number] learning cycles",
              ],
            },
          },
          {
            type: "quiz",
            content: {
              question:
                'What is the purpose of an "Improvement Strategy" in Nova?',
              options: [
                "To visualize the model's performance",
                "To update the model's parameters based on the calculated error",
                "To evaluate the model's performance on test data",
                "To preprocess the input data",
              ],
              correctAnswer: 1,
              explanation:
                'In Nova, an "Improvement Strategy" (optimizer in PyTorch) is responsible for updating the model\'s parameters based on the calculated error, helping the model improve over time.',
            },
          },
        ],
      },
      {
        id: "evaluating-models",
        title: "Evaluating Model Performance",
        description:
          "Learn how to evaluate your models and interpret the results.",
        complexity: "beginner",
        estimatedTime: 25,
        content: [
          {
            type: "text",
            content:
              "# Evaluating Model Performance\n\nAfter training a model, it's crucial to evaluate its performance to understand how well it generalizes to new data. In this lesson, we'll learn how to evaluate models using Nova and interpret the results.\n\n## Evaluation Concepts\n\n- **Test Data**: Data not seen during training\n- **Predictions**: Model outputs for given inputs\n- **Performance Metrics**: Measures of model quality (accuracy, precision, recall, etc.)\n- **Confusion Matrix**: Table showing correct and incorrect predictions",
          },
          {
            type: "code",
            content: {
              nova: "# Evaluating a model in Nova\nevaluate digit_recognizer on mnist_test_stream:\n    calculate accuracy\n    calculate confusion_matrix",
              pytorch:
                'import torch\nimport numpy as np\nfrom sklearn.metrics import accuracy_score, confusion_matrix\n\n# Assuming digit_recognizer and mnist_test_stream are defined\nmodel = digit_recognizer\nmodel.eval()  # Set the model to evaluation mode\n\nall_predictions = []\nall_targets = []\n\nwith torch.no_grad():  # Disable gradient computation\n    for data, target in mnist_test_stream:\n        # Forward pass\n        output = model(data)\n        \n        # Get the predicted class\n        _, predicted = torch.max(output, 1)\n        \n        # Store predictions and targets\n        all_predictions.extend(predicted.numpy())\n        all_targets.extend(target.numpy())\n\n# Calculate accuracy\naccuracy = accuracy_score(all_targets, all_predictions)\nprint(f"Accuracy: {accuracy:.4f}")\n\n# Calculate confusion matrix\ncm = confusion_matrix(all_targets, all_predictions)\nprint("Confusion Matrix:")\nprint(cm)',
            },
            explanation:
              "This code shows how to evaluate a model in Nova and PyTorch. The Nova code is much more concise.",
          },
          {
            type: "text",
            content:
              "## Common Performance Metrics\n\n- **Accuracy**: Proportion of correct predictions (correct / total)\n- **Precision**: Proportion of true positives among positive predictions (TP / (TP + FP))\n- **Recall**: Proportion of true positives among actual positives (TP / (TP + FN))\n- **F1 Score**: Harmonic mean of precision and recall\n- **ROC Curve**: Plots true positive rate vs. false positive rate at different thresholds",
          },
          {
            type: "interactive-code",
            content: {
              nova: "# Complete the evaluation code\n\ncreate processing pipeline sentiment_classifier:\n    add transformation stage fully_connected with 100 inputs and 64 outputs\n    apply relu activation\n    add transformation stage fully_connected with 64 inputs and 2 outputs\n    apply softmax activation\n\n# Train the model (assume it's already trained)\n\n# Add your evaluation code below\nevaluate sentiment_classifier on review_test_stream:\n    # Add metrics to calculate",
              pytorch: "",
              solution:
                "create processing pipeline sentiment_classifier:\n    add transformation stage fully_connected with 100 inputs and 64 outputs\n    apply relu activation\n    add transformation stage fully_connected with 64 inputs and 2 outputs\n    apply softmax activation\n\n# Train the model (assume it's already trained)\n\nevaluate sentiment_classifier on review_test_stream:\n    calculate accuracy\n    calculate precision\n    calculate recall\n    calculate f1_score",
              instructions:
                "Complete the evaluation code by adding appropriate metrics: accuracy, precision, recall, and f1_score.",
              hints: [
                "Start with the most common metric: accuracy",
                "For binary classification, precision and recall are important",
                "F1 score combines precision and recall",
                "The syntax is: calculate [metric_name]",
              ],
            },
          },
          {
            type: "quiz",
            content: {
              question:
                "Why is it important to evaluate a model on test data rather than training data?",
              options: [
                "Because test data is easier to process",
                "To measure how well the model generalizes to new, unseen data",
                "Because training data is usually corrupted",
                "To make the evaluation process faster",
              ],
              correctAnswer: 1,
              explanation:
                "Evaluating on test data (data not seen during training) helps measure how well the model generalizes to new, unseen data. This is crucial because the ultimate goal is to make accurate predictions on new data in real-world scenarios.",
            },
          },
        ],
      },
      {
        id: "saving-loading-models",
        title: "Saving and Loading Models",
        description:
          "Learn how to save your trained models and load them for later use.",
        complexity: "beginner",
        estimatedTime: 15,
        content: [
          {
            type: "text",
            content:
              "# Saving and Loading Models\n\nAfter training a valuable model, you'll want to save it for future use without retraining. In this lesson, we'll learn how to save and load models in Nova.\n\n## Why Save Models?\n\n- **Reuse**: Use the trained model for predictions without retraining\n- **Sharing**: Share the model with others\n- **Deployment**: Use the model in production applications\n- **Checkpointing**: Save model progress during long training processes",
          },
          {
            type: "code",
            content: {
              nova: '# Saving a model in Nova\nsave model digit_recognizer to "models/digit_recognizer.nova"\n\n# Loading a model in Nova\nload model from "models/digit_recognizer.nova" as loaded_recognizer',
              pytorch:
                'import torch\nimport torch.nn as nn\n\n# Assuming digit_recognizer is defined and trained\n\n# Save the model\ntorch.save(digit_recognizer.state_dict(), "models/digit_recognizer.pth")\n\n# Load the model\n# First, create a model instance with the same architecture\nloaded_recognizer = DigitRecognizer()  # Assuming this class is defined elsewhere\n\n# Then load the state dictionary\nloaded_recognizer.load_state_dict(torch.load("models/digit_recognizer.pth"))\n\n# Put the model in evaluation mode\nloaded_recognizer.eval()',
            },
            explanation:
              "This code demonstrates how to save and load models in Nova and PyTorch. Notice how Nova simplifies the process.",
          },
          {
            type: "text",
            content:
              "## What Gets Saved?\n\nWhen you save a model, various components are stored:\n\n- **Model Architecture**: The structure of the network\n- **Parameter Values**: Weights and biases learned during training\n- **Optimizer State** (optional): State of the optimizer, useful for resuming training\n- **Training History** (optional): Loss values and metrics during training",
          },
          {
            type: "interactive-code",
            content: {
              nova: "# Complete the code to save and load a model\n\ncreate processing pipeline image_classifier:\n    add transformation stage convolutional with 3 channels, 16 filters, and 3x3 kernel\n    apply relu activation\n    add pooling stage max_pooling with 2x2 size and 2x2 stride\n    add transformation stage fully_connected with 16*13*13 inputs and 10 outputs\n    apply softmax activation\n\ntrain image_classifier on cifar10_data_stream:\n    measure error using cross_entropy\n    improve using adam with learning rate 0.001\n    repeat for 5 learning cycles\n\n# Add code to save the model\n\n\n# Add code to load the model with a different name",
              pytorch: "",
              solution:
                'create processing pipeline image_classifier:\n    add transformation stage convolutional with 3 channels, 16 filters, and 3x3 kernel\n    apply relu activation\n    add pooling stage max_pooling with 2x2 size and 2x2 stride\n    add transformation stage fully_connected with 16*13*13 inputs and 10 outputs\n    apply softmax activation\n\ntrain image_classifier on cifar10_data_stream:\n    measure error using cross_entropy\n    improve using adam with learning rate 0.001\n    repeat for 5 learning cycles\n\n# Save the model\nsave model image_classifier to "models/cifar_classifier.nova"\n\n# Load the model with a different name\nload model from "models/cifar_classifier.nova" as loaded_classifier',
              instructions:
                "Complete the code by adding commands to save the trained image_classifier and then load it with a new name (loaded_classifier).",
              hints: [
                "Use save model [model_name] to [file_path]",
                "Use load model from [file_path] as [new_model_name]",
                "Choose appropriate file names with the .nova extension",
                "Make sure the file paths are in quotes",
              ],
            },
          },
          {
            type: "quiz",
            content: {
              question:
                "What is the primary benefit of saving a trained model?",
              options: [
                "It makes the model more accurate",
                "It allows you to use the model without retraining it every time",
                "It automatically improves model performance",
                "It increases the model's learning rate",
              ],
              correctAnswer: 1,
              explanation:
                "The primary benefit of saving a trained model is that it allows you to use the model for predictions without having to retrain it every time. This saves time and computational resources, especially for large models that take a long time to train.",
            },
          },
        ],
      },
      {
        id: "making-predictions",
        title: "Making Predictions with Nova",
        description:
          "Learn how to use your trained models to make predictions on new data.",
        complexity: "beginner",
        estimatedTime: 20,
        content: [
          {
            type: "text",
            content:
              "# Making Predictions with Nova\n\nAfter training and evaluating a model, the ultimate goal is to use it to make predictions on new, unseen data. In this lesson, we'll learn how to use Nova models to make predictions.\n\n## Prediction Process\n\n1. **Prepare Input Data**: Format new data to match the model's expected input\n2. **Run the Model**: Pass the data through the model\n3. **Process Output**: Interpret the model's predictions\n4. **Apply Decision Logic**: Take actions based on predictions",
          },
          {
            type: "code",
            content: {
              nova: '# Making predictions with a trained model in Nova\ncreate feature grid input_image with shape [1, 28, 28, 1] from "sample_digit.png"\n\n# Get prediction\npredict using digit_recognizer on input_image\n\n# Get prediction with probability\npredict using digit_recognizer on input_image with probabilities',
              pytorch:
                'import torch\nfrom PIL import Image\nimport torchvision.transforms as transforms\nimport numpy as np\n\n# Load and preprocess the image\nimage = Image.open("sample_digit.png").convert("L")  # Convert to grayscale\ntransform = transforms.Compose([\n    transforms.Resize((28, 28)),\n    transforms.ToTensor(),\n    transforms.Normalize((0.1307,), (0.3081,))\n])\ninput_image = transform(image).unsqueeze(0)  # Add batch dimension\n\n# Make prediction\nmodel = digit_recognizer\nmodel.eval()\nwith torch.no_grad():\n    output = model(input_image)\n    \n    # Get the predicted class\n    _, predicted_class = torch.max(output, 1)\n    print(f"Predicted digit: {predicted_class.item()}")\n    \n    # Get probabilities\n    probabilities = torch.nn.functional.softmax(output, dim=1)\n    print(f"Probabilities: {probabilities.squeeze().numpy()}")',
            },
            explanation:
              "This code shows how to make predictions with a trained model in Nova and PyTorch. Nova's syntax is much simpler and more intuitive.",
          },
          {
            type: "text",
            content:
              "## Types of Predictions\n\n- **Classification**: Predicting a category or class (e.g., digit recognition)\n- **Regression**: Predicting a continuous value (e.g., house price prediction)\n- **Probability**: Predicting the likelihood of each possible outcome\n- **Sequence**: Predicting a sequence of values (e.g., text generation)\n- **Structured**: Predicting structured outputs (e.g., image segmentation)",
          },
          {
            type: "interactive-code",
            content: {
              nova: '# Complete the code to make predictions with a sentiment classifier\n\n# Assume the model is already trained\nload model from "models/sentiment_classifier.nova" as sentiment_model\n\n# Create input features for two sample reviews\ncreate feature grid positive_review_features with shape [1, 100] from "Great product, highly recommend it!"\ncreate feature grid negative_review_features with shape [1, 100] from "Terrible quality, don\'t waste your money."\n\n# Make predictions\n# Add code to predict sentiment for the positive review\n\n\n# Add code to predict sentiment for the negative review with probabilities',
              pytorch: "",
              solution:
                'load model from "models/sentiment_classifier.nova" as sentiment_model\n\ncreate feature grid positive_review_features with shape [1, 100] from "Great product, highly recommend it!"\ncreate feature grid negative_review_features with shape [1, 100] from "Terrible quality, don\'t waste your money."\n\n# Make prediction for positive review\npredict using sentiment_model on positive_review_features\n\n# Make prediction for negative review with probabilities\npredict using sentiment_model on negative_review_features with probabilities',
              instructions:
                "Complete the code to make predictions on the positive and negative reviews. For the negative review, also show the probabilities.",
              hints: [
                "Use predict using [model_name] on [input_data]",
                'To get probabilities, add "with probabilities" at the end',
                "The syntax is similar for both predictions, but with different inputs",
                "Make sure to use the correct model and input names",
              ],
            },
          },
          {
            type: "quiz",
            content: {
              question:
                "Why might you want to get prediction probabilities rather than just the predicted class?",
              options: [
                "Because probabilities are always more accurate",
                "To understand the model's confidence in its prediction",
                "Because it's faster to compute",
                "Probabilities are required for model evaluation",
              ],
              correctAnswer: 1,
              explanation:
                "Getting prediction probabilities allows you to understand the model's confidence in its prediction. A high probability (e.g., 0.98) suggests high confidence, while a lower probability (e.g., 0.51) indicates the model is less certain. This information can be crucial for decision-making, especially in critical applications.",
            },
          },
        ],
      },
    ],
  },
  {
    id: "computer-vision",
    title: "Computer Vision with Nova",
    description:
      "Create image classification and object detection models with Nova's simplified syntax.",
    image: "/images/course-cv.jpg",
    tags: ["intermediate", "computer vision", "CNNs"],
    lessons: [
      {
        id: "intro-to-cnns",
        title: "Introduction to Convolutional Networks",
        description:
          "Learn how to build and train convolutional neural networks for image processing.",
        complexity: "intermediate",
        estimatedTime: 30,
        content: [
          {
            type: "text",
            content:
              "# Introduction to Convolutional Networks\n\nConvolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images. In this lesson, you will learn how to define CNNs using Nova syntax.",
          },
          {
            type: "code",
            content: {
              nova: "create processing pipeline image_classifier:\n    add transformation stage convolutional with 3 channels, 16 filters, and 3x3 kernel\n    apply relu activation\n    add pooling stage max_pooling with 2x2 size and 2x2 stride\n    add transformation stage fully_connected with 16*13*13 inputs and 10 outputs\n    apply softmax activation",
              pytorch:
                "import torch\nimport torch.nn as nn\n\nclass ImageClassifier(nn.Module):\n    def __init__(self):\n        super(ImageClassifier, self).__init__()\n        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)\n        self.relu = nn.ReLU()\n        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)\n        self.fc = nn.Linear(16*13*13, 10)\n        self.softmax = nn.Softmax(dim=1)\n        \n    def forward(self, x):\n        x = self.conv1(x)\n        x = self.relu(x)\n        x = self.pool(x)\n        x = x.view(-1, 16*13*13)\n        x = self.fc(x)\n        x = self.softmax(x)\n        return x",
            },
            explanation:
              "This code creates a simple CNN for image classification with one convolutional layer followed by pooling and a fully connected layer.",
          },
        ],
      },
    ],
  },
  {
    id: "advanced-techniques",
    title: "Advanced ML Techniques",
    description:
      "Learn transfer learning, fine-tuning, and other advanced techniques using Nova.",
    image: "/images/course-advanced.jpg",
    tags: ["advanced", "transfer learning", "fine-tuning"],
    lessons: [
      {
        id: "transfer-learning",
        title: "Transfer Learning with Pre-trained Models",
        description:
          "Learn how to leverage pre-trained models for your specific tasks.",
        complexity: "advanced",
        estimatedTime: 35,
        content: [
          {
            type: "text",
            content:
              "# Transfer Learning with Nova\n\nTransfer learning allows you to use knowledge from pre-trained models on new tasks. This lesson will show you how to use transfer learning with Nova.",
          },
          {
            type: "code",
            content: {
              nova: 'load pretrained model "resnet18"\n\n# Freeze the early layers\nmark early_layers as fixed from layer 1 to layer 4\n\n# Replace the classifier\nreplace classifier with:\n    add transformation stage fully_connected with 512 inputs and 128 outputs\n    apply relu activation\n    add transformation stage fully_connected with 128 inputs and 5 outputs\n    apply softmax activation',
              pytorch:
                "import torch\nimport torchvision.models as models\nimport torch.nn as nn\n\n# Load pre-trained ResNet18\nmodel = models.resnet18(pretrained=True)\n\n# Freeze early layers\nfor idx, param in enumerate(list(model.parameters())[:4]):\n    param.requires_grad = False\n\n# Replace the classifier\nmodel.fc = nn.Sequential(\n    nn.Linear(512, 128),\n    nn.ReLU(),\n    nn.Linear(128, 5),\n    nn.Softmax(dim=1)\n)",
            },
            explanation:
              "This code demonstrates how to load a pre-trained ResNet18 model, freeze the early layers, and replace the classifier with a custom one for a specific task.",
          },
        ],
      },
    ],
  },
];

const CoursesPage: React.FC = () => {
  const { courseId } = useParams<{ courseId: string }>();
  const navigate = useNavigate();
  const { progress, isLessonCompleted } = useProgress();
  const [selectedCourse, setSelectedCourse] = useState<Course | null>(null);

  useEffect(() => {
    if (courseId) {
      const course = coursesData.find((c) => c.id === courseId);
      setSelectedCourse(course || null);

      if (!course) {
        navigate("/courses", { replace: true });
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
    navigate("/courses");
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case "beginner":
        return "success";
      case "intermediate":
        return "warning";
      case "advanced":
        return "error";
      default:
        return "default";
    }
  };

  const calculateCourseProgress = (course: Course): number => {
    if (!course.lessons.length) return 0;

    const completedCount = course.lessons.filter((lesson) =>
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
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  transition: "transform 0.2s, box-shadow 0.2s",
                  "&:hover": {
                    transform: "translateY(-4px)",
                    boxShadow: "0 8px 16px rgba(0,0,0,0.1)",
                  },
                }}
                onClick={() => navigate(`/courses/${course.id}`)}
              >
                <CardMedia
                  component="img"
                  height="140"
                  image={
                    course.image ||
                    "https://via.placeholder.com/300x140?text=Nova+Course"
                  }
                  alt={course.title}
                />

                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h5" component="h2" gutterBottom>
                    {course.title}
                  </Typography>

                  <Typography variant="body2" color="text.secondary" paragraph>
                    {course.description}
                  </Typography>

                  <Box
                    sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 2 }}
                  >
                    {course.tags.map((tag) => (
                      <Chip
                        key={tag}
                        label={tag}
                        size="small"
                        color={
                          tag === "beginner"
                            ? "success"
                            : tag === "intermediate"
                            ? "warning"
                            : tag === "advanced"
                            ? "error"
                            : "default"
                        }
                        variant="outlined"
                      />
                    ))}
                  </Box>

                  <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
                    <SchoolIcon
                      fontSize="small"
                      sx={{ mr: 1, color: "primary.main" }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      {course.lessons.length}{" "}
                      {course.lessons.length === 1 ? "Lesson" : "Lessons"}
                    </Typography>
                  </Box>
                </CardContent>

                <Box sx={{ p: 2, pt: 0 }}>
                  <Box sx={{ width: "100%", mb: 1 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mb: 0.5,
                      }}
                    >
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
                    {progress > 0 ? "Continue Learning" : "Start Course"}
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
          <Box
            sx={{ display: "flex", flexDirection: { xs: "column", md: "row" } }}
          >
            <CardMedia
              component="img"
              sx={{
                width: { xs: "100%", md: 340 },
                height: { xs: 200, md: "auto" },
              }}
              image={
                selectedCourse.image ||
                "https://via.placeholder.com/600x400?text=Nova+Course"
              }
              alt={selectedCourse.title}
            />

            <CardContent sx={{ flex: "1 0 auto", p: 3 }}>
              <Typography variant="h4" component="h1" gutterBottom>
                {selectedCourse.title}
              </Typography>

              <Typography variant="body1" paragraph>
                {selectedCourse.description}
              </Typography>

              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 2 }}>
                {selectedCourse.tags.map((tag) => (
                  <Chip
                    key={tag}
                    label={tag}
                    size="small"
                    color={
                      tag === "beginner"
                        ? "success"
                        : tag === "intermediate"
                        ? "warning"
                        : tag === "advanced"
                        ? "error"
                        : "default"
                    }
                  />
                ))}
              </Box>

              <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
                <SchoolIcon fontSize="small" sx={{ mr: 1 }} />
                <Typography variant="body2" color="text.secondary">
                  {selectedCourse.lessons.length}{" "}
                  {selectedCourse.lessons.length === 1 ? "Lesson" : "Lessons"}
                </Typography>
              </Box>

              <Box sx={{ width: "100%", mb: 2 }}>
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    mb: 0.5,
                  }}
                >
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
          <List sx={{ width: "100%" }}>
            {selectedCourse.lessons.map((lesson, index) => {
              const isCompleted = isLessonCompleted(lesson.id);

              return (
                <React.Fragment key={lesson.id}>
                  {index > 0 && <Divider component="li" />}
                  <ListItem
                    alignItems="flex-start"
                    sx={{
                      py: 2,
                      cursor: "pointer",
                      "&:hover": {
                        bgcolor: "action.hover",
                      },
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
                          <Typography
                            variant="body2"
                            color="text.secondary"
                            paragraph
                          >
                            {lesson.description}
                          </Typography>

                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              mt: 1,
                            }}
                          >
                            <Chip
                              size="small"
                              label={lesson.complexity}
                              color={
                                getComplexityColor(lesson.complexity) as any
                              }
                              sx={{ mr: 2 }}
                            />

                            <Box sx={{ display: "flex", alignItems: "center" }}>
                              <TimerIcon fontSize="small" sx={{ mr: 0.5 }} />
                              <Typography
                                variant="body2"
                                color="text.secondary"
                              >
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
                      {isCompleted ? "Review" : "Start"}
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
