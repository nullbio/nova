import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Breadcrumbs, 
  Link, 
  Button,
  Paper,
  Alert,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle 
} from '@mui/material';
import { useParams, useNavigate, Link as RouterLink } from 'react-router-dom';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import LessonContent from '../components/LessonContent';
import { useProgress } from '../context/ProgressContext';
import { Course, Lesson } from '../types';

// Import mock course data from CoursesPage
// In a real application, you would fetch this from an API
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

const LessonPage: React.FC = () => {
  const { courseId, lessonId } = useParams<{ courseId: string; lessonId: string }>();
  const navigate = useNavigate();
  const { isLessonCompleted, completeLesson } = useProgress();
  const [course, setCourse] = useState<Course | null>(null);
  const [lesson, setLesson] = useState<Lesson | null>(null);
  const [nextLesson, setNextLesson] = useState<Lesson | null>(null);
  const [completionDialogOpen, setCompletionDialogOpen] = useState(false);

  useEffect(() => {
    if (courseId && lessonId) {
      const foundCourse = coursesData.find(c => c.id === courseId);
      
      if (foundCourse) {
        setCourse(foundCourse);
        
        const foundLesson = foundCourse.lessons.find(l => l.id === lessonId);
        
        if (foundLesson) {
          setLesson(foundLesson);
          
          // Find next lesson for navigation
          const currentIndex = foundCourse.lessons.findIndex(l => l.id === lessonId);
          const nextIndex = currentIndex + 1;
          
          if (nextIndex < foundCourse.lessons.length) {
            setNextLesson(foundCourse.lessons[nextIndex]);
          } else {
            setNextLesson(null);
          }
        } else {
          navigate(`/courses/${courseId}`, { replace: true });
        }
      } else {
        navigate('/courses', { replace: true });
      }
    }
  }, [courseId, lessonId, navigate]);

  const handleBackToCourse = () => {
    if (courseId) {
      navigate(`/courses/${courseId}`);
    } else {
      navigate('/courses');
    }
  };

  const handleLessonComplete = () => {
    // Only complete if lesson exists
    if (lesson) {
      // Pass the lesson complexity to award appropriate XP
      completeLesson(lesson.id, lesson.complexity);
    }
    setCompletionDialogOpen(true);
  };

  const handleNextLesson = () => {
    setCompletionDialogOpen(false);
    if (nextLesson && course) {
      navigate(`/courses/${course.id}/lessons/${nextLesson.id}`);
    } else if (course) {
      navigate(`/courses/${course.id}`);
    }
  };

  const handleBackToCourseFromDialog = () => {
    setCompletionDialogOpen(false);
    handleBackToCourse();
  };

  if (!lesson || !course) {
    return (
      <Container maxWidth="lg">
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <Typography variant="h5">Loading lesson...</Typography>
        </Box>
      </Container>
    );
  }

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
          <Link 
            component={RouterLink} 
            to={`/courses/${course.id}`} 
            underline="hover" 
            color="inherit"
          >
            {course.title}
          </Link>
          <Typography color="text.primary">{lesson.title}</Typography>
        </Breadcrumbs>
      </Box>
      
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={handleBackToCourse}
        sx={{ mb: 2 }}
      >
        Back to Course
      </Button>
      
      {isLessonCompleted(lesson.id) && (
        <Alert 
          icon={<CheckCircleIcon fontSize="inherit" />} 
          severity="success" 
          sx={{ mb: 3 }}
        >
          You have already completed this lesson. Feel free to review it again!
        </Alert>
      )}
      
      <LessonContent 
        lesson={lesson} 
        onComplete={handleLessonComplete}
        onBack={handleBackToCourse}
      />
      
      <Dialog
        open={completionDialogOpen}
        onClose={() => setCompletionDialogOpen(false)}
        aria-labelledby="lesson-completion-dialog"
      >
        <DialogTitle id="lesson-completion-dialog">
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CheckCircleIcon color="success" sx={{ mr: 1 }} />
            Lesson Completed!
          </Box>
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            Congratulations on completing the <strong>{lesson.title}</strong> lesson!
            {nextLesson ? (
              <Box component="span">
                {' '}Would you like to continue to the next lesson: <strong>{nextLesson.title}</strong>?
              </Box>
            ) : (
              <Box component="span">
                {' '}This was the last lesson in this course. You've completed the entire course!
              </Box>
            )}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleBackToCourseFromDialog}>
            Back to Course
          </Button>
          {nextLesson && (
            <Button 
              onClick={handleNextLesson} 
              variant="contained" 
              color="primary" 
              autoFocus
            >
              Continue to Next Lesson
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default LessonPage;