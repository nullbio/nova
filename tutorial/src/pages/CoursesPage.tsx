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