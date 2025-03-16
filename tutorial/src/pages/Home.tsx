import React from 'react';
import { 
  Container,
  Typography,
  Box,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  CardMedia,
  Paper,
  Divider,
  Chip
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useProgress } from '../context/ProgressContext';
import SchoolIcon from '@mui/icons-material/School';
import CodeIcon from '@mui/icons-material/Code';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const Home: React.FC = () => {
  const navigate = useNavigate();
  const { progress } = useProgress();
  
  const featuredCourses = [
    {
      id: 'intro-to-nova',
      title: 'Introduction to Nova',
      description: 'Learn the basics of the Nova language and how to create simple neural networks.',
      image: '/images/course-intro.jpg',
      complexity: 'beginner',
      estimatedTime: 30,
      completedLessons: 2,
      totalLessons: 5,
    },
    {
      id: 'computer-vision',
      title: 'Computer Vision with Nova',
      description: 'Create image classification and object detection models with Nova\'s simplified syntax.',
      image: '/images/course-cv.jpg',
      complexity: 'intermediate', 
      estimatedTime: 45,
      completedLessons: 0,
      totalLessons: 4,
    },
    {
      id: 'advanced-techniques',
      title: 'Advanced ML Techniques',
      description: 'Learn transfer learning, fine-tuning, and other advanced techniques using Nova.',
      image: '/images/course-advanced.jpg',
      complexity: 'advanced',
      estimatedTime: 60,
      completedLessons: 0,
      totalLessons: 6,
    }
  ];

  const handleCourseClick = (courseId: string) => {
    navigate(`/courses/${courseId}`);
  };

  const handleStartLearning = () => {
    navigate('/courses');
  };

  const handleGoToPlayground = () => {
    navigate('/playground');
  };

  const getCompletionPercentage = (completed: number, total: number) => {
    return Math.round((completed / total) * 100);
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

  return (
    <Container maxWidth="lg">
      {/* Hero Section */}
      <Box 
        sx={{ 
          textAlign: 'center', 
          py: 6,
          mb: 6,
          background: 'linear-gradient(145deg, rgba(80, 70, 228, 0.1) 0%, rgba(255, 64, 129, 0.05) 100%)',
          borderRadius: 4,
          boxShadow: 'inset 0 0 20px rgba(80, 70, 228, 0.05)'
        }}
      >
        <Typography variant="h2" component="h1" gutterBottom>
          Learn Nova Interactively
        </Typography>
        
        <Typography variant="h5" color="text.secondary" paragraph sx={{ maxWidth: '80%', mx: 'auto' }}>
          Master the most intuitive way to build deep learning models with step-by-step tutorials and interactive exercises.
        </Typography>
        
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center', gap: 2 }}>
          <Button 
            variant="contained" 
            size="large"
            startIcon={<SchoolIcon />}
            onClick={handleStartLearning}
          >
            Start Learning
          </Button>
          
          <Button 
            variant="outlined" 
            size="large"
            startIcon={<CodeIcon />}
            onClick={handleGoToPlayground}
          >
            Try Playground
          </Button>
        </Box>
        
        {progress.completedLessons.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Chip 
              icon={<CheckCircleIcon />}
              label={`${progress.completedLessons.length} Lessons Completed`}
              color="success"
              variant="outlined"
            />
          </Box>
        )}
      </Box>
      
      {/* Course Cards Section */}
      <Typography variant="h4" component="h2" gutterBottom>
        Featured Courses
      </Typography>
      
      <Typography variant="body1" paragraph color="text.secondary">
        Begin your Nova journey with these carefully designed courses, from basic concepts to advanced techniques.
      </Typography>
      
      <Grid container spacing={4} sx={{ mb: 6 }}>
        {featuredCourses.map((course) => (
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
              onClick={() => handleCourseClick(course.id)}
            >
              <CardMedia
                component="img"
                height="140"
                image={course.image || "https://via.placeholder.com/300x140?text=Nova+Course"}
                alt={course.title}
              />
              
              <CardContent sx={{ flexGrow: 1 }}>
                <Box sx={{ mb: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Chip 
                    label={course.complexity} 
                    size="small" 
                    color={getComplexityColor(course.complexity) as any}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {course.estimatedTime} min
                  </Typography>
                </Box>
                
                <Typography variant="h5" component="h3" gutterBottom>
                  {course.title}
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  {course.description}
                </Typography>
              </CardContent>
              
              <Divider />
              
              <CardActions>
                <Box sx={{ width: '100%', px: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                    <Typography variant="body2" color="text.secondary">
                      Progress: {getCompletionPercentage(course.completedLessons, course.totalLessons)}%
                    </Typography>
                    
                    <Button 
                      size="small" 
                      color="primary" 
                      sx={{ ml: 1 }}
                      endIcon={<PlayArrowIcon />}
                    >
                      Continue
                    </Button>
                  </Box>
                </Box>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
      
      {/* Features Section */}
      <Paper elevation={1} sx={{ p: 4, mb: 6 }}>
        <Typography variant="h4" component="h2" gutterBottom>
          Why Learn Nova?
        </Typography>
        
        <Grid container spacing={4} sx={{ mt: 2 }}>
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Simplified Syntax
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Nova's natural language-like syntax makes deep learning concepts more intuitive and easier to understand.
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="h6" gutterBottom>
                PyTorch Integration
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Learn how your Nova code translates to PyTorch, giving you a bridge to the powerful PyTorch ecosystem.
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Box>
              <Typography variant="h6" gutterBottom>
                Interactive Learning
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Hands-on exercises, live code execution, and step-by-step tutorials make learning engaging and effective.
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Getting Started Section */}
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" component="h2" gutterBottom>
          Ready to Get Started?
        </Typography>
        
        <Typography variant="body1" paragraph>
          Begin your journey with Nova by exploring our courses or trying out the code playground.
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button 
            variant="contained" 
            color="primary"
            onClick={handleStartLearning}
          >
            Explore Courses
          </Button>
          
          <Button 
            variant="outlined"
            onClick={handleGoToPlayground}
          >
            Try Playground
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default Home;