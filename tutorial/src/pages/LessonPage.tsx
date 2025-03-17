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

// Import coursesData from the CoursesPage
import { coursesData } from './CoursesPage';

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