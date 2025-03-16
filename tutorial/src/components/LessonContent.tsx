import React, { useState } from 'react';
import { Box, Button, Typography, Card, CardContent, Stepper, Step, StepLabel, Paper, Alert } from '@mui/material';
import ReactMarkdown from 'react-markdown';
import { Lesson, LessonContent as LessonContentType, CodeExample, Quiz, InteractiveCodeBlock } from '../types';
import CodeEditor from './CodeEditor';
import { useProgress } from '../context/ProgressContext';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import QuizComponent from './Quiz';

interface LessonContentProps {
  lesson: Lesson;
  onComplete: () => void;
  onBack?: () => void;
}

const LessonContent: React.FC<LessonContentProps> = ({ lesson, onComplete, onBack }) => {
  const { progress, completeLesson, addCodeSubmission, setCurrentLesson } = useProgress();
  const [activeStep, setActiveStep] = useState(progress.currentStep || 0);
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [codeSubmissions, setCodeSubmissions] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  const isLastStep = activeStep === lesson.content.length - 1;
  const isCompleted = progress.completedLessons.includes(lesson.id);

  const handleNext = () => {
    const nextStep = activeStep + 1;
    if (nextStep < lesson.content.length) {
      setActiveStep(nextStep);
      setCurrentLesson(lesson.id, nextStep);
    } else {
      completeLesson(lesson.id, lesson.complexity);
      onComplete();
    }
  };

  const handleBack = () => {
    if (activeStep > 0) {
      setActiveStep(activeStep - 1);
      setCurrentLesson(lesson.id, activeStep - 1);
    } else if (onBack) {
      onBack();
    }
  };

  const handleQuizSubmit = (quizId: string, answer: number, isCorrect: boolean) => {
    setAnswers(prev => ({ ...prev, [quizId]: answer }));
    if (isCorrect) {
      setError(null);
    } else {
      setError('That answer is incorrect. Please try again.');
    }
  };

  const handleCodeSubmit = (blockId: string, code: string) => {
    setCodeSubmissions(prev => ({ ...prev, [blockId]: code }));
    addCodeSubmission(blockId, code);
  };

  const canProceed = () => {
    const currentContent = lesson.content[activeStep];
    
    if (currentContent.type === 'quiz') {
      const quiz = currentContent.content as Quiz;
      const quizId = `${lesson.id}-quiz-${activeStep}`;
      return answers[quizId] === quiz.correctAnswer;
    }
    
    if (currentContent.type === 'interactive-code') {
      const codeBlock = currentContent.content as InteractiveCodeBlock;
      const blockId = `${lesson.id}-code-${activeStep}`;
      
      // For simplicity, allow proceeding if code is submitted
      // In a real app, you might want to validate the code 
      return blockId in codeSubmissions;
    }
    
    return true;
  };

  const renderContent = (content: LessonContentType) => {
    switch (content.type) {
      case 'text':
        return (
          <Paper elevation={1} sx={{ p: 3, mb: 3 }}>
            <ReactMarkdown>{content.content as string}</ReactMarkdown>
            {content.explanation && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Note:
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {content.explanation}
                </Typography>
              </Box>
            )}
          </Paper>
        );
        
      case 'code':
        return (
          <Box sx={{ mb: 3 }}>
            <CodeEditor 
              initialCode={content.content as CodeExample} 
              readOnly={true}
            />
            {content.explanation && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  {content.explanation}
                </Typography>
              </Box>
            )}
          </Box>
        );
        
      case 'interactive-code':
        const interactiveBlock = content.content as InteractiveCodeBlock;
        const blockId = `${lesson.id}-code-${activeStep}`;
        
        return (
          <Box sx={{ mb: 3 }}>
            <CodeEditor 
              initialCode={interactiveBlock}
              readOnly={false}
              onCodeChange={(code) => handleCodeSubmit(blockId, code)}
              instructions={interactiveBlock.instructions}
              hints={interactiveBlock.hints}
            />
            {content.explanation && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  {content.explanation}
                </Typography>
              </Box>
            )}
          </Box>
        );
        
      case 'quiz':
        const quiz = content.content as Quiz;
        const quizId = `${lesson.id}-quiz-${activeStep}`;
        
        return (
          <Box sx={{ mb: 3 }}>
            <QuizComponent 
              quiz={quiz}
              onSubmit={(answer, isCorrect) => handleQuizSubmit(quizId, answer, isCorrect)}
              selectedAnswer={answers[quizId]}
            />
          </Box>
        );
        
      case 'image':
        return (
          <Box sx={{ mb: 3, textAlign: 'center' }}>
            <img 
              src={content.content as string} 
              alt={content.explanation || 'Lesson image'} 
              style={{ maxWidth: '100%', maxHeight: '400px' }} 
            />
            {content.explanation && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {content.explanation}
              </Typography>
            )}
          </Box>
        );
        
      default:
        return (
          <Typography variant="body1">
            Unsupported content type
          </Typography>
        );
    }
  };

  return (
    <Box>
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h4" gutterBottom>
            {lesson.title}
            {isCompleted && (
              <CheckCircleIcon color="success" sx={{ ml: 1, verticalAlign: 'middle' }} />
            )}
          </Typography>
          
          <Typography variant="body2" color="text.secondary" gutterBottom>
            {lesson.description}
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Estimated time: {lesson.estimatedTime} min
            </Typography>
            <Typography 
              variant="body2" 
              color="text.secondary" 
              sx={{ 
                ml: 2, 
                px: 1, 
                py: 0.5, 
                borderRadius: 1,
                bgcolor: lesson.complexity === 'beginner' ? 'success.light' : 
                         lesson.complexity === 'intermediate' ? 'warning.light' : 'error.light'
              }}
            >
              {lesson.complexity}
            </Typography>
          </Box>
        </CardContent>
      </Card>
      
      <Box sx={{ mb: 4 }}>
        <Stepper activeStep={activeStep} alternativeLabel>
          {lesson.content.map((_, index) => (
            <Step key={index} completed={index < activeStep}>
              <StepLabel />
            </Step>
          ))}
        </Stepper>
      </Box>
      
      <Box sx={{ mb: 4 }}>
        {renderContent(lesson.content[activeStep])}
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button 
          onClick={handleBack}
          startIcon={<ArrowBackIcon />}
        >
          Back
        </Button>
        
        <Button 
          variant="contained" 
          color="primary" 
          onClick={handleNext}
          disabled={!canProceed()}
          endIcon={<ArrowForwardIcon />}
        >
          {isLastStep ? 'Complete' : 'Next'}
        </Button>
      </Box>
    </Box>
  );
};

export default LessonContent;