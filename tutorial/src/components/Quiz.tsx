import React, { useState } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  FormControl, 
  RadioGroup, 
  FormControlLabel, 
  Radio, 
  Button, 
  Alert,
  Collapse
} from '@mui/material';
import { Quiz as QuizType } from '../types';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';

interface QuizProps {
  quiz: QuizType;
  onSubmit: (selectedAnswer: number, isCorrect: boolean) => void;
  selectedAnswer?: number;
}

const Quiz: React.FC<QuizProps> = ({ quiz, onSubmit, selectedAnswer }) => {
  const [value, setValue] = useState<number | undefined>(selectedAnswer);
  const [submitted, setSubmitted] = useState(selectedAnswer !== undefined);
  const [error, setError] = useState<string | null>(null);
  
  const isCorrect = value === quiz.correctAnswer;
  
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setValue(parseInt(event.target.value, 10));
    setError(null);
  };
  
  const handleSubmit = () => {
    if (value === undefined) {
      setError('Please select an answer');
      return;
    }
    
    setSubmitted(true);
    onSubmit(value, isCorrect);
  };
  
  return (
    <Paper elevation={3} sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        {quiz.question}
      </Typography>
      
      <FormControl component="fieldset" sx={{ width: '100%', mt: 2 }}>
        <RadioGroup value={value} onChange={handleChange}>
          {quiz.options.map((option, index) => (
            <FormControlLabel 
              key={index} 
              value={index} 
              control={<Radio />} 
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <span>{option}</span>
                  {submitted && (
                    index === quiz.correctAnswer ? (
                      <CheckIcon color="success" sx={{ ml: 1 }} />
                    ) : index === value && index !== quiz.correctAnswer ? (
                      <CloseIcon color="error" sx={{ ml: 1 }} />
                    ) : null
                  )}
                </Box>
              }
              disabled={submitted}
              sx={{ 
                mb: 1, 
                p: 1, 
                borderRadius: 1,
                ...(submitted && index === quiz.correctAnswer && { 
                  bgcolor: 'success.light',
                  color: 'success.contrastText'
                }),
                ...(submitted && index === value && index !== quiz.correctAnswer && { 
                  bgcolor: 'error.light',
                  color: 'error.contrastText'
                })
              }}
            />
          ))}
        </RadioGroup>
      </FormControl>
      
      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
      
      <Collapse in={submitted} sx={{ mt: 2 }}>
        <Alert severity={isCorrect ? "success" : "error"}>
          <Typography variant="subtitle1" gutterBottom>
            {isCorrect ? 'Correct!' : 'Incorrect!'}
          </Typography>
          <Typography variant="body2">
            {quiz.explanation}
          </Typography>
        </Alert>
      </Collapse>
      
      {!submitted && (
        <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
          <Button 
            variant="contained" 
            color="primary" 
            onClick={handleSubmit}
          >
            Submit Answer
          </Button>
        </Box>
      )}
    </Paper>
  );
};

export default Quiz;