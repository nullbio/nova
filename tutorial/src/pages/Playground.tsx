import React, { useState } from 'react';
import { Container, Typography, Box, Paper, Alert, Button } from '@mui/material';
import CodeEditor from '../components/CodeEditor';
import { CodeExample, NovaInterpretationResult } from '../types';
import { useInterpreter } from '../context/InterpreterContext';
import InfoIcon from '@mui/icons-material/Info';

const initialCode: CodeExample = {
  nova: `# Welcome to the Nova Playground
# Write and test your Nova code here

# Example: Create a simple neural network
create processing pipeline simple_model:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
    apply softmax activation

# You can also train models, load data, etc.
# Check the examples in the tutorial courses for more ideas!
`,
  pytorch: `# The equivalent PyTorch code will appear here when you run your Nova code
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
`
};

const Playground: React.FC = () => {
  const { isConnected } = useInterpreter();
  const [lastResult, setLastResult] = useState<NovaInterpretationResult | null>(null);
  const [showResults, setShowResults] = useState(false);

  const handleCodeRun = (result: NovaInterpretationResult) => {
    setLastResult(result);
    setShowResults(true);
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Nova Code Playground
      </Typography>
      
      <Typography variant="body1" paragraph>
        Experiment with Nova code freely in this playground. Write your Nova code, run it, and see the equivalent PyTorch code and execution results.
      </Typography>
      
      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="body1">
            Interpreter not connected. Your code will not be executed.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Check Settings to configure the interpreter connection.
          </Typography>
        </Alert>
      )}
      
      <Box sx={{ mb: 4 }}>
        <CodeEditor 
          initialCode={initialCode}
          onCodeRun={handleCodeRun}
        />
      </Box>
      
      {showResults && lastResult && (
        <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <InfoIcon color="info" sx={{ mr: 1 }} />
            <Typography variant="h6">Execution Results</Typography>
          </Box>
          
          {lastResult.executionTime && (
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Execution time: {lastResult.executionTime.toFixed(2)}ms
            </Typography>
          )}
          
          {lastResult.error ? (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="subtitle2">Error:</Typography>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {lastResult.error}
              </pre>
            </Alert>
          ) : (
            lastResult.output && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2">Output:</Typography>
                <pre style={{ 
                  margin: 0, 
                  padding: '10px', 
                  background: '#f5f5f5', 
                  borderRadius: '4px',
                  maxHeight: '200px',
                  overflow: 'auto'
                }}>
                  {lastResult.output}
                </pre>
              </Box>
            )
          )}
          
          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button 
              variant="outlined" 
              size="small" 
              onClick={() => setShowResults(false)}
            >
              Hide Results
            </Button>
          </Box>
        </Paper>
      )}
      
      <Paper elevation={1} sx={{ p: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Tips:
        </Typography>
        <ul>
          <li>Use the <strong>Run</strong> button to interpret your Nova code</li>
          <li>View the equivalent PyTorch code in the PyTorch tab</li>
          <li>Your code will be executed if connected to an interpreter</li>
          <li>Check the Courses section for guided examples and tutorials</li>
        </ul>
      </Paper>
    </Container>
  );
};

export default Playground;