import React from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Paper, 
  Grid, 
  Divider, 
  Card,
  CardContent,
  Link,
  Button
} from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import SchoolIcon from '@mui/icons-material/School';
import WebIcon from '@mui/icons-material/Web';
import { useNavigate } from 'react-router-dom';

const AboutPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 6, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom>
          About Nova
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ maxWidth: '800px', mx: 'auto' }}>
          Nova is a natural language interface for creating deep learning models with PyTorch, designed to make machine learning more accessible.
        </Typography>
      </Box>

      <Paper elevation={2} sx={{ p: 4, mb: 6 }}>
        <Typography variant="h4" gutterBottom>
          What is Nova?
        </Typography>
        
        <Typography variant="body1" paragraph>
          Nova is a structured natural language pseudocode system that bridges the gap between everyday programming concepts and PyTorch's machine learning operations. It leverages the strengths of modern AI assistants while making deep learning more accessible to programmers of all levels.
        </Typography>
        
        <Typography variant="body1" paragraph>
          The primary goal of Nova is to address the challenge of vague terminology and nomenclature in machine learning, making it easier for programmers without extensive mathematical or ML background to create effective machine learning models.
        </Typography>
        
        <Box sx={{ my: 4 }}>
          <Divider />
        </Box>
        
        <Typography variant="h5" gutterBottom>
          Core Components
        </Typography>
        
        <Grid container spacing={4} sx={{ mb: 4 }}>
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Language Guide
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  A comprehensive definition of our pseudocode language, including vocabulary, syntax rules, and semantic mappings to PyTorch concepts.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Interpreter
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  A system that translates Nova instructions into equivalent PyTorch code, providing explanations and handling implementation details.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Examples Collection
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  A rich set of examples showing common ML tasks expressed in Nova with equivalent PyTorch implementations and step-by-step explanations.
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>
      
      <Paper elevation={2} sx={{ p: 4, mb: 6 }}>
        <Typography variant="h4" gutterBottom>
          Key Concepts
        </Typography>
        
        <Typography variant="body1" paragraph>
          Nova uses intuitive terminology to represent complex machine learning concepts:
        </Typography>
        
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom>
              Data Concepts
            </Typography>
            <Box component="ul" sx={{ pl: 2 }}>
              <li>
                <Typography variant="body2">
                  <strong>Data Collection</strong> = Dataset (e.g., MNIST)
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Sample</strong> = Individual data point
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Feature Grid</strong> = Tensor (with dimensions and types)
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Data Stream</strong> = DataLoader
                </Typography>
              </li>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom>
              Processing Concepts
            </Typography>
            <Box component="ul" sx={{ pl: 2 }}>
              <li>
                <Typography variant="body2">
                  <strong>Processing Pipeline</strong> = Neural network model
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Transformation Stage</strong> = Layer
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Connection Strengths</strong> = Weights and biases
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Activation Pattern</strong> = Activation function results
                </Typography>
              </li>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Typography variant="h6" gutterBottom>
              Learning Concepts
            </Typography>
            <Box component="ul" sx={{ pl: 2 }}>
              <li>
                <Typography variant="body2">
                  <strong>Error Measure</strong> = Loss function
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Improvement Strategy</strong> = Optimizer
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Learning Cycle</strong> = Epoch
                </Typography>
              </li>
              <li>
                <Typography variant="body2">
                  <strong>Improvement Step</strong> = Backpropagation and update
                </Typography>
              </li>
            </Box>
          </Grid>
        </Grid>
      </Paper>
      
      <Paper elevation={2} sx={{ p: 4, mb: 6 }}>
        <Typography variant="h4" gutterBottom>
          Example Translation
        </Typography>
        
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Nova Code:
          </Typography>
          <Box 
            component="pre" 
            sx={{ 
              p: 2, 
              backgroundColor: '#f5f5f5', 
              borderRadius: 1,
              overflowX: 'auto'
            }}
          >
{`create processing pipeline digit_recognizer:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
    apply softmax activation

train digit_recognizer on mnist_data_stream:
    measure error using cross_entropy
    improve using gradient_descent with learning rate 0.01
    repeat for 10 learning cycles`}
          </Box>
        </Box>
        
        <Box>
          <Typography variant="h6" gutterBottom>
            PyTorch Equivalent:
          </Typography>
          <Box 
            component="pre" 
            sx={{ 
              p: 2, 
              backgroundColor: '#f5f5f5', 
              borderRadius: 1,
              overflowX: 'auto'
            }}
          >
{`import torch
import torch.nn as nn
import torch.optim as optim

# Define the model
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
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

model = DigitRecognizer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for data, target in mnist_data_stream:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()`}
          </Box>
        </Box>
      </Paper>
      
      <Box sx={{ mb: 6 }}>
        <Typography variant="h4" gutterBottom>
          Learn More
        </Typography>
        
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Button 
              variant="outlined" 
              fullWidth
              startIcon={<SchoolIcon />}
              onClick={() => navigate('/courses')}
              sx={{ py: 2 }}
            >
              Start Learning
            </Button>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Button 
              variant="outlined" 
              fullWidth
              startIcon={<WebIcon />}
              component={Link}
              href="https://nova-ml.github.io"
              target="_blank"
              rel="noopener noreferrer"
              sx={{ py: 2 }}
            >
              Documentation
            </Button>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Button 
              variant="outlined" 
              fullWidth
              startIcon={<GitHubIcon />}
              component={Link}
              href="https://github.com/nova-ml/nova"
              target="_blank"
              rel="noopener noreferrer"
              sx={{ py: 2 }}
            >
              GitHub Repository
            </Button>
          </Grid>
        </Grid>
      </Box>
      
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="body2" color="text.secondary">
          Nova is an open-source project. Contributions and feedback are welcome!
        </Typography>
      </Box>
    </Container>
  );
};

export default AboutPage;