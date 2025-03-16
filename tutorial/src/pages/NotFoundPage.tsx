import React from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Button, 
  Paper,
  Grid
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import HomeIcon from '@mui/icons-material/Home';
import SchoolIcon from '@mui/icons-material/School';
import CodeIcon from '@mui/icons-material/Code';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="md">
      <Paper 
        elevation={3} 
        sx={{ 
          p: 6, 
          mt: 6, 
          textAlign: 'center',
          borderRadius: 2
        }}
      >
        <Box sx={{ mb: 3 }}>
          <ErrorOutlineIcon sx={{ fontSize: 80, color: 'text.secondary' }} />
        </Box>

        <Typography variant="h3" component="h1" gutterBottom>
          Page Not Found
        </Typography>
        
        <Typography variant="h6" color="text.secondary" paragraph>
          Sorry, the page you are looking for doesn't exist or has been moved.
        </Typography>
        
        <Box sx={{ mt: 6, mb: 3 }}>
          <Typography variant="body1" gutterBottom>
            Here are some helpful links to get you back on track:
          </Typography>
          
          <Grid container spacing={2} sx={{ mt: 3, justifyContent: 'center' }}>
            <Grid item xs={12} sm={4}>
              <Button 
                variant="contained" 
                color="primary" 
                startIcon={<HomeIcon />}
                fullWidth
                size="large"
                onClick={() => navigate('/')}
              >
                Home
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <Button 
                variant="outlined" 
                startIcon={<SchoolIcon />}
                fullWidth
                size="large"
                onClick={() => navigate('/courses')}
              >
                Courses
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={4}>
              <Button 
                variant="outlined" 
                startIcon={<CodeIcon />}
                fullWidth
                size="large"
                onClick={() => navigate('/playground')}
              >
                Playground
              </Button>
            </Grid>
          </Grid>
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mt: 4 }}>
          If you believe this is a technical error, please contact our support team.
        </Typography>
      </Paper>
    </Container>
  );
};

export default NotFoundPage;