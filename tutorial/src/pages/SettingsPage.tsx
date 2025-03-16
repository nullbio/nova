import React, { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  Paper,
  Divider,
  Switch,
  FormControlLabel,
  Alert,
  Grid,
  CircularProgress,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Snackbar,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import DeleteIcon from '@mui/icons-material/Delete';
import SaveIcon from '@mui/icons-material/Save';
import PowerSettingsNewIcon from '@mui/icons-material/PowerSettingsNew';
import { useInterpreter } from '../context/InterpreterContext';
import { useProgress } from '../context/ProgressContext';

const SettingsPage: React.FC = () => {
  const { isConnected, isLoading, setInterpreterUrl, disconnectInterpreter } = useInterpreter();
  const { resetProgress } = useProgress();
  
  const [interpreterUrl, setUrl] = useState('ws://localhost:5000');
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const [autorun, setAutorun] = useState(false);
  
  const handleInterpreterConnect = async () => {
    if (!interpreterUrl.trim()) {
      setError('Please enter a valid URL');
      return;
    }
    
    setConnecting(true);
    setError(null);
    
    try {
      const success = await setInterpreterUrl(interpreterUrl);
      if (success) {
        setSnackbarMessage('Connected to interpreter successfully!');
        setSnackbarOpen(true);
      } else {
        setError('Failed to connect to interpreter. Please check the URL and try again.');
      }
    } catch (err) {
      setError('An error occurred while connecting to the interpreter.');
    } finally {
      setConnecting(false);
    }
  };
  
  const handleInterpreterDisconnect = () => {
    disconnectInterpreter();
    setSnackbarMessage('Disconnected from interpreter.');
    setSnackbarOpen(true);
  };
  
  const handleResetProgress = () => {
    if (window.confirm('Are you sure you want to reset all your progress? This cannot be undone.')) {
      resetProgress();
      setSnackbarMessage('Progress reset successfully.');
      setSnackbarOpen(true);
    }
  };
  
  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };
  
  const handleSaveSettings = () => {
    // In a real app, we would save these settings to localStorage or a backend
    localStorage.setItem('nova-tutorial-settings', JSON.stringify({
      darkMode,
      autorun,
      interpreterUrl,
    }));
    
    setSnackbarMessage('Settings saved successfully.');
    setSnackbarOpen(true);
  };
  
  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      
      <Box sx={{ mb: 4 }}>
        <Typography variant="body1" color="text.secondary">
          Configure your Nova tutorial experience and interpreter connection.
        </Typography>
      </Box>
      
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardHeader 
              title="Interpreter Connection" 
              subheader="Connect to a local or remote Nova interpreter"
              action={
                <IconButton 
                  color={isConnected ? 'success' : 'default'} 
                  aria-label="connection status"
                >
                  <PowerSettingsNewIcon />
                </IconButton>
              }
            />
            <CardContent>
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', mb: 2 }}>
                  <TextField
                    label="Interpreter URL"
                    variant="outlined"
                    fullWidth
                    value={interpreterUrl}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="ws://localhost:5000"
                    disabled={connecting || isLoading}
                    size="small"
                    error={!!error}
                    helperText={error}
                  />
                </Box>
                
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    variant={isConnected ? "outlined" : "contained"}
                    color={isConnected ? "secondary" : "primary"}
                    onClick={isConnected ? handleInterpreterDisconnect : handleInterpreterConnect}
                    disabled={connecting || isLoading}
                    startIcon={connecting || isLoading ? <CircularProgress size={20} color="inherit" /> : null}
                  >
                    {isConnected ? "Disconnect" : connecting || isLoading ? "Connecting..." : "Connect"}
                  </Button>
                </Box>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Status: {isConnected ? "Connected" : "Disconnected"}
              </Typography>
              
              <Typography variant="body2" color="text.secondary">
                The interpreter connection allows you to run Nova code and see real-time translations to PyTorch.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardHeader 
              title="App Settings" 
              subheader="Customize your tutorial experience"
              action={
                <IconButton 
                  color="primary" 
                  aria-label="save settings"
                  onClick={handleSaveSettings}
                >
                  <SaveIcon />
                </IconButton>
              }
            />
            <CardContent>
              <Box sx={{ mb: 3 }}>
                <FormControlLabel
                  control={
                    <Switch 
                      checked={darkMode} 
                      onChange={(e) => setDarkMode(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Dark Mode"
                />
                
                <Typography variant="body2" color="text.secondary" sx={{ ml: 4, mb: 2 }}>
                  Switch between light and dark theme.
                </Typography>
                
                <FormControlLabel
                  control={
                    <Switch 
                      checked={autorun} 
                      onChange={(e) => setAutorun(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Auto-run code examples"
                />
                
                <Typography variant="body2" color="text.secondary" sx={{ ml: 4 }}>
                  Automatically run code examples when navigating to a new lesson.
                </Typography>
              </Box>
              
              <Divider sx={{ my: 2 }} />
              
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Progress & Data
                </Typography>
                
                <Button
                  variant="outlined"
                  color="error"
                  onClick={handleResetProgress}
                  startIcon={<DeleteIcon />}
                  size="small"
                  sx={{ mt: 1 }}
                >
                  Reset All Progress
                </Button>
                
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  This will reset all your course progress and submissions.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
      
      <Paper elevation={1} sx={{ p: 3, mt: 4 }}>
        <Typography variant="subtitle1" gutterBottom>
          About the Interpreter
        </Typography>
        
        <Typography variant="body2" paragraph color="text.secondary">
          The Nova Interpreter allows you to run Nova code and see the equivalent PyTorch code. It can be run locally or on a remote server.
        </Typography>
        
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="body2">
            To run the interpreter locally, follow the instructions in the <strong>Nova Documentation</strong>.
          </Typography>
        </Alert>
      </Paper>
      
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={snackbarMessage}
      />
    </Container>
  );
};

export default SettingsPage;