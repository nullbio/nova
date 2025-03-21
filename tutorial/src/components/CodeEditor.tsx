import React, { useState, useEffect } from 'react';
import { Editor } from '@monaco-editor/react';
import { Box, Button, Paper, Typography, Tab, Tabs, CircularProgress, Tooltip, Alert } from '@mui/material';
import { useInterpreter } from '../context/InterpreterContext';
import { CodeExample, NovaInterpretationResult } from '../types';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import ReplayIcon from '@mui/icons-material/Replay';
import InfoIcon from '@mui/icons-material/Info';

// Add TypeScript support for ResizeObserver
declare global {
  interface Window {
    ResizeObserver: typeof ResizeObserver;
  }
}

// More aggressive workaround for ResizeObserver errors
if (typeof window !== 'undefined') {
  // Prevent error events from triggering for ResizeObserver
  window.addEventListener('error', (e) => {
    if (e.message === 'ResizeObserver loop limit exceeded' || 
        e.message.includes('ResizeObserver loop completed with undelivered notifications')) {
      e.stopImmediatePropagation();
      e.preventDefault();
      return false;
    }
  }, true);
  
  // Simpler solution - just suppress errors in the console
  const originalConsoleError = console.error;
  console.error = (...args: any[]) => {
    if (args[0] && typeof args[0] === 'string' && args[0].includes('ResizeObserver')) {
      return;
    }
    originalConsoleError(...args);
  };
}

interface CodeEditorProps {
  initialCode: CodeExample;
  readOnly?: boolean;
  showOutput?: boolean;
  onCodeRun?: (result: NovaInterpretationResult) => void;
  onCodeChange?: (code: string) => void;
  instructions?: string;
  hints?: string[];
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`code-tabpanel-${index}`}
      aria-labelledby={`code-tab-${index}`}
      style={{ height: 'calc(100% - 48px)' }}
      {...other}
    >
      {value === index && <Box sx={{ height: '100%', pt: 1 }}>{children}</Box>}
    </div>
  );
};

const CodeEditor: React.FC<CodeEditorProps> = ({
  initialCode,
  readOnly = false,
  showOutput = true,
  onCodeRun,
  onCodeChange,
  instructions,
  hints = [],
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [novaCode, setNovaCode] = useState(initialCode.nova);
  const [pythonCode, setPythonCode] = useState(initialCode.pytorch);
  const [output, setOutput] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [currentHint, setCurrentHint] = useState(0);
  const [copySuccess, setCopySuccess] = useState(false);

  const { interpretCode, isConnected } = useInterpreter();

  // Reset error when code changes
  useEffect(() => {
    setError(null);
  }, [novaCode]);

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleCodeChange = (value: string | undefined) => {
    if (value !== undefined) {
      setNovaCode(value);
      if (onCodeChange) {
        onCodeChange(value);
      }
    }
  };

  const handlePythonChange = (value: string | undefined) => {
    if (value !== undefined) {
      setPythonCode(value);
    }
  };

  const runCode = async () => {
    if (!novaCode.trim()) return;
    
    setIsRunning(true);
    setError(null);
    setOutput('');
    
    try {
      const result = await interpretCode(novaCode);
      setPythonCode(result.pythonCode);
      
      if (result.output) {
        setOutput(result.output);
      } else if (result.error) {
        setError(result.error);
      } else {
        setOutput('Code executed successfully with no output.');
      }
      
      if (onCodeRun) {
        onCodeRun(result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setIsRunning(false);
    }
  };

  const resetCode = () => {
    setNovaCode(initialCode.nova);
    setPythonCode(initialCode.pytorch);
    setOutput('');
    setError(null);
    setShowHint(false);
    setCurrentHint(0);
  };

  const toggleHints = () => {
    // Use a small delay to help with ResizeObserver issues
    setTimeout(() => {
      // Toggle hints visibility
      if (showHint) {
        setShowHint(false);
      } else {
        setShowHint(true);
        setCurrentHint(0);
      }
    }, 10);
  };
  
  const showPreviousHint = () => {
    if (currentHint > 0) {
      setCurrentHint(prev => prev - 1);
    }
  };
  
  const showNextHint = () => {
    if (currentHint < hints.length - 1) {
      setCurrentHint(prev => prev + 1);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    });
  };

  return (
    <Paper elevation={3} sx={{ height: '500px', display: 'flex', flexDirection: 'column' }}>
      {instructions && (
        <Box sx={{ p: 2, bgcolor: 'primary.main', color: 'white' }}>
          <Typography variant="body1">{instructions}</Typography>
        </Box>
      )}
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="code tabs">
          <Tab label="Nova" sx={{ color: activeTab === 0 ? 'primary.main' : 'inherit' }} />
          <Tab label="PyTorch" sx={{ color: activeTab === 1 ? 'secondary.main' : 'inherit' }} />
          {showOutput && <Tab label="Output" sx={{ color: activeTab === 2 ? 'success.main' : 'inherit' }} />}
        </Tabs>
      </Box>
      
      <Box sx={{ flexGrow: 1, position: 'relative' }}>
        <TabPanel value={activeTab} index={0}>
          <Editor
            height="100%"
            defaultLanguage="python"
            value={novaCode}
            onChange={handleCodeChange}
            options={{
              readOnly: readOnly,
              minimap: { enabled: false },
              fontSize: 14,
              scrollBeyondLastLine: false,
              automaticLayout: true,
              fixedOverflowWidgets: true,
              renderLineHighlight: 'none',
              contextmenu: false,
            }}
            theme="vs-light"
            onMount={(editor) => {
              // Force a layout recalculation after component is mounted
              setTimeout(() => {
                editor.layout();
              }, 100);
            }}
          />
        </TabPanel>
        
        <TabPanel value={activeTab} index={1}>
          <Editor
            height="100%"
            defaultLanguage="python"
            value={pythonCode}
            onChange={handlePythonChange}
            options={{
              readOnly: true,
              minimap: { enabled: false },
              fontSize: 14,
              scrollBeyondLastLine: false,
              automaticLayout: true,
              fixedOverflowWidgets: true,
              renderLineHighlight: 'none',
              contextmenu: false,
            }}
            theme="vs-light"
            onMount={(editor) => {
              // Force a layout recalculation after component is mounted
              setTimeout(() => {
                editor.layout();
              }, 100);
            }}
          />
        </TabPanel>
        
        {showOutput && (
          <TabPanel value={activeTab} index={2}>
            <Box 
              sx={{ 
                height: '100%', 
                p: 2, 
                fontFamily: 'monospace', 
                whiteSpace: 'pre-wrap',
                overflow: 'auto',
                bgcolor: '#f5f5f5',
                borderRadius: 1
              }}
            >
              {error ? (
                <Alert severity="error" sx={{ whiteSpace: 'pre-wrap' }}>{error}</Alert>
              ) : (
                output || 'Run the code to see output here.'
              )}
            </Box>
          </TabPanel>
        )}
      </Box>
      
      <Box sx={{ p: 1, display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #eee' }}>
        <Box>
          <Tooltip title={isConnected ? "Run code" : "Interpreter not connected"}>
            <span>
              <Button
                variant="contained"
                color="primary"
                onClick={runCode}
                disabled={readOnly || isRunning || !isConnected}
                startIcon={isRunning ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
                sx={{ mr: 1 }}
              >
                {isRunning ? 'Running...' : 'Run'}
              </Button>
            </span>
          </Tooltip>
          
          <Button
            variant="outlined"
            onClick={resetCode}
            startIcon={<ReplayIcon />}
            sx={{ mr: 1 }}
          >
            Reset
          </Button>
          
          <Button
            variant="outlined"
            onClick={() => copyToClipboard(activeTab === 0 ? novaCode : pythonCode)}
            startIcon={<ContentCopyIcon />}
          >
            {copySuccess ? 'Copied!' : 'Copy'}
          </Button>
        </Box>
        
        {hints.length > 0 && (
          <Button
            startIcon={<InfoIcon />}
            variant="text"
            color="info"
            onClick={toggleHints}
          >
            {showHint ? 'Hide Hints' : 'Show Hints'}
          </Button>
        )}
      </Box>
      
      {showHint && hints.length > 0 && (
        <Box sx={{ p: 2, bgcolor: 'info.light', color: 'info.contrastText' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2" fontWeight="bold">
              Hint {currentHint + 1}/{hints.length}
            </Typography>
            <Box>
              <Button 
                size="small" 
                variant="text" 
                disabled={currentHint === 0}
                onClick={showPreviousHint}
                sx={{ minWidth: 'auto', p: 0.5, mr: 1, color: 'info.contrastText' }}
              >
                Previous
              </Button>
              <Button 
                size="small" 
                variant="text" 
                disabled={currentHint === hints.length - 1}
                onClick={showNextHint}
                sx={{ minWidth: 'auto', p: 0.5, color: 'info.contrastText' }}
              >
                Next
              </Button>
            </Box>
          </Box>
          <Typography variant="body2">
            {hints[currentHint]}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default CodeEditor;