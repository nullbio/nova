import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';

// Contexts
import { InterpreterProvider } from './context/InterpreterContext';
import { ProgressProvider } from './context/ProgressContext';

// Components
import AppLayout from './components/AppLayout';

// Pages
import Home from './pages/Home';
import Playground from './pages/Playground';
import CoursesPage from './pages/CoursesPage';
import LessonPage from './pages/LessonPage';
import SettingsPage from './pages/SettingsPage';
import AboutPage from './pages/AboutPage';
import NotFoundPage from './pages/NotFoundPage';
import AchievementsPage from './pages/AchievementsPage';

// Theme configuration
const theme = createTheme({
  palette: {
    primary: {
      main: '#5046E4', // Nova blue
      light: '#7A71EB',
      dark: '#3933B6',
      contrastText: '#fff',
    },
    secondary: {
      main: '#FF4081', // Pink accent
      light: '#FF79B0',
      dark: '#C60055',
      contrastText: '#fff',
    },
    background: {
      default: '#F9F9FC',
      paper: '#fff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
    },
    h2: {
      fontWeight: 600,
    },
    h3: {
      fontWeight: 600,
    },
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
    h6: {
      fontWeight: 500,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
        containedPrimary: {
          boxShadow: '0 2px 8px rgba(80, 70, 228, 0.3)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <InterpreterProvider>
        <ProgressProvider>
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<AppLayout />}>
                <Route index element={<Home />} />
                <Route path="courses" element={<CoursesPage />} />
                <Route path="courses/:courseId" element={<CoursesPage />} />
                <Route path="courses/:courseId/lessons/:lessonId" element={<LessonPage />} />
                <Route path="playground" element={<Playground />} />
                <Route path="achievements" element={<AchievementsPage />} />
                <Route path="settings" element={<SettingsPage />} />
                <Route path="about" element={<AboutPage />} />
                <Route path="404" element={<NotFoundPage />} />
                <Route path="*" element={<Navigate to="/404" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ProgressProvider>
      </InterpreterProvider>
    </ThemeProvider>
  );
}

export default App;
