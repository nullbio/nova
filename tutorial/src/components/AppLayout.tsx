import React, { useState } from 'react';
import {
  AppBar,
  Box,
  CssBaseline,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Divider,
  Tooltip,
  Badge,
  CircularProgress,
  Chip,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import HomeIcon from '@mui/icons-material/Home';
import SchoolIcon from '@mui/icons-material/School';
import SettingsIcon from '@mui/icons-material/Settings';
import CodeIcon from '@mui/icons-material/Code';
import InfoIcon from '@mui/icons-material/Info';
import SignalWifiOffIcon from '@mui/icons-material/SignalWifiOff';
import SignalWifiStatusbar4BarIcon from '@mui/icons-material/SignalWifiStatusbar4Bar';
import PersonIcon from '@mui/icons-material/Person';
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useInterpreter } from '../context/InterpreterContext';
import { useProgress } from '../context/ProgressContext';

const drawerWidth = 240;

const AppLayout: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { isConnected, isLoading } = useInterpreter();
  const { progress } = useProgress();
  const navigate = useNavigate();
  const location = useLocation();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'Home', icon: <HomeIcon />, path: '/' },
    { text: 'Courses', icon: <SchoolIcon />, path: '/courses' },
    { text: 'Playground', icon: <CodeIcon />, path: '/playground' },
    { text: 'Achievements', icon: <EmojiEventsIcon />, path: '/achievements' },
    { text: 'About', icon: <InfoIcon />, path: '/about' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  ];

  const drawer = (
    <div>
      <Toolbar sx={{ justifyContent: 'center', py: 1 }}>
        <Typography variant="h6" noWrap component="div">
          Nova Tutorial
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                setMobileOpen(false);
              }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
              {item.text === 'Courses' && progress.completedLessons.length > 0 && (
                <Chip 
                  size="small" 
                  label={progress.completedLessons.length} 
                  color="primary" 
                  sx={{ ml: 1 }} 
                />
              )}
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <Box sx={{ p: 2 }}>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Interpreter Status:
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {isLoading ? (
            <>
              <CircularProgress size={16} sx={{ mr: 1 }} />
              <Typography variant="body2">Connecting...</Typography>
            </>
          ) : isConnected ? (
            <>
              <SignalWifiStatusbar4BarIcon color="success" sx={{ mr: 1 }} />
              <Typography variant="body2" color="success.main">
                Connected
              </Typography>
            </>
          ) : (
            <>
              <SignalWifiOffIcon color="error" sx={{ mr: 1 }} />
              <Typography variant="body2" color="error">
                Disconnected
              </Typography>
            </>
          )}
        </Box>
      </Box>
    </div>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Nova Interactive Tutorial
          </Typography>
          <Tooltip title="Interpreter Status">
            <IconButton color="inherit">
              {isLoading ? (
                <CircularProgress color="inherit" size={24} />
              ) : isConnected ? (
                <SignalWifiStatusbar4BarIcon />
              ) : (
                <Badge badgeContent="!" color="error">
                  <SignalWifiOffIcon />
                </Badge>
              )}
            </IconButton>
          </Tooltip>
          <Tooltip title="Profile">
            <IconButton color="inherit">
              <PersonIcon />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
        aria-label="menu items"
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          overflow: 'auto',
        }}
      >
        <Toolbar /> {/* This adds space at the top for the AppBar */}
        <Outlet />
      </Box>
    </Box>
  );
};

export default AppLayout;