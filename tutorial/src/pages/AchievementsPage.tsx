import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Grid,
  Paper,
  Card,
  CardContent,
  Divider,
  Chip,
  LinearProgress,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Tab,
  Tabs,
  CircularProgress,
  Alert,
  Tooltip
} from '@mui/material';
import { useProgress } from '../context/ProgressContext';
import { Achievement } from '../types';
import LockIcon from '@mui/icons-material/Lock';
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents';
import SchoolIcon from '@mui/icons-material/School';
import WorkspacePremiumIcon from '@mui/icons-material/WorkspacePremium';
import BoltIcon from '@mui/icons-material/Bolt';
import FilterNoneIcon from '@mui/icons-material/FilterNone';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import FormatListNumberedIcon from '@mui/icons-material/FormatListNumbered';

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
      id={`achievements-tabpanel-${index}`}
      aria-labelledby={`achievements-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
};

// Mock data for all achievements - in a real app this would be fetched from a service
const allAchievements: Achievement[] = [
  {
    id: 'first_lesson_completed',
    title: 'First Steps',
    description: 'Complete your first Nova lesson',
    icon: '🌟',
    points: 50,
    unlockedAt: null
  },
  {
    id: 'first_course_completed',
    title: 'Course Champion',
    description: 'Complete your first Nova course',
    icon: '🏆',
    points: 200,
    unlockedAt: null
  },
  {
    id: 'beginner_master',
    title: 'Beginner Master',
    description: 'Complete all beginner lessons',
    icon: '🥉',
    points: 300,
    unlockedAt: null
  },
  {
    id: 'intermediate_master',
    title: 'Intermediate Master',
    description: 'Complete all intermediate lessons',
    icon: '🥈',
    points: 500,
    unlockedAt: null
  },
  {
    id: 'advanced_master',
    title: 'Advanced Master',
    description: 'Complete all advanced lessons',
    icon: '🥇',
    points: 1000,
    unlockedAt: null
  },
  {
    id: 'quiz_ace',
    title: 'Quiz Ace',
    description: 'Get 10 quiz questions correct',
    icon: '📝',
    points: 150,
    unlockedAt: null
  },
  {
    id: 'code_master',
    title: 'Code Master',
    description: 'Complete 5 coding exercises',
    icon: '💻',
    points: 250,
    unlockedAt: null
  },
  {
    id: 'nova_expert',
    title: 'Nova Expert',
    description: 'Complete all lessons in all courses',
    icon: '🔥',
    points: 2000,
    unlockedAt: null
  }
];

const AchievementsPage: React.FC = () => {
  const { 
    progress, 
    getUnlockedAchievements, 
    calculateLevelProgress, 
    getLevel, 
    getExperiencePoints 
  } = useProgress();
  
  const [tabValue, setTabValue] = useState(0);
  const [unlockedAchievements, setUnlockedAchievements] = useState<Achievement[]>([]);
  const [lockedAchievements, setLockedAchievements] = useState<Achievement[]>([]);
  
  useEffect(() => {
    // Get unlocked achievements
    const unlocked = getUnlockedAchievements();
    setUnlockedAchievements(unlocked);
    
    // Calculate locked achievements
    const unlockedIds = progress.achievements;
    const locked = allAchievements.filter(a => !unlockedIds.includes(a.id));
    setLockedAchievements(locked);
  }, [progress, getUnlockedAchievements]);

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  // Calculate progress for different levels
  const beginnerProgress = calculateLevelProgress('beginner');
  const intermediateProgress = calculateLevelProgress('intermediate');
  const advancedProgress = calculateLevelProgress('advanced');
  const overallProgress = calculateLevelProgress('overall');
  
  // Get user's level and XP
  const userLevel = getLevel();
  const userXP = getExperiencePoints();
  
  // Calculate next level
  const levels = Object.entries({
    'Novice': 0,
    'Apprentice': 300,
    'Practitioner': 1000,
    'Expert': 2500,
    'Master': 5000
  });
  
  // Sort levels by threshold
  levels.sort((a, b) => a[1] - b[1]);
  
  // Find the next level
  let nextLevel = 'Master';
  let nextLevelXP = 5000;
  let currentLevelXP = 0;
  
  for (let i = 0; i < levels.length; i++) {
    if (userXP >= levels[i][1]) {
      currentLevelXP = levels[i][1];
      
      if (i < levels.length - 1) {
        nextLevel = levels[i + 1][0];
        nextLevelXP = levels[i + 1][1];
      }
    }
  }
  
  // Calculate progress to next level
  const nextLevelProgress = Math.min(Math.round(((userXP - currentLevelXP) / (nextLevelXP - currentLevelXP)) * 100), 100);
  
  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Your Progress & Achievements
      </Typography>
      
      <Grid container spacing={4}>
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 3, height: '100%' }}>
            <Box sx={{ textAlign: 'center', mb: 2 }}>
              <Avatar 
                sx={{ 
                  width: 80, 
                  height: 80, 
                  bgcolor: 'primary.main', 
                  margin: '0 auto',
                  fontSize: '2rem'
                }}
              >
                {userLevel === 'Master' ? '🔥' : <WorkspacePremiumIcon fontSize="large" />}
              </Avatar>
              
              <Typography variant="h5" sx={{ mt: 2 }}>
                {userLevel}
              </Typography>
              
              <Chip 
                label={`${userXP} XP`} 
                color="primary" 
                variant="outlined"
                sx={{ mt: 1 }}
              />
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            {userLevel !== 'Master' && (
              <Box sx={{ mt: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Progress to {nextLevel}</Typography>
                  <Typography variant="body2">{nextLevelProgress}%</Typography>
                </Box>
                
                <LinearProgress 
                  variant="determinate" 
                  value={nextLevelProgress} 
                  sx={{ height: 8, borderRadius: 4 }}
                />
                
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {userXP} / {nextLevelXP} XP
                </Typography>
              </Box>
            )}
            
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Achievements Unlocked
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="h5">
                  {progress.achievements.length}
                </Typography>
                
                <Typography variant="h5" color="text.secondary">
                  / {allAchievements.length}
                </Typography>
              </Box>
              
              <LinearProgress 
                variant="determinate" 
                value={(progress.achievements.length / allAchievements.length) * 100} 
                sx={{ height: 8, borderRadius: 4, mt: 1 }}
              />
            </Box>
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={8}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Level Progress
            </Typography>
            
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} md={4}>
                <Card sx={{ bgcolor: 'success.light', color: 'success.contrastText' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Beginner
                    </Typography>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <CircularProgress 
                        variant="determinate"
                        value={beginnerProgress.percentage}
                        size={40}
                        thickness={4}
                        sx={{ mr: 2, color: 'white' }}
                      />
                      <Typography variant="body1">
                        {beginnerProgress.completed} / {beginnerProgress.total} completed
                      </Typography>
                    </Box>
                    
                    <LinearProgress 
                      variant="determinate" 
                      value={beginnerProgress.percentage} 
                      sx={{ height: 6, borderRadius: 3, bgcolor: 'rgba(255,255,255,0.3)' }}
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card sx={{ bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Intermediate
                    </Typography>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <CircularProgress 
                        variant="determinate"
                        value={intermediateProgress.percentage}
                        size={40}
                        thickness={4}
                        sx={{ mr: 2, color: 'white' }}
                      />
                      <Typography variant="body1">
                        {intermediateProgress.completed} / {intermediateProgress.total} completed
                      </Typography>
                    </Box>
                    
                    <LinearProgress 
                      variant="determinate" 
                      value={intermediateProgress.percentage} 
                      sx={{ height: 6, borderRadius: 3, bgcolor: 'rgba(255,255,255,0.3)' }}
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card sx={{ bgcolor: 'error.light', color: 'error.contrastText' }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Advanced
                    </Typography>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <CircularProgress 
                        variant="determinate"
                        value={advancedProgress.percentage}
                        size={40}
                        thickness={4}
                        sx={{ mr: 2, color: 'white' }}
                      />
                      <Typography variant="body1">
                        {advancedProgress.completed} / {advancedProgress.total} completed
                      </Typography>
                    </Box>
                    
                    <LinearProgress 
                      variant="determinate" 
                      value={advancedProgress.percentage} 
                      sx={{ height: 6, borderRadius: 3, bgcolor: 'rgba(255,255,255,0.3)' }}
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <BoltIcon sx={{ mr: 1, color: 'primary.main' }} />
              <Typography variant="subtitle1">
                Overall Progress:
              </Typography>
              <Typography variant="body1" sx={{ ml: 1 }}>
                {overallProgress.completed} / {overallProgress.total} lessons completed
              </Typography>
            </Box>
            
            <LinearProgress 
              variant="determinate" 
              value={overallProgress.percentage} 
              sx={{ height: 10, borderRadius: 5 }}
            />
          </Paper>
        </Grid>
      </Grid>
      
      <Paper elevation={3} sx={{ mt: 4, p: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="achievement tabs">
            <Tab 
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <EmojiEventsIcon sx={{ mr: 1 }} />
                  All Achievements
                </Box>
              } 
              id="achievements-tab-0"
            />
            <Tab 
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <CheckCircleIcon sx={{ mr: 1 }} />
                  Unlocked
                  <Chip 
                    label={unlockedAchievements.length} 
                    size="small" 
                    color="primary" 
                    sx={{ ml: 1 }} 
                  />
                </Box>
              } 
              id="achievements-tab-1"
            />
            <Tab 
              label={
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <LockIcon sx={{ mr: 1 }} />
                  Locked
                  <Chip 
                    label={lockedAchievements.length} 
                    size="small" 
                    color="default" 
                    sx={{ ml: 1 }} 
                  />
                </Box>
              } 
              id="achievements-tab-2"
            />
          </Tabs>
        </Box>
        
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            {allAchievements.map((achievement) => {
              const isUnlocked = progress.achievements.includes(achievement.id);
              
              return (
                <Grid item xs={12} sm={6} md={4} key={achievement.id}>
                  <Card 
                    sx={{ 
                      height: '100%',
                      position: 'relative',
                      opacity: isUnlocked ? 1 : 0.7,
                      filter: isUnlocked ? 'none' : 'grayscale(1)',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: isUnlocked ? 'translateY(-4px)' : 'none',
                        boxShadow: isUnlocked ? 4 : 1
                      }
                    }}
                  >
                    {!isUnlocked && (
                      <Box 
                        sx={{ 
                          position: 'absolute', 
                          top: 0, 
                          left: 0, 
                          right: 0, 
                          bottom: 0, 
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'center',
                          zIndex: 2,
                          bgcolor: 'rgba(0,0,0,0.1)'
                        }}
                      >
                        <LockIcon sx={{ fontSize: 40, color: 'text.disabled' }} />
                      </Box>
                    )}
                    
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Avatar 
                          sx={{ 
                            bgcolor: isUnlocked ? 'primary.main' : 'action.disabled',
                            fontSize: '1.25rem',
                            mr: 2
                          }}
                        >
                          {achievement.icon}
                        </Avatar>
                        <Typography variant="h6">
                          {achievement.title}
                        </Typography>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {achievement.description}
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                        <Chip
                          label={`${achievement.points} XP`}
                          color={isUnlocked ? 'primary' : 'default'}
                          variant={isUnlocked ? 'filled' : 'outlined'}
                          size="small"
                        />
                        
                        {isUnlocked ? (
                          <Chip 
                            label="Completed" 
                            color="success" 
                            size="small" 
                            icon={<CheckCircleIcon />} 
                          />
                        ) : (
                          <Chip 
                            label="Locked" 
                            color="default" 
                            size="small" 
                            icon={<LockIcon />} 
                          />
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </TabPanel>
        
        <TabPanel value={tabValue} index={1}>
          {unlockedAchievements.length > 0 ? (
            <Grid container spacing={3}>
              {unlockedAchievements.map((achievement) => (
                <Grid item xs={12} sm={6} md={4} key={achievement.id}>
                  <Card 
                    sx={{ 
                      height: '100%',
                      transition: 'transform 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: 4
                      }
                    }}
                  >
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Avatar 
                          sx={{ 
                            bgcolor: 'primary.main',
                            fontSize: '1.25rem',
                            mr: 2
                          }}
                        >
                          {achievement.icon}
                        </Avatar>
                        <Typography variant="h6">
                          {achievement.title}
                        </Typography>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {achievement.description}
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                        <Chip
                          label={`${achievement.points} XP`}
                          color="primary"
                          size="small"
                        />
                        
                        <Tooltip title={achievement.unlockedAt ? new Date(achievement.unlockedAt).toLocaleString() : ''}>
                          <Chip 
                            label="Completed" 
                            color="success" 
                            size="small" 
                            icon={<CheckCircleIcon />} 
                          />
                        </Tooltip>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box sx={{ py: 4, textAlign: 'center' }}>
              <FilterNoneIcon sx={{ fontSize: 60, color: 'text.disabled', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                No achievements unlocked yet
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Complete lessons and challenges to earn achievements
              </Typography>
            </Box>
          )}
        </TabPanel>
        
        <TabPanel value={tabValue} index={2}>
          {lockedAchievements.length > 0 ? (
            <Grid container spacing={3}>
              {lockedAchievements.map((achievement) => (
                <Grid item xs={12} sm={6} md={4} key={achievement.id}>
                  <Card 
                    sx={{ 
                      height: '100%',
                      opacity: 0.7,
                      filter: 'grayscale(1)',
                      position: 'relative'
                    }}
                  >
                    <Box 
                      sx={{ 
                        position: 'absolute', 
                        top: 0, 
                        left: 0, 
                        right: 0, 
                        bottom: 0, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        zIndex: 2,
                        bgcolor: 'rgba(0,0,0,0.1)'
                      }}
                    >
                      <LockIcon sx={{ fontSize: 40, color: 'text.disabled' }} />
                    </Box>
                    
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Avatar 
                          sx={{ 
                            bgcolor: 'action.disabled',
                            fontSize: '1.25rem',
                            mr: 2
                          }}
                        >
                          {achievement.icon}
                        </Avatar>
                        <Typography variant="h6">
                          {achievement.title}
                        </Typography>
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {achievement.description}
                      </Typography>
                      
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
                        <Chip
                          label={`${achievement.points} XP`}
                          color="default"
                          variant="outlined"
                          size="small"
                        />
                        
                        <Chip 
                          label="Locked" 
                          color="default" 
                          size="small" 
                          icon={<LockIcon />} 
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Box sx={{ py: 4, textAlign: 'center' }}>
              <CheckCircleIcon sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
              <Typography variant="h6" color="success.main">
                All achievements unlocked!
              </Typography>
              <Typography variant="body2">
                Congratulations! You've earned every achievement.
              </Typography>
            </Box>
          )}
        </TabPanel>
      </Paper>
      
      <Alert severity="info" sx={{ mt: 4 }}>
        <Typography variant="body1">
          Continue completing lessons and challenges to unlock more achievements and earn experience points.
        </Typography>
      </Alert>
    </Container>
  );
};

export default AchievementsPage;