import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { UserProgress, Achievement } from '../types';

interface ProgressContextType {
  progress: UserProgress;
  updateProgress: (updates: Partial<UserProgress>) => void;
  completeLesson: (lessonId: string, complexity: string) => void;
  setCurrentLesson: (lessonId: string, step?: number) => void;
  addCodeSubmission: (lessonId: string, code: string) => void;
  setQuizResult: (quizId: string, correct: boolean) => void;
  resetProgress: () => void;
  isLessonCompleted: (lessonId: string) => boolean;
  getUnlockedAchievements: () => Achievement[];
  calculateLevelProgress: (level: string) => { completed: number; total: number; percentage: number };
  checkForNewAchievements: () => Achievement[];
  getLevel: () => string;
  getExperiencePoints: () => number;
}

// Define available achievements
const availableAchievements: Achievement[] = [
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

// Map of levels based on experience points
const levelThresholds = {
  'Novice': 0,
  'Apprentice': 300,
  'Practitioner': 1000,
  'Expert': 2500,
  'Master': 5000
};

// Initial progress state
const defaultProgress: UserProgress = {
  completedLessons: [],
  achievements: [],
  codeSubmissions: {},
  quizResults: {},
  experiencePoints: 0,
  completedBeginner: 0,
  completedIntermediate: 0,
  completedAdvanced: 0,
  totalBeginner: 0,
  totalIntermediate: 0,
  totalAdvanced: 0
};

// Mock lesson complexity mapping - in a real app this would come from your courses data
const mockLessonComplexity: Record<string, string> = {
  'getting-started': 'beginner',
  'first-neural-network': 'beginner',
  'intro-to-cnns': 'intermediate',
  'transfer-learning': 'advanced'
};

const ProgressContext = createContext<ProgressContextType>({
  progress: defaultProgress,
  updateProgress: () => {},
  completeLesson: () => {},
  setCurrentLesson: () => {},
  addCodeSubmission: () => {},
  setQuizResult: () => {},
  resetProgress: () => {},
  isLessonCompleted: () => false,
  getUnlockedAchievements: () => [],
  calculateLevelProgress: () => ({ completed: 0, total: 0, percentage: 0 }),
  checkForNewAchievements: () => [],
  getLevel: () => 'Novice',
  getExperiencePoints: () => 0
});

export const useProgress = () => useContext(ProgressContext);

interface ProgressProviderProps {
  children: ReactNode;
}

const STORAGE_KEY = 'nova_tutorial_progress';

export const ProgressProvider: React.FC<ProgressProviderProps> = ({ children }) => {
  const [progress, setProgress] = useState<UserProgress>(() => {
    // Initialize from localStorage if available
    const savedProgress = localStorage.getItem(STORAGE_KEY);
    if (savedProgress) {
      try {
        return JSON.parse(savedProgress) as UserProgress;
      } catch (e) {
        console.error('Failed to parse saved progress:', e);
      }
    }
    return defaultProgress;
  });

  // Save to localStorage whenever progress changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  }, [progress]);

  // Initialize lesson counts for each complexity level
  useEffect(() => {
    // This would normally come from your API or course data
    // For now, let's set some placeholder values based on our mock data
    if (
      progress.totalBeginner === 0 && 
      progress.totalIntermediate === 0 && 
      progress.totalAdvanced === 0
    ) {
      setProgress(prev => ({
        ...prev,
        totalBeginner: 2, // Number of beginner lessons in our mock data
        totalIntermediate: 1, // Number of intermediate lessons
        totalAdvanced: 1 // Number of advanced lessons
      }));
    }
  }, []);

  const updateProgress = (updates: Partial<UserProgress>) => {
    setProgress(prev => ({ ...prev, ...updates }));
  };

  const completeLesson = (lessonId: string, complexity: string = 'beginner') => {
    // Only update if not already completed
    if (progress.completedLessons.includes(lessonId)) {
      return;
    }

    let complexityCount = {};
    let xpGain = 0;

    // Assign experience points based on complexity
    switch (complexity) {
      case 'beginner':
        xpGain = 50;
        complexityCount = { completedBeginner: progress.completedBeginner + 1 };
        break;
      case 'intermediate':
        xpGain = 100;
        complexityCount = { completedIntermediate: progress.completedIntermediate + 1 };
        break;
      case 'advanced':
        xpGain = 200;
        complexityCount = { completedAdvanced: progress.completedAdvanced + 1 };
        break;
      default:
        xpGain = 50;
        complexityCount = { completedBeginner: progress.completedBeginner + 1 };
    }

    setProgress(prev => ({
      ...prev,
      completedLessons: [...prev.completedLessons, lessonId],
      experiencePoints: prev.experiencePoints + xpGain,
      ...complexityCount
    }));

    // Check for new achievements
    checkForNewAchievements();
  };

  const setCurrentLesson = (lessonId: string, step?: number) => {
    setProgress(prev => ({
      ...prev,
      currentLesson: lessonId,
      currentStep: step !== undefined ? step : 0
    }));
  };

  const addCodeSubmission = (lessonId: string, code: string) => {
    setProgress(prev => {
      const submissions = prev.codeSubmissions[lessonId] || [];
      return {
        ...prev,
        codeSubmissions: {
          ...prev.codeSubmissions,
          [lessonId]: [...submissions, code]
        }
      };
    });
  };

  const setQuizResult = (quizId: string, correct: boolean) => {
    // Award XP for correct answers
    const xpGain = correct ? 25 : 0;
    
    setProgress(prev => ({
      ...prev,
      quizResults: {
        ...prev.quizResults,
        [quizId]: correct
      },
      experiencePoints: prev.experiencePoints + xpGain
    }));

    // Check for achievements like quiz_ace
    checkForNewAchievements();
  };

  const resetProgress = () => {
    setProgress(defaultProgress);
  };

  const isLessonCompleted = (lessonId: string): boolean => {
    return progress.completedLessons.includes(lessonId);
  };

  const getUnlockedAchievements = (): Achievement[] => {
    return availableAchievements.filter(achievement => 
      progress.achievements.includes(achievement.id)
    );
  };

  const calculateLevelProgress = (level: string): { completed: number; total: number; percentage: number } => {
    let completed = 0;
    let total = 0;
    
    switch (level) {
      case 'beginner':
        completed = progress.completedBeginner;
        total = progress.totalBeginner;
        break;
      case 'intermediate':
        completed = progress.completedIntermediate;
        total = progress.totalIntermediate;
        break;
      case 'advanced':
        completed = progress.completedAdvanced;
        total = progress.totalAdvanced;
        break;
      default:
        // Return overall progress
        completed = progress.completedLessons.length;
        total = progress.totalBeginner + progress.totalIntermediate + progress.totalAdvanced;
    }
    
    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    return { completed, total, percentage };
  };

  const checkForNewAchievements = (): Achievement[] => {
    const newAchievements: Achievement[] = [];
    
    // Helper to unlock an achievement if not already unlocked
    const unlockAchievement = (id: string) => {
      if (!progress.achievements.includes(id)) {
        const achievement = availableAchievements.find(a => a.id === id);
        if (achievement) {
          // Clone the achievement and set the unlock time
          const unlockedAchievement = {...achievement, unlockedAt: new Date().toISOString()};
          newAchievements.push(unlockedAchievement);
          
          // Add the achievement ID and award XP
          setProgress(prev => ({
            ...prev,
            achievements: [...prev.achievements, id],
            experiencePoints: prev.experiencePoints + (achievement.points || 0)
          }));
        }
      }
    };
    
    // Check for first lesson completion
    if (progress.completedLessons.length === 1) {
      unlockAchievement('first_lesson_completed');
    }
    
    // Check for beginner mastery
    if (progress.completedBeginner === progress.totalBeginner && progress.totalBeginner > 0) {
      unlockAchievement('beginner_master');
    }
    
    // Check for intermediate mastery
    if (progress.completedIntermediate === progress.totalIntermediate && progress.totalIntermediate > 0) {
      unlockAchievement('intermediate_master');
    }
    
    // Check for advanced mastery
    if (progress.completedAdvanced === progress.totalAdvanced && progress.totalAdvanced > 0) {
      unlockAchievement('advanced_master');
    }
    
    // Check for quiz ace (10 correct answers)
    const correctQuizzes = Object.values(progress.quizResults).filter(result => result === true).length;
    if (correctQuizzes >= 10) {
      unlockAchievement('quiz_ace');
    }
    
    // Check for code master (5 code submissions)
    const codeSubmissionsCount = Object.keys(progress.codeSubmissions).length;
    if (codeSubmissionsCount >= 5) {
      unlockAchievement('code_master');
    }
    
    // Check for complete mastery (all lessons)
    const totalLessons = progress.totalBeginner + progress.totalIntermediate + progress.totalAdvanced;
    if (progress.completedLessons.length === totalLessons && totalLessons > 0) {
      unlockAchievement('nova_expert');
    }
    
    return newAchievements;
  };

  const getLevel = (): string => {
    const xp = progress.experiencePoints;
    const levels = Object.entries(levelThresholds);
    
    // Sort levels by XP threshold in descending order
    levels.sort((a, b) => b[1] - a[1]);
    
    // Find the highest level threshold that the user has reached
    for (const [level, threshold] of levels) {
      if (xp >= threshold) {
        return level;
      }
    }
    
    return 'Novice'; // Default level
  };

  const getExperiencePoints = (): number => {
    return progress.experiencePoints;
  };

  const value: ProgressContextType = {
    progress,
    updateProgress,
    completeLesson,
    setCurrentLesson,
    addCodeSubmission,
    setQuizResult,
    resetProgress,
    isLessonCompleted,
    getUnlockedAchievements,
    calculateLevelProgress,
    checkForNewAchievements,
    getLevel,
    getExperiencePoints
  };

  return (
    <ProgressContext.Provider value={value}>
      {children}
    </ProgressContext.Provider>
  );
};