export interface Lesson {
  id: string;
  title: string;
  description: string;
  complexity: 'beginner' | 'intermediate' | 'advanced';
  content: LessonContent[];
  estimatedTime: number; // in minutes
}

export type LessonContentType = 
  | 'text' 
  | 'code' 
  | 'image' 
  | 'interactive-code'
  | 'quiz'
  | 'video';

export interface LessonContent {
  type: LessonContentType;
  content: any;
  explanation?: string;
}

export interface CodeExample {
  nova: string;
  pytorch: string;
  expected_output?: string;
}

export interface InteractiveCodeBlock extends CodeExample {
  instructions: string;
  hints: string[];
  solution: string;
  validation_criteria?: string;
}

export interface Quiz {
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
  points: number;
  unlockedAt: string | null;
}

export interface UserProgress {
  completedLessons: string[];
  currentLesson?: string;
  currentStep?: number;
  achievements: string[];
  codeSubmissions: Record<string, string[]>;
  quizResults: Record<string, boolean>;
  experiencePoints: number;
  completedBeginner: number;
  completedIntermediate: number;
  completedAdvanced: number;
  totalBeginner: number;
  totalIntermediate: number;
  totalAdvanced: number;
}

export interface NovaInterpretationResult {
  pythonCode: string;
  output?: string;
  error?: string;
  executionTime?: number;
}

export interface InterpretationRequest {
  novaCode: string;
  sessionId: string;
}

export interface Course {
  id: string;
  title: string;
  description: string;
  image: string;
  lessons: Lesson[];
  prerequisites?: string[];
  tags: string[];
}