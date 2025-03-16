import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import interpreterService from '../services/interpreterService';
import { NovaInterpretationResult } from '../types';

interface InterpreterContextType {
  isConnected: boolean;
  isLoading: boolean;
  interpretCode: (code: string) => Promise<NovaInterpretationResult>;
  setInterpreterUrl: (url: string) => Promise<boolean>;
  disconnectInterpreter: () => void;
}

const initialContext: InterpreterContextType = {
  isConnected: false,
  isLoading: false,
  interpretCode: () => Promise.reject(new Error('Interpreter context not initialized')),
  setInterpreterUrl: () => Promise.reject(new Error('Interpreter context not initialized')),
  disconnectInterpreter: () => {},
};

const InterpreterContext = createContext<InterpreterContextType>(initialContext);

export const useInterpreter = () => useContext(InterpreterContext);

interface InterpreterProviderProps {
  children: ReactNode;
  initialUrl?: string;
}

export const InterpreterProvider: React.FC<InterpreterProviderProps> = ({ 
  children, 
  initialUrl = 'ws://localhost:5000'
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Setup connection change callback
    const unsubscribe = interpreterService.onConnectionChange((connected) => {
      setIsConnected(connected);
      setIsLoading(false);
    });

    // Try to connect on component mount
    setIsLoading(true);
    interpreterService.connect(initialUrl)
      .finally(() => {
        setIsLoading(false);
      });

    return () => {
      unsubscribe();
      interpreterService.disconnect();
    };
  }, [initialUrl]);

  const interpretCode = async (code: string): Promise<NovaInterpretationResult> => {
    if (!code.trim()) {
      throw new Error('Code cannot be empty');
    }

    try {
      return await interpreterService.interpretCode(code);
    } catch (error) {
      console.error('Error interpreting code:', error);
      throw error;
    }
  };

  const setInterpreterUrl = async (url: string): Promise<boolean> => {
    setIsLoading(true);
    try {
      const result = await interpreterService.connect(url);
      return result;
    } finally {
      setIsLoading(false);
    }
  };

  const disconnectInterpreter = () => {
    interpreterService.disconnect();
  };

  const value: InterpreterContextType = {
    isConnected,
    isLoading,
    interpretCode,
    setInterpreterUrl,
    disconnectInterpreter,
  };

  return (
    <InterpreterContext.Provider value={value}>
      {children}
    </InterpreterContext.Provider>
  );
};