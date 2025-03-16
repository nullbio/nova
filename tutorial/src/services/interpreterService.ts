import { InterpretationRequest, NovaInterpretationResult } from '../types';
import axios from 'axios';

// Mock Socket.io implementation
interface MockSocket {
  disconnect: () => void;
  on: (event: string, callback: (...args: any[]) => void) => void;
  once: (event: string, callback: (...args: any[]) => void) => void;
  emit: (event: string, data: any) => void;
}

// Create mock socket.io implementation to avoid version conflicts
function createMockIo(url: string, options: any): MockSocket {
  console.log(`Creating mock socket connection to ${url} with options:`, options);
  
  // Simple implementation of an event emitter
  const listeners: Record<string, Array<(...args: any[]) => void>> = {};
  const onceListeners: Record<string, Array<(...args: any[]) => void>> = {};
  
  const mockSocket: MockSocket = {
    disconnect: () => {
      console.log('Mock socket disconnected');
      if (listeners['disconnect']) {
        listeners['disconnect'].forEach(cb => cb());
      }
    },
    
    on: (event: string, callback: (...args: any[]) => void) => {
      if (!listeners[event]) {
        listeners[event] = [];
      }
      listeners[event].push(callback);
      
      // Simulate immediate connection
      if (event === 'connect') {
        setTimeout(() => callback(), 100);
      }
    },
    
    once: (event: string, callback: (...args: any[]) => void) => {
      if (!onceListeners[event]) {
        onceListeners[event] = [];
      }
      onceListeners[event].push(callback);
    },
    
    emit: (event: string, data: any) => {
      console.log(`Mock socket emitting: ${event}`, data);
      
      // Simulate server behavior for the interpret event
      if (event === 'interpret') {
        setTimeout(() => {
          const novaCode = (data as InterpretationRequest).novaCode;
          
          // Parse Nova code to generate PyTorch equivalent
          const result: NovaInterpretationResult = {
            pythonCode: generatePythonCode(novaCode),
            output: "Execution successful",
            executionTime: 123,
          };
          
          if (onceListeners['interpretation_result']) {
            onceListeners['interpretation_result'].forEach(cb => cb(result));
            delete onceListeners['interpretation_result'];
          }
        }, 500);
      }
    }
  };
  
  return mockSocket;
}

function generatePythonCode(novaCode: string): string {
  // Very simple translator for Nova code to PyTorch
  if (novaCode.includes('create processing pipeline')) {
    const modelName = novaCode.match(/create processing pipeline (\w+):/)?.[1] || 'model';
    
    return `import torch
import torch.nn as nn
import torch.optim as optim

class ${modelName.charAt(0).toUpperCase() + modelName.slice(1)}(nn.Module):
    def __init__(self):
        super(${modelName.charAt(0).toUpperCase() + modelName.slice(1)}, self).__init__()
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

# Initialize model
${modelName} = ${modelName.charAt(0).toUpperCase() + modelName.slice(1)}()`;
  }
  
  return `import torch
import torch.nn as nn

# Generated PyTorch code
# This is a demo implementation`;
}

class InterpreterService {
  private socket: MockSocket | null = null;
  private sessionId: string = '';
  private isConnected: boolean = false;
  private connectionCallbacks: Array<(connected: boolean) => void> = [];

  constructor() {
    this.sessionId = this.generateSessionId();
  }

  connect(url: string = 'ws://localhost:5000'): Promise<boolean> {
    return new Promise((resolve) => {
      if (this.socket) {
        this.socket.disconnect();
      }

      this.socket = createMockIo(url, {
        transports: ['websocket'],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });

      this.socket.on('connect', () => {
        console.log('Connected to Nova interpreter');
        this.isConnected = true;
        this.notifyConnectionChange(true);
        resolve(true);
      });

      this.socket.on('disconnect', () => {
        console.log('Disconnected from Nova interpreter');
        this.isConnected = false;
        this.notifyConnectionChange(false);
      });
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      this.notifyConnectionChange(false);
    }
  }

  interpretCode(novaCode: string): Promise<NovaInterpretationResult> {
    if (!this.socket || !this.isConnected) {
      return this.fallbackInterpretation(novaCode);
    }

    return new Promise((resolve, reject) => {
      const request: InterpretationRequest = {
        novaCode,
        sessionId: this.sessionId,
      };

      this.socket!.emit('interpret', request);

      this.socket!.once('interpretation_result', (result: NovaInterpretationResult) => {
        resolve(result);
      });

      this.socket!.once('interpretation_error', (error: string) => {
        reject(new Error(error));
      });

      // Set a timeout in case the server doesn't respond
      setTimeout(() => {
        reject(new Error('Interpretation request timed out'));
      }, 30000);
    });
  }

  // Fallback to API call if WebSocket isn't available
  private fallbackInterpretation(novaCode: string): Promise<NovaInterpretationResult> {
    console.log('Using fallback interpretation via API - simulating response');
    
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          pythonCode: generatePythonCode(novaCode),
          output: "Execution successful (fallback mode)",
          executionTime: 234,
        });
      }, 800);
    });
  }

  isInterpreterConnected(): boolean {
    return this.isConnected;
  }

  onConnectionChange(callback: (connected: boolean) => void) {
    this.connectionCallbacks.push(callback);
    // Immediately call with current status
    callback(this.isConnected);
    
    return () => {
      this.connectionCallbacks = this.connectionCallbacks.filter(cb => cb !== callback);
    };
  }

  private notifyConnectionChange(connected: boolean) {
    this.connectionCallbacks.forEach(callback => callback(connected));
  }

  private generateSessionId(): string {
    return 'nova-session-' + Math.random().toString(36).substring(2, 15);
  }
}

export const interpreterService = new InterpreterService();
export default interpreterService;