import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

// Add TypeScript definition for webpack dev server client
declare global {
  interface Window {
    __webpack_dev_server_client__?: {
      overlay: (options: any) => any;
    };
  }
}

// Suppress ResizeObserver errors more aggressively
const originalConsoleError = console.error;
console.error = (...args) => {
  if (
    typeof args[0] === 'string' &&
    args[0].includes('ResizeObserver loop') 
  ) {
    return;
  }
  originalConsoleError(...args);
};

// Directly prevent the error overlay from showing ResizeObserver errors
window.addEventListener('error', (event) => {
  if (event.message && event.message.includes('ResizeObserver loop')) {
    event.stopImmediatePropagation();
    event.preventDefault();
    return false;
  }
}, true);

// Remove existing error overlay and intercept webpack overlay
const removeResizeObserverOverlay = () => {
  // Look for any error dialog
  const errorOverlay = document.querySelector('[role="dialog"]');
  if (errorOverlay && errorOverlay.innerHTML && errorOverlay.innerHTML.includes('ResizeObserver')) {
    errorOverlay.remove();
  }
  
  // Look for webpack error overlay (which has a specific structure)
  const webpackOverlay = document.getElementById('webpack-dev-server-client-overlay');
  if (webpackOverlay && webpackOverlay.innerHTML && webpackOverlay.innerHTML.includes('ResizeObserver')) {
    webpackOverlay.style.display = 'none';
  }
};

// Initial cleanup
setTimeout(removeResizeObserverOverlay, 100);

// Continuously check and remove ResizeObserver error overlays
setInterval(removeResizeObserverOverlay, 500);

// Attempt to patch webpack-dev-server's overlay
if (typeof window !== 'undefined' && window.__webpack_dev_server_client__) {
  try {
    const originalOverlay = window.__webpack_dev_server_client__.overlay;
    window.__webpack_dev_server_client__.overlay = (options) => {
      if (options.message && options.message.includes('ResizeObserver')) {
        return null;
      }
      return originalOverlay(options);
    };
  } catch (e) {
    console.log('Failed to patch webpack overlay', e);
  }
}

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
