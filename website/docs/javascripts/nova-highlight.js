/**
 * Nova Language Syntax Highlighter
 * 
 * This script adds syntax highlighting for Nova code blocks in the documentation.
 * It works by identifying Nova code blocks and applying custom styling to them.
 */

document.addEventListener('DOMContentLoaded', function() {
  // Define Nova keywords and patterns
  const KEYWORDS = [
    'load', 'data', 'collection', 'from', 'with', 'apply', 'create', 'processing',
    'pipeline', 'add', 'transformation', 'stage', 'inputs', 'outputs', 'activation',
    'on', 'measure', 'error', 'using', 'improve', 'repeat', 'for', 'learning',
    'cycles', 'evaluate', 'report', 'results', 'save', 'to', 'prepare', 'stream',
    'batch', 'size', 'shuffle', 'enabled', 'disable', 'split', 'into',
    'testing', 'print', 'progress', 'every', 'convert', 'feature', 'grid'
  ];

  // Define keywords that should only match if they're "standalone"
  // and not part of a variable/property name
  const STANDALONE_KEYWORDS = ['train', 'training'];
  
  // Define Nova special constructs
  const CONSTRUCTS = [
    'fully_connected', 'relu', 'sigmoid', 'tanh', 'softmax', 'dropout', 'rate',
    'cross_entropy', 'binary_cross_entropy', 'mean_squared_error', 'adam', 
    'gradient_descent', 'learning', 'rate', 'weight', 'decay', 'normalization',
    'mean', 'deviation', 'accuracy', 'precision', 'recall', 'f1', 'score'
  ];

  // Define numeric pattern
  const NUMBER_PATTERN = /\b\d+(\.\d+)?\b/g;

  // Define comment pattern - matches Python's greenish-gray color (#408080)
  const COMMENT_PATTERN = /#.*/g;

  // Function for variable color - using black for variables like Python
  function getVariableColor() {
    // Black for variables, matching Python
    return '#000000';
  }

  // Function to highlight Nova code
  function highlightNovaCode() {
    // Find all tabbed-sets where the first tab has "Nova" in the label
    const tabbedSets = document.querySelectorAll('.tabbed-set');
    
    tabbedSets.forEach(function(set) {
      const labels = set.querySelectorAll('.tabbed-labels label');
      const blocks = set.querySelectorAll('.tabbed-block');
      
      // Check if the first label contains "Nova"
      if (labels.length > 0 && labels[0].textContent.trim().includes('Nova')) {
        // Find the first code block
        if (blocks.length > 0) {
          const codeElement = blocks[0].querySelector('pre code');
          
          if (codeElement) {
            // Store original content to preserve indentation
            let content = codeElement.innerHTML;
            
            // First, clean up line number elements directly in content
            content = content.replace(/<a\s+id="__codelineno[^"]*"[^>]*>.*?<\/a>/g, '');
            
            // Create a temporary div to extract plain text
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = content;
            const plainText = tempDiv.textContent;
            
            // Identify variable names
            const variablePattern = /\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\b/g;
            const variables = new Set();
            const variableMatches = plainText.matchAll(variablePattern);
            
            for (const match of variableMatches) {
              const varName = match[0];
              // Skip if it's a keyword or construct
              if (!KEYWORDS.includes(varName) && 
                  !CONSTRUCTS.includes(varName) && 
                  !STANDALONE_KEYWORDS.includes(varName)) {
                variables.add(varName);
              }
            }
            
            // Get a single color for all variables (Python-like)
            const variableColor = getVariableColor();
            
            // Apply highlighting to the cleaned content
            let highlighted = content;
            
            // Highlight comments first (they take precedence)
            // Make sure to wrap the entire comment line
            highlighted = highlighted.replace(COMMENT_PATTERN, match => 
              `<span class="nova-comment">${match}</span>`);
            
            // Highlight numbers
            highlighted = highlighted.replace(NUMBER_PATTERN, match => 
              `<span class="nova-number">${match}</span>`);
            
            // Highlight standalone keywords (.train vs train)
            STANDALONE_KEYWORDS.forEach(keyword => {
              // Only match the keyword when it's standalone (not preceded by a dot)
              const pattern = new RegExp(`(?<!\\.)\\b${keyword}\\b`, 'g');
              highlighted = highlighted.replace(pattern, match => 
                `<span class="nova-keyword">${match}</span>`);
            });
            
            // Highlight regular keywords
            KEYWORDS.forEach(keyword => {
              const pattern = new RegExp(`\\b${keyword}\\b`, 'g');
              highlighted = highlighted.replace(pattern, match => 
                `<span class="nova-keyword">${match}</span>`);
            });
            
            // Highlight constructs
            CONSTRUCTS.forEach(construct => {
              const pattern = new RegExp(`\\b${construct}\\b`, 'g');
              highlighted = highlighted.replace(pattern, match => 
                `<span class="nova-construct">${match}</span>`);
            });
            
            // Highlight variables with a consistent color (Python-like)
            // Sort variables by length (descending) to handle overlap correctly
            const sortedVariables = Array.from(variables).sort((a, b) => b.length - a.length);
            
            sortedVariables.forEach(variable => {
              // Don't match variables inside other strings that were already highlighted
              const safeVarPattern = new RegExp(`(?<!span class=")\\b${variable.replace(/\./g, '\\.')}\\b`, 'g');
              highlighted = highlighted.replace(safeVarPattern, match => 
                `<span class="nova-variable">${match}</span>`);
            });
            
            // Set highlighted content
            codeElement.innerHTML = highlighted;
            
            // Add Nova highlighting class to the code block
            codeElement.classList.add('nova-highlighted');
            
            // Double-check and remove any remaining line number elements that might have been missed
            const remainingLineNos = codeElement.querySelectorAll('a[id^="__codelineno"]');
            if (remainingLineNos.length > 0) {
              remainingLineNos.forEach(el => {
                el.parentNode.removeChild(el);
              });
            }
            
            // Add copy and highlight buttons
            addCodeActionButtons(codeElement);
          }
        }
      }
    });
  }

  // Function to add copy and highlight buttons to code blocks
  function addCodeActionButtons(codeElement) {
    // Get parent pre element
    const preElement = codeElement.closest('pre');
    if (!preElement) return;
    
    // Create the actions container
    const actionsContainer = document.createElement('div');
    actionsContainer.className = 'code-actions';
    
    // Create copy button
    const copyButton = document.createElement('button');
    copyButton.className = 'code-action-button copy-button';
    copyButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>Copy`;
    copyButton.addEventListener('click', function() {
      // Get the text content of the code element
      const textToCopy = codeElement.textContent;
      
      // Use the clipboard API to copy the text
      navigator.clipboard.writeText(textToCopy).then(function() {
        // Change button text temporarily
        const originalText = copyButton.innerHTML;
        copyButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"></path></svg>Copied!`;
        
        // Reset button text after 2 seconds
        setTimeout(function() {
          copyButton.innerHTML = originalText;
        }, 2000);
      }).catch(function(err) {
        console.error('Could not copy text: ', err);
      });
    });
    
    // Create "Select All" button (renamed from "Highlight")
    const selectButton = document.createElement('button');
    selectButton.className = 'code-action-button select-button';
    selectButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path></svg>Select`;
    selectButton.addEventListener('click', function() {
      // Create a range and select the code element contents
      const range = document.createRange();
      range.selectNodeContents(codeElement);
      
      // Remove any existing selection
      const selection = window.getSelection();
      selection.removeAllRanges();
      
      // Apply the new selection
      selection.addRange(range);
      
      // Provide visual feedback that the text was selected
      const originalText = selectButton.innerHTML;
      selectButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"></path></svg>Selected!`;
      
      // Reset button text after 2 seconds
      setTimeout(function() {
        selectButton.innerHTML = originalText;
      }, 2000);
    });
    
    // Add buttons to the container
    actionsContainer.appendChild(copyButton);
    actionsContainer.appendChild(selectButton);
    
    // Add the container to the pre element
    preElement.style.position = 'relative';
    preElement.appendChild(actionsContainer);
  }

  // Also add buttons to PyTorch code blocks
  function addButtonsToAllCodeBlocks() {
    const allCodeBlocks = document.querySelectorAll('.tabbed-set .tabbed-block pre code');
    allCodeBlocks.forEach(codeElement => {
      // Skip if buttons are already added
      if (codeElement.closest('pre').querySelector('.code-actions')) return;
      
      addCodeActionButtons(codeElement);
    });
  }

  // Run highlighting
  highlightNovaCode();
  
  // Add buttons to all code blocks
  addButtonsToAllCodeBlocks();
  
  // Add MutationObserver to handle dynamic content loading
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      if (mutation.addedNodes && mutation.addedNodes.length > 0) {
        // Check if any of the added nodes might contain code blocks
        for (let i = 0; i < mutation.addedNodes.length; i++) {
          const node = mutation.addedNodes[i];
          if (node.nodeType === 1) { // Element node
            if (node.classList && (
              node.classList.contains('tabbed-set') || 
              node.querySelector('.tabbed-set')
            )) {
              highlightNovaCode();
              addButtonsToAllCodeBlocks();
              break;
            }
          }
        }
      }
    });
  });
  
  // Observe changes in the document body
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
});