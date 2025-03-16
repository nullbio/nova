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

  // Define comment pattern
  const COMMENT_PATTERN = /#.*/g;

  // Function to generate consistent colors for variable names
  function generateConsistentColor(variableName) {
    // Simple hash function for string
    let hash = 0;
    for (let i = 0; i < variableName.length; i++) {
      hash = ((hash << 5) - hash) + variableName.charCodeAt(i);
      hash |= 0; // Convert to 32bit integer
    }
    
    // Define a set of Python-friendly colors that are readable on white
    // and distinct from other syntax elements
    const colors = [
      '#2F597F', // Darker blue
      '#8B4513', // SaddleBrown
      '#008B8B', // DarkCyan
      '#8B008B', // DarkMagenta
      '#556B2F', // DarkOliveGreen
      '#5F2F6A', // Dark purple
      '#703324', // Dark brown-red
      '#284E60', // Dark teal
      '#4A5459', // Slate gray
      '#504C4E', // Dark gray
      '#0C5C11', // Dark forest green
      '#1C5B80', // Darker teal
      '#7D3C98', // Violet
      '#5B552A', // Olive
      '#4C3C4F'  // Dark mauve
    ];
    
    // Use the hash to select a color from the array
    const colorIndex = Math.abs(hash) % colors.length;
    return colors[colorIndex];
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
            
            // Generate colors for variables
            const variableColors = {};
            variables.forEach(variable => {
              variableColors[variable] = generateConsistentColor(variable);
            });
            
            // Apply highlighting to the cleaned content
            let highlighted = content;
            
            // Highlight comments first (they take precedence)
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
            
            // Highlight variables with their unique colors
            // Sort variables by length (descending) to handle overlap correctly
            const sortedVariables = Array.from(variables).sort((a, b) => b.length - a.length);
            
            sortedVariables.forEach(variable => {
              // Don't match variables inside other strings that were already highlighted
              const safeVarPattern = new RegExp(`(?<!span class=")\\b${variable.replace(/\./g, '\\.')}\\b`, 'g');
              const color = variableColors[variable];
              highlighted = highlighted.replace(safeVarPattern, match => 
                `<span class="nova-variable" style="color: ${color};">${match}</span>`);
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
          }
        }
      }
    });
  }

  // Run highlighting
  highlightNovaCode();
  
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