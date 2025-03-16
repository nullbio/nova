document.addEventListener('DOMContentLoaded', function() {
  // Create the font size control bar
  var fontSizeBar = document.createElement('div');
  fontSizeBar.className = 'font-size-bar';
  
  // Create a container to match the width of the main content
  var container = document.createElement('div');
  container.className = 'font-size-container';
  
  // Create the font size control
  var control = document.createElement('div');
  control.className = 'font-size-control';
  control.innerHTML = `
    <span>Text Size:</span>
    <button class="smaller">A-</button>
    <button class="reset">Reset</button>
    <button class="larger">A+</button>
  `;
  
  // Add the control to the container, then the container to the bar
  container.appendChild(control);
  fontSizeBar.appendChild(container);
  
  // Insert the bar at the top of the page, before the header
  var header = document.querySelector('.md-header');
  if (header) {
    header.parentNode.insertBefore(fontSizeBar, header);
  } else {
    // Fallback to beginning of body
    document.body.prepend(fontSizeBar);
  }

  // Load the saved font size from localStorage
  var savedSize = localStorage.getItem('novaDocsFontSize');
  if (savedSize) {
    document.documentElement.style.fontSize = savedSize;
  }

  // Add event listeners
  control.querySelector('.smaller').addEventListener('click', function() {
    changeSize(-10);
  });

  control.querySelector('.reset').addEventListener('click', function() {
    // Reset to 95% which is our base size
    document.documentElement.style.fontSize = '95%';
    localStorage.setItem('novaDocsFontSize', '95%');
  });

  control.querySelector('.larger').addEventListener('click', function() {
    changeSize(10);
  });

  function changeSize(percent) {
    currentSize = parseFloat(getComputedStyle(document.documentElement).fontSize);
    var newSize = currentSize * (1 + percent/100);
    // Set a reasonable limit
    if (newSize >= 8 && newSize <= 24) {
      var newSizeStr = newSize + 'px';
      document.documentElement.style.fontSize = newSizeStr;
      // Save the font size to localStorage
      localStorage.setItem('novaDocsFontSize', newSizeStr);
    }
  }
});