document.addEventListener('DOMContentLoaded', function() {
  // Find all code comparison elements
  const codeComparisons = document.querySelectorAll('.code-comparison');
  
  codeComparisons.forEach(function(container) {
    // Create the tab container
    const tabContainer = document.createElement('div');
    tabContainer.className = 'code-tabs-container';
    
    // Extract the Nova and PyTorch divs
    const divs = container.querySelectorAll(':scope > div');
    if (divs.length !== 2) return;
    
    const novaDiv = divs[0];
    const pytorchDiv = divs[1];
    
    // Get titles
    const novaTitle = novaDiv.querySelector('.title')?.textContent || 'Nova';
    const pytorchTitle = pytorchDiv.querySelector('.title')?.textContent || 'PyTorch';
    
    // Create tab navigation
    const tabNav = document.createElement('div');
    tabNav.className = 'code-tabs-nav';
    tabNav.innerHTML = `
      <div class="code-tab nova-tab active" data-tab="nova">${novaTitle}</div>
      <div class="code-tab pytorch-tab" data-tab="pytorch">${pytorchTitle}</div>
    `;
    
    // Create content area
    const contentArea = document.createElement('div');
    contentArea.className = 'code-tabs-content';
    
    // Create Nova pane
    const novaPane = document.createElement('div');
    novaPane.className = 'code-pane nova-pane active';
    novaPane.innerHTML = novaDiv.innerHTML;
    
    // Create PyTorch pane
    const pytorchPane = document.createElement('div');
    pytorchPane.className = 'code-pane pytorch-pane';
    pytorchPane.innerHTML = pytorchDiv.innerHTML;
    
    // Add panes to content area
    contentArea.appendChild(novaPane);
    contentArea.appendChild(pytorchPane);
    
    // Add nav and content to container
    tabContainer.appendChild(tabNav);
    tabContainer.appendChild(contentArea);
    
    // Replace original with tabbed version
    container.parentNode.replaceChild(tabContainer, container);
    
    // Add event listeners to tabs
    tabNav.addEventListener('click', function(e) {
      if (e.target.classList.contains('code-tab')) {
        // Remove active class from all tabs and panes
        tabNav.querySelectorAll('.code-tab').forEach(tab => tab.classList.remove('active'));
        contentArea.querySelectorAll('.code-pane').forEach(pane => pane.classList.remove('active'));
        
        // Add active class to clicked tab
        e.target.classList.add('active');
        
        // Activate corresponding pane
        const tabType = e.target.getAttribute('data-tab');
        if (tabType === 'nova') {
          novaPane.classList.add('active');
        } else if (tabType === 'pytorch') {
          pytorchPane.classList.add('active');
        }
      }
    });
  });
});