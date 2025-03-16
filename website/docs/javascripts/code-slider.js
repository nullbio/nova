document.addEventListener('DOMContentLoaded', function() {
  // Find and convert all code-comparison divs to tabbed interfaces
  const codeComparisons = document.querySelectorAll('.code-comparison');
  
  codeComparisons.forEach(function(comparison) {
    convertToTabbedInterface(comparison);
  });
  
  function convertToTabbedInterface(comparisonDiv) {
    // Get the title and content divs
    const divs = comparisonDiv.querySelectorAll('> div');
    if (divs.length !== 2) return; // Only handle pairs
    
    const novaDiv = divs[0];
    const pytorchDiv = divs[1];
    
    const novaTitle = novaDiv.querySelector('.title')?.textContent || 'Nova';
    const pytorchTitle = pytorchDiv.querySelector('.title')?.textContent || 'PyTorch';
    
    // Create tabbed container
    const tabsContainer = document.createElement('div');
    tabsContainer.className = 'code-tabs-container';
    
    // Create tabs navigation
    const tabsNav = document.createElement('div');
    tabsNav.className = 'code-tabs-nav';
    
    const novaTab = document.createElement('div');
    novaTab.className = 'code-tab nova-tab active';
    novaTab.textContent = novaTitle;
    novaTab.addEventListener('click', function() {
      activateTab(novaTab, novaPane);
    });
    
    const pytorchTab = document.createElement('div');
    pytorchTab.className = 'code-tab pytorch-tab';
    pytorchTab.textContent = pytorchTitle;
    pytorchTab.addEventListener('click', function() {
      activateTab(pytorchTab, pytorchPane);
    });
    
    tabsNav.appendChild(novaTab);
    tabsNav.appendChild(pytorchTab);
    
    // Create content
    const tabsContent = document.createElement('div');
    tabsContent.className = 'code-tabs-content';
    
    const novaPane = document.createElement('div');
    novaPane.className = 'code-pane nova-pane active';
    novaPane.innerHTML = novaDiv.querySelector('.nova-code')?.innerHTML || novaDiv.innerHTML;
    
    const pytorchPane = document.createElement('div');
    pytorchPane.className = 'code-pane pytorch-pane';
    pytorchPane.innerHTML = pytorchDiv.querySelector('.pytorch-code')?.innerHTML || pytorchDiv.innerHTML;
    
    tabsContent.appendChild(novaPane);
    tabsContent.appendChild(pytorchPane);
    
    // Assemble tabbed interface
    tabsContainer.appendChild(tabsNav);
    tabsContainer.appendChild(tabsContent);
    
    // Replace the original comparison with the tabbed interface
    comparisonDiv.parentNode.replaceChild(tabsContainer, comparisonDiv);
    
    // Tab switching function
    function activateTab(tab, pane) {
      // Deactivate all tabs and panes
      tabsNav.querySelectorAll('.code-tab').forEach(function(t) {
        t.classList.remove('active');
      });
      
      tabsContent.querySelectorAll('.code-pane').forEach(function(p) {
        p.classList.remove('active');
      });
      
      // Activate the selected tab and pane
      tab.classList.add('active');
      pane.classList.add('active');
    }
  }
});