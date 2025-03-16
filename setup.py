"""
Setup file for the Nova package.
"""

import os
import sys
from setuptools import setup, find_packages

# Check if PyTorch is installed and prompt for installation if not
def check_pytorch():
    try:
        import torch
        return True
    except ImportError:
        return False

# Function to display interactive checkbox prompt
def checkbox_prompt(options, default_checked=None):
    """
    Display an interactive checkbox prompt in the terminal.
    
    Args:
        options: List of options to display
        default_checked: List of options to check by default
    
    Returns:
        List of selected options
    """
    if default_checked is None:
        default_checked = []
    
    selected = [opt in default_checked for opt in options]
    
    # Check if we're running in a terminal that supports interactive prompts
    if not sys.stdout.isatty():
        # If not in interactive terminal, just return default selection
        return [opt for i, opt in enumerate(options) if selected[i]]
    
    try:
        import curses
        
        def draw_menu(stdscr):
            curses.curs_set(0)  # Hide cursor
            current_row = 0
            
            # Function to draw the current state
            def draw_current_state():
                stdscr.clear()
                h, w = stdscr.getmaxlines(), stdscr.getmaxyx()[1]
                stdscr.addstr(0, 0, "Select additional ML libraries to install:")
                stdscr.addstr(1, 0, "(Use arrow keys to navigate, SPACE to toggle, ENTER to confirm)")
                
                for i, option in enumerate(options):
                    if i == current_row:
                        stdscr.attron(curses.A_REVERSE)
                    checkbox = "[X]" if selected[i] else "[ ]"
                    stdscr.addstr(i + 3, 0, f"{checkbox} {option}")
                    if i == current_row:
                        stdscr.attroff(curses.A_REVERSE)
                
                # Add buttons
                button_row = len(options) + 4
                stdscr.addstr(button_row, 0, "[ All ]")
                stdscr.addstr(button_row, 10, "[ None ]")
                
                # Highlight the Next button
                stdscr.attron(curses.A_REVERSE)
                stdscr.addstr(button_row, 20, "[ Next ]")
                stdscr.attroff(curses.A_REVERSE)
            
            # Initial draw
            draw_current_state()
            
            # Handle input
            while True:
                key = stdscr.getch()
                
                if key == curses.KEY_UP and current_row > 0:
                    current_row -= 1
                elif key == curses.KEY_DOWN and current_row < len(options) - 1:
                    current_row += 1
                elif key == ord(' '):
                    selected[current_row] = not selected[current_row]
                elif key == ord('a'):  # 'a' for All
                    selected = [True] * len(options)
                elif key == ord('n'):  # 'n' for None
                    selected = [False] * len(options)
                elif key == 10:  # Enter key to confirm
                    break
                
                draw_current_state()
            
            return [opt for i, opt in enumerate(options) if selected[i]]
        
        # Run the curses application
        return curses.wrapper(draw_menu)
        
    except (ImportError, Exception) as e:
        # Fallback for environments without curses or other issues
        print("\nSelect additional ML libraries to install:")
        selected_options = []
        
        for i, option in enumerate(options):
            default = "Y" if option in default_checked else "n"
            response = input(f"Install {option}? [y/N]: ").strip().lower() or default.lower()
            if response in ["y", "yes"]:
                selected_options.append(option)
        
        return selected_options

# Define dependencies
dependencies = [
    "numpy>=1.20.0",
    "tqdm>=4.60.0",
]

# Optional ML libraries with their pip package names
ml_libraries = {
    "PyTorch": "torch>=2.0.0",
    "TorchVision (for computer vision)": "torchvision>=0.15.0",
    "TorchAudio (for audio processing)": "torchaudio>=2.0.0",
    "TorchText (for NLP)": "torchtext>=0.15.0",
    "scikit-learn": "scikit-learn>=1.0.0",
    "TensorFlow": "tensorflow>=2.0.0",
    "Hugging Face Transformers": "transformers>=4.0.0",
    "Pandas (for data manipulation)": "pandas>=1.3.0"
}

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "black>=22.0.0",
        "isort>=5.0.0",
        "flake8>=4.0.0",
        "mypy>=0.910",
    ],
    "docs": [
        "mkdocs>=1.3.0",
        "mkdocs-material>=8.0.0",
        "pymdown-extensions>=9.0",
    ],
}

# Track if user selected PyTorch in the first prompt
pytorch_selected = False

# Add PyTorch to dependencies if user confirms
if not check_pytorch():
    if "--quiet" not in sys.argv and "--help" not in sys.argv:
        print("")
        print("PyTorch not detected. PyTorch is required for Nova to function.")
        response = input("Would you like to install PyTorch now? [Y/n]: ").strip().lower()
        
        if response in ["", "y", "yes"]:
            pytorch_selected = True
            cuda_available = False
            try:
                import subprocess
                result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cuda_available = result.returncode == 0
            except:
                pass
                
            if cuda_available:
                print("CUDA detected. Installing PyTorch with CUDA support.")
                print("If you need a specific CUDA version, please cancel and visit:")
                print("https://pytorch.org/get-started/locally/")
                response = input("Continue with default CUDA version? [Y/n]: ").strip().lower()
                if response in ["", "y", "yes"]:
                    dependencies.append(ml_libraries["PyTorch"])
                else:
                    print("Continuing without installing PyTorch.")
                    print("Please install PyTorch manually following instructions at:")
                    print("https://pytorch.org/get-started/locally/")
                    pytorch_selected = False
            else:
                print("No CUDA detected. Installing CPU version of PyTorch.")
                dependencies.append(ml_libraries["PyTorch"])
        else:
            print("Continuing without installing PyTorch.")
            print("Please install PyTorch manually following instructions at:")
            print("https://pytorch.org/get-started/locally/")
        print("")
else:
    # PyTorch is already installed
    pytorch_selected = True

# Prompt for additional ML libraries
if "--quiet" not in sys.argv and "--help" not in sys.argv:
    print("\nWould you like to install additional ML libraries?")
    
    # Get list of available libraries (excluding PyTorch since it was handled separately)
    available_libraries = list(ml_libraries.keys())
    if "PyTorch" in available_libraries:
        available_libraries.remove("PyTorch")
    
    # Set default selection
    default_selected = []
    if pytorch_selected:
        # If PyTorch was selected, we'll offer TorchVision, TorchAudio, TorchText as they're related
        if "TorchVision (for computer vision)" in available_libraries:
            default_selected.append("TorchVision (for computer vision)")
        if "TorchAudio (for audio processing)" in available_libraries:
            default_selected.append("TorchAudio (for audio processing)")
        if "TorchText (for NLP)" in available_libraries:
            default_selected.append("TorchText (for NLP)")
    
    # Show checkbox prompt
    if available_libraries:
        print("\nSelect additional ML libraries to install:")
        print("(Use numbers to select/deselect, 'a' for All, 'n' for None, 'c' to Continue)")
        
        # Simple fallback approach for environments without interactive features
        selected = []
        for i, lib in enumerate(available_libraries):
            is_default = lib in default_selected
            default_choice = "Y" if is_default else "n"
            choice = input(f"{i+1}. {lib} [{default_choice}]: ").strip().lower()
            
            if choice == "a":  # All
                selected = available_libraries
                break
            elif choice == "n":  # None
                selected = []
                break
            elif choice == "c":  # Continue with current selection
                selected = [lib for i, lib in enumerate(available_libraries) 
                            if lib in default_selected and (choice != "n")]
                break
            elif choice in ["", "y", "yes"] and is_default:
                selected.append(lib)
            elif choice in ["y", "yes"]:
                selected.append(lib)
        
        # Add selected libraries to dependencies
        for lib in selected:
            if lib in ml_libraries:
                dependencies.append(ml_libraries[lib])
        
        print(f"\nSelected libraries: {', '.join(selected) if selected else 'None'}")
        print("")

setup(
    name="nova-dl",
    version="0.1.0",
    author="Nova Team",
    author_email="info@nova-dl.org",
    description="Natural Language Interface for Deep Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nova-team/nova",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=dependencies,
    extras_require=extras_require,
)