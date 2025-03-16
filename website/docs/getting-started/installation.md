# Installation

This guide will help you install Nova and set up your development environment.

## Prerequisites

Before installing Nova, make sure you have the following prerequisites:

- Python 3.8 or higher
- pip (Python package installer)

## Installing Nova

### Using pip

The simplest way to install Nova is using pip:

```bash
pip install nova-dl
```

The installation process provides two interactive prompts:

#### 1. PyTorch Installation Prompt

During installation, Nova will check if PyTorch is already installed:

- If PyTorch is not installed, you'll be prompted to confirm installation
- You can choose to install PyTorch automatically or manually install it later

```
PyTorch not detected. PyTorch is required for Nova to function.
Would you like to install PyTorch now? [Y/n]: 
```

If you have CUDA-compatible hardware, the installer will also detect this and ask:

```
CUDA detected. Installing PyTorch with CUDA support.
If you need a specific CUDA version, please cancel and visit:
https://pytorch.org/get-started/locally/
Continue with default CUDA version? [Y/n]:
```

#### 2. Additional ML Libraries Prompt

Next, you'll be prompted to install additional machine learning libraries:

```
Would you like to install additional ML libraries?

Select additional ML libraries to install:
(Use numbers to select/deselect, 'a' for All, 'n' for None, 'c' to Continue)

1. TorchVision (for computer vision) [Y]: 
2. TorchAudio (for audio processing) [Y]: 
3. TorchText (for NLP) [Y]: 
4. scikit-learn [n]: 
5. TensorFlow [n]: 
6. Hugging Face Transformers [n]: 
7. Pandas (for data manipulation) [n]: 
```

You can:
- Select libraries individually by typing 'y' or 'n'
- Press 'a' to select all libraries
- Press 'n' to select none
- Press 'c' to continue with the current selection
- Press Enter to accept the default values (PyTorch-related libraries are checked by default if PyTorch was selected)

If you choose to install PyTorch manually later, you can follow the instructions on the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

### From Source

To install Nova from source, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/nova-team/nova.git
cd nova
```

2. Install the package in development mode:

```bash
pip install -e .
```

This will also guide you through the same interactive prompts for installing PyTorch and additional ML libraries.

## Verifying Installation

To verify that Nova is installed correctly, run the following Python code:

```python
import nova
print(nova.__version__)
```

You should see the version number of your Nova installation printed.

## Setting Up Your Environment

### Virtual Environment (Recommended)

It's recommended to use a virtual environment for your Nova projects:

```bash
python -m venv nova-env
source nova-env/bin/activate  # On Windows: nova-env\Scripts\activate
pip install nova-dl
```

### Jupyter Notebook Setup

If you're using Jupyter Notebooks, install the following packages:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=nova-env --display-name="Python (Nova)"
```

Then start Jupyter and select the "Python (Nova)" kernel.

## Installing Additional Dependencies Later

If you decide to install additional libraries later, you can do so with pip:

```bash
# PyTorch ecosystem
pip install torch torchvision torchaudio torchtext

# Other ML libraries
pip install scikit-learn tensorflow transformers pandas

# Visualization libraries
pip install matplotlib seaborn
```

## Troubleshooting

### Common Installation Issues

#### PyTorch Installation

If you encounter issues with PyTorch, visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform-specific installation instructions.

#### CUDA Compatibility

For GPU acceleration, ensure your CUDA version is compatible with your PyTorch version. You can check your CUDA version with:

```bash
nvidia-smi
```

And install a compatible PyTorch version using the instructions on the PyTorch website.

#### Package Conflicts

If you encounter package conflicts, try creating a fresh virtual environment:

```bash
python -m venv nova-clean-env
source nova-clean-env/bin/activate
pip install nova-dl
```

## Next Steps

Now that you have Nova installed, you can proceed to the [Quick Start Guide](quick-start.md) to create your first model using Nova.