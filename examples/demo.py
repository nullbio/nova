"""
Nova Interpreter Demo

This script demonstrates the use of the Nova interpreter to translate
Nova natural language code to executable PyTorch code.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from nova.interpreter import NovaInterpreter

def main():
    """Run the Nova interpreter demo."""
    # Create a Nova interpreter instance
    interpreter = NovaInterpreter()
    
    # Example Nova code
    nova_code = """
    # Simple MNIST classifier in Nova
    create processing pipeline digit_classifier:
        add transformation stage fully_connected with 784 inputs and 256 outputs
        apply relu activation
        add transformation stage fully_connected with 256 inputs and 128 outputs
        apply relu activation
        add transformation stage fully_connected with 128 inputs and 10 outputs
    
    train digit_classifier on mnist_dataloader:
        measure error using cross_entropy
        improve using adam with learning rate 0.001
        repeat for 5 learning cycles
    """
    
    # Translate the Nova code to PyTorch
    pytorch_code = interpreter.translate(nova_code)
    
    # Generate an explanation of the translation
    explanation = interpreter.explain_translation(nova_code, pytorch_code)
    
    # Print the results
    print("=== Nova Code ===")
    print(nova_code)
    print("\n=== PyTorch Code ===")
    print(pytorch_code)
    print("\n=== Explanation ===")
    print(explanation)
    
    # More complex example with CNN
    nova_cnn_code = """
    # CNN for image classification in Nova
    create image processing pipeline image_classifier:
        add feature detector with 3 input channels, 16 output channels and 3x3 filter size
        apply relu activation
        add downsampling using max method with size 2x2
        add feature detector with 16 input channels, 32 output channels and 3x3 filter size
        apply relu activation
        add downsampling using max method with size 2x2
        add transformation stage fully_connected with 2048 inputs and 128 outputs
        apply relu activation
        add transformation stage fully_connected with 128 inputs and 10 outputs
    
    train image_classifier on cifar_dataloader:
        measure error using cross_entropy
        improve using adam with learning rate 0.001
        repeat for 10 learning cycles
    """
    
    # Translate the CNN Nova code to PyTorch
    cnn_pytorch_code = interpreter.translate(nova_cnn_code)
    
    # Generate an explanation of the translation
    cnn_explanation = interpreter.explain_translation(nova_cnn_code, cnn_pytorch_code)
    
    # Print the results
    print("\n\n=== Nova CNN Code ===")
    print(nova_cnn_code)
    print("\n=== PyTorch CNN Code ===")
    print(cnn_pytorch_code)
    print("\n=== CNN Explanation ===")
    print(cnn_explanation)

if __name__ == "__main__":
    main()
