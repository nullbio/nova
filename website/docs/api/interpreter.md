# Nova Interpreter API

The Nova interpreter is the core component that translates Nova natural language code into executable PyTorch code. This document covers the public API of the interpreter, its classes, methods, and how to extend its functionality.

## NovaInterpreter Class

The main entry point for using Nova is the `NovaInterpreter` class.

```python
from nova import NovaInterpreter

# Create an interpreter instance
interpreter = NovaInterpreter()
```

### Key Methods

#### translate

```python
def translate(self, nova_code: str) -> str:
    """
    Translate Nova code to PyTorch code.
    
    Args:
        nova_code: The Nova code to translate
        
    Returns:
        str: The corresponding PyTorch code
    """
```

This is the primary method for translating Nova code to PyTorch code. It takes a string containing Nova code and returns a string containing the equivalent PyTorch code.

**Example:**
```python
nova_code = """
create processing pipeline simple_model:
    add transformation stage fully_connected with 10 inputs and 5 outputs
    apply relu activation
    add transformation stage fully_connected with 5 inputs and 1 outputs
"""

pytorch_code = interpreter.translate(nova_code)
print(pytorch_code)
```

#### explain_translation

```python
def explain_translation(self, nova_code: str, pytorch_code: str) -> str:
    """
    Generate an explanation of the translation from Nova to PyTorch.
    
    Args:
        nova_code: The original Nova code
        pytorch_code: The translated PyTorch code
        
    Returns:
        str: An explanation of the translation
    """
```

This method generates a detailed explanation of how the Nova code was translated to PyTorch code. It's useful for understanding the translation process and learning PyTorch concepts.

**Example:**
```python
explanation = interpreter.explain_translation(nova_code, pytorch_code)
print(explanation)
```

### Configuration Options

The `NovaInterpreter` class accepts several configuration options when instantiated:

```python
interpreter = NovaInterpreter(
    verbose=False,           # Print additional information during translation
    include_comments=True,   # Include explanatory comments in the generated code
    optimize_code=False,     # Apply optimizations to the generated code
    custom_mappings=None     # Custom mappings for extending the interpreter
)
```

## Working with the Interpreter

### Basic Usage

```python
from nova import NovaInterpreter

# Create an interpreter instance
interpreter = NovaInterpreter()

# Define Nova code
nova_code = """
create processing pipeline model:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
"""

# Translate to PyTorch
pytorch_code = interpreter.translate(nova_code)

# Execute the generated code
exec(pytorch_code)

# Now the 'model' variable is available in the namespace
print(model)
```

### Interactive Usage

```python
def interactive_session():
    interpreter = NovaInterpreter()
    
    while True:
        print("\nEnter Nova code (or 'exit' to quit):")
        lines = []
        
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        
        nova_code = "\n".join(lines)
        
        if nova_code.strip().lower() == "exit":
            break
        
        try:
            pytorch_code = interpreter.translate(nova_code)
            print("\nGenerated PyTorch code:")
            print(pytorch_code)
            
            print("\nExplanation:")
            explanation = interpreter.explain_translation(nova_code, pytorch_code)
            print(explanation)
            
            execute = input("\nExecute the code? (y/n): ")
            if execute.lower() == 'y':
                exec(pytorch_code)
                print("Code executed successfully.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_session()
```

## Component Classes

The interpreter consists of several component classes that handle different aspects of the translation process:

### NovaParser

Parses Nova code into an intermediate representation that can be processed by the translator.

```python
from nova.parser import NovaParser

parser = NovaParser()
components = parser.parse(nova_code)
```

### NovaTranslator

Translates the parsed components into PyTorch code.

```python
from nova.translator import NovaTranslator

translator = NovaTranslator()
pytorch_code = translator.translate(components)
```

### NovaExplainer

Generates explanations of the translation process.

```python
from nova.explainer import NovaExplainer

explainer = NovaExplainer()
explanation = explainer.explain(components, pytorch_code)
```

## Extending the Interpreter

The Nova interpreter is designed to be extensible, allowing you to add support for new operations, layers, or frameworks.

### Custom Mappings

You can provide custom mappings when creating an interpreter:

```python
custom_mappings = {
    "error_measures": {
        "custom_loss": "CustomLoss()",
    },
    "improvement_strategies": {
        "custom_optimizer": "CustomOptimizer",
    },
    "activations": {
        "custom_activation": "CustomActivation()",
    }
}

interpreter = NovaInterpreter(custom_mappings=custom_mappings)
```

### Creating Custom Components

For more complex extensions, you can create custom component classes:

```python
from nova.components import LayerComponent

class CustomLayerComponent(LayerComponent):
    def __init__(self, name, params):
        super().__init__(name, params)
        
    def to_pytorch(self):
        # Generate PyTorch code for this component
        return f"nn.CustomLayer({self.params['param']})"

# Register the component
from nova.registry import register_component

register_component("custom_layer", CustomLayerComponent)
```

### Plugin System

Nova supports a plugin system for more comprehensive extensions:

```python
from nova.plugin import NovaPlugin

class MyNovaPlugin(NovaPlugin):
    def __init__(self):
        super().__init__("my_plugin")
        
    def register(self, interpreter):
        # Register components, mappings, etc.
        interpreter.register_mapping("error_measures", "my_loss", "MyLoss()")
        
    def unregister(self, interpreter):
        # Clean up when plugin is unregistered
        interpreter.unregister_mapping("error_measures", "my_loss")

# Use the plugin
from nova import NovaInterpreter
from my_plugin import MyNovaPlugin

interpreter = NovaInterpreter()
my_plugin = MyNovaPlugin()
interpreter.register_plugin(my_plugin)
```

## Error Handling

The Nova interpreter provides detailed error messages when it encounters issues in the Nova code:

```python
try:
    pytorch_code = interpreter.translate(nova_code)
except nova.errors.SyntaxError as e:
    print(f"Syntax error: {e}")
except nova.errors.SemanticError as e:
    print(f"Semantic error: {e}")
except nova.errors.TranslationError as e:
    print(f"Translation error: {e}")
except Exception as e:
    print(f"Unknown error: {e}")
```

## Performance Considerations

- The interpreter is designed for interactive use and performance is generally not a bottleneck
- For large-scale batch processing, consider using a persistent interpreter instance
- The `optimize_code` option can improve the generated code's performance at the cost of readability

## Next Steps

Continue to the [Extensions API](extensions.md) to learn how to extend Nova with custom components and behavior.