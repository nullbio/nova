# Nova Extensions API

The Nova extensions API allows you to extend the core functionality of Nova with custom components, operations, and integrations. This document covers how to create and use Nova extensions.

## Extension Types

There are several ways to extend Nova:

1. **Custom Mappings**: Simple mappings for new terminology
2. **Custom Components**: New component classes for more complex behavior
3. **Plugins**: Comprehensive extensions that can modify multiple aspects of Nova
4. **Framework Adapters**: Support for translating to frameworks other than PyTorch

## Creating Custom Mappings

The simplest way to extend Nova is by adding custom mappings for new terminology.

### Error Measures (Loss Functions)

```python
from nova import NovaInterpreter

# Define custom mappings
custom_error_measures = {
    "weighted_cross_entropy": "nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]))",
    "huber_loss": "nn.HuberLoss(delta=1.0)",
    "contrastive_loss": "ContrastiveLoss(margin=1.0)"
}

# Create interpreter with custom mappings
interpreter = NovaInterpreter(
    custom_mappings={"error_measures": custom_error_measures}
)

# Now you can use these in Nova code
nova_code = """
train model on data_stream:
    measure error using weighted_cross_entropy
    improve using adam with learning rate 0.001
    repeat for 10 learning cycles
"""

pytorch_code = interpreter.translate(nova_code)
```

### Improvement Strategies (Optimizers)

```python
custom_optimizers = {
    "adamw": "optim.AdamW",
    "radam": "optim.RAdam",
    "lookahead": "Lookahead"
}

interpreter = NovaInterpreter(
    custom_mappings={"improvement_strategies": custom_optimizers}
)
```

### Activation Functions

```python
custom_activations = {
    "swish": "nn.SiLU()",
    "mish": "nn.Mish()",
    "gelu": "nn.GELU()"
}

interpreter = NovaInterpreter(
    custom_mappings={"activations": custom_activations}
)
```

## Creating Custom Components

For more complex extensions, you can create custom component classes.

### Component Base Classes

Nova provides several base classes for components:

- `ModelComponent`: Base for model definitions
- `LayerComponent`: Base for layer definitions
- `ActivationComponent`: Base for activation functions
- `TrainingComponent`: Base for training operations

### Example: Custom Layer Component

```python
from nova.components import LayerComponent
from nova.registry import register_component

class SelfAttentionComponent(LayerComponent):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.embed_dim = params.get('embed_dim', 512)
        self.num_heads = params.get('num_heads', 8)
        
    def to_pytorch(self):
        return f"nn.MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})"
    
    def forward_code(self, input_var):
        return f"attn_output, _ = self.{self.name}({input_var}, {input_var}, {input_var})"

# Register the component
register_component("self_attention", SelfAttentionComponent)
```

Now you can use your custom component in Nova code:

```
create processing pipeline transformer:
    add self_attention with embed_dim 512 and num_heads 8
    add layer normalization with 512 features
    add transformation stage fully_connected with 512 inputs and 2048 outputs
    apply relu activation
    add transformation stage fully_connected with 2048 inputs and 512 outputs
    add layer normalization with 512 features
```

### Example: Custom Activation Component

```python
from nova.components import ActivationComponent
from nova.registry import register_component

class GELUComponent(ActivationComponent):
    def __init__(self):
        super().__init__("gelu")
        
    def to_pytorch(self):
        return "nn.GELU()"
    
    def forward_code(self, input_var):
        return f"x = self.gelu({input_var})"

# Register the component
register_component("gelu_activation", GELUComponent)
```

## Creating Plugins

Plugins provide a more structured way to extend Nova with multiple related components.

### Plugin Base Class

```python
from nova.plugin import NovaPlugin

class TransformerPlugin(NovaPlugin):
    def __init__(self):
        super().__init__("transformer")
        
    def register(self, interpreter):
        # Register components
        from .components import (
            SelfAttentionComponent, 
            LayerNormComponent,
            MultiHeadAttentionComponent
        )
        
        interpreter.register_component("self_attention", SelfAttentionComponent)
        interpreter.register_component("layer_norm", LayerNormComponent)
        interpreter.register_component("multi_head_attention", MultiHeadAttentionComponent)
        
        # Register mappings
        interpreter.register_mapping("activations", "gelu", "nn.GELU()")
        
    def unregister(self, interpreter):
        # Clean up when plugin is unregistered
        interpreter.unregister_component("self_attention")
        interpreter.unregister_component("layer_norm")
        interpreter.unregister_component("multi_head_attention")
        interpreter.unregister_mapping("activations", "gelu")
```

### Using Plugins

```python
from nova import NovaInterpreter
from transformer_plugin import TransformerPlugin

# Create interpreter
interpreter = NovaInterpreter()

# Register plugin
transformer_plugin = TransformerPlugin()
interpreter.register_plugin(transformer_plugin)

# Now you can use transformer components in Nova code
nova_code = """
create processing pipeline transformer:
    add self_attention with embed_dim 512 and num_heads 8
    add layer_norm with 512 features
"""

pytorch_code = interpreter.translate(nova_code)
```

## Framework Adapters

Nova can be extended to generate code for frameworks other than PyTorch.

### Creating a Framework Adapter

```python
from nova.adapters import FrameworkAdapter

class TensorFlowAdapter(FrameworkAdapter):
    def __init__(self):
        super().__init__("tensorflow")
        
    def translate_imports(self, components):
        return "import tensorflow as tf\nfrom tensorflow import keras\n"
    
    def translate_model(self, model_component):
        # Translate model definition to TensorFlow code
        # ...
        
    def translate_layer(self, layer_component):
        # Translate layer to TensorFlow
        if layer_component.type == "fully_connected":
            return f"keras.layers.Dense({layer_component.outputs})"
        # ...
        
    def translate_activation(self, activation_component):
        # Translate activation to TensorFlow
        if activation_component.name == "relu":
            return "keras.activations.relu"
        # ...
        
    def translate_training(self, training_component):
        # Translate training code to TensorFlow
        # ...
```

### Using a Framework Adapter

```python
from nova import NovaInterpreter
from tensorflow_adapter import TensorFlowAdapter

# Create interpreter with TensorFlow adapter
interpreter = NovaInterpreter(adapter=TensorFlowAdapter())

# Now Nova code will be translated to TensorFlow
nova_code = """
create processing pipeline model:
    add transformation stage fully_connected with 784 inputs and 128 outputs
    apply relu activation
    add transformation stage fully_connected with 128 inputs and 10 outputs
"""

tensorflow_code = interpreter.translate(nova_code)
```

## Handling Special Syntax

You can extend Nova's parser to handle special syntax for your custom components.

### Custom Parser Rules

```python
from nova.parser import NovaParser, register_rule
import re

class CustomParser(NovaParser):
    def __init__(self):
        super().__init__()
        
    @register_rule
    def parse_attention(self, line, context):
        pattern = r'add attention from (\w+) to (\w+) with (\d+) heads'
        match = re.match(pattern, line.strip())
        
        if match:
            from_var, to_var, num_heads = match.groups()
            return {
                "type": "attention",
                "from_var": from_var,
                "to_var": to_var,
                "num_heads": int(num_heads)
            }
        
        return None

# Use the custom parser
from nova import NovaInterpreter

interpreter = NovaInterpreter(parser=CustomParser())
```

## Creating Domain-Specific Extensions

Nova can be extended with domain-specific knowledge for areas like computer vision, NLP, or reinforcement learning.

### Example: Computer Vision Extension

```python
from nova.plugin import NovaPlugin

class ComputerVisionPlugin(NovaPlugin):
    def __init__(self):
        super().__init__("computer_vision")
        
    def register(self, interpreter):
        # Register components for computer vision
        from .components import (
            ResNetBlockComponent,
            InceptionBlockComponent,
            DepthwiseSeparableConvComponent
        )
        
        interpreter.register_component("resnet_block", ResNetBlockComponent)
        interpreter.register_component("inception_block", InceptionBlockComponent)
        interpreter.register_component("depthwise_conv", DepthwiseSeparableConvComponent)
        
        # Register mappings for common CV operations
        cv_transforms = {
            "random_crop": "transforms.RandomCrop",
            "color_jitter": "transforms.ColorJitter",
            "random_erasing": "transforms.RandomErasing"
        }
        
        interpreter.register_mappings("data_transforms", cv_transforms)
```

## Publishing and Sharing Extensions

Extensions can be packaged and distributed as Python packages.

### Package Structure

```
nova-cv-extension/
├── setup.py
├── README.md
├── nova_cv/
│   ├── __init__.py
│   ├── plugin.py
│   └── components/
│       ├── __init__.py
│       ├── resnet.py
│       ├── inception.py
│       └── depthwise.py
```

### setup.py

```python
from setuptools import setup, find_packages

setup(
    name="nova-cv",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "nova-dl>=0.1.0",
        "torch>=1.8.0",
        "torchvision>=0.9.0"
    ],
    entry_points={
        "nova.plugins": [
            "cv=nova_cv.plugin:ComputerVisionPlugin"
        ]
    }
)
```

### Auto-discovery

With the entry points configuration, Nova can automatically discover and load plugins:

```python
from nova import NovaInterpreter

# Create interpreter with auto-discovery
interpreter = NovaInterpreter(discover_plugins=True)
```

## Next Steps

- Review the [examples](../examples/basic-models.md) to see how to use extensions
- Check out the [community page](../community/contributing.md) for guidelines on contributing extensions
- Explore the [roadmap](../community/roadmap.md) to see planned extensions and features