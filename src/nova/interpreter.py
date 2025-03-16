"""
Nova Interpreter: Translates Nova natural language code to PyTorch code.
"""

import re
from typing import Dict, List, Any, Optional, Tuple

class NovaInterpreter:
    """
    The Nova Interpreter translates natural language Nova code into executable PyTorch code.
    
    This class handles the parsing, analysis, and translation of Nova code.
    """
    
    def __init__(self):
        """Initialize the Nova interpreter with mappings and patterns."""
        # Mappings for error measures (loss functions)
        self.error_measures = {
            "mean_squared_error": "nn.MSELoss()",
            "cross_entropy": "nn.CrossEntropyLoss()",
            "binary_cross_entropy": "nn.BCELoss()",
            "negative_log_likelihood": "nn.NLLLoss()",
            "kl_divergence": "nn.KLDivLoss()"
        }
        
        # Mappings for improvement strategies (optimizers)
        self.improvement_strategies = {
            "gradient_descent": "optim.SGD",
            "sgd": "optim.SGD",
            "adam": "optim.Adam",
            "rmsprop": "optim.RMSprop",
            "adagrad": "optim.Adagrad"
        }
        
        # Mappings for activation functions
        self.activations = {
            "relu": "nn.ReLU()",
            "sigmoid": "nn.Sigmoid()",
            "tanh": "nn.Tanh()",
            "softmax": "nn.Softmax(dim=1)",
            "leaky_relu": "nn.LeakyReLU()"
        }
        
        # Compile regex patterns for parsing
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for parsing Nova code."""
        # Pattern for model creation
        self.model_pattern = re.compile(
            r'create\s+(?:processing|image|sequence)\s+pipeline\s+(\w+)'
        )
        
        # Pattern for layer definitions
        self.layer_patterns = {
            "fully_connected": re.compile(
                r'add\s+transformation\s+stage\s+fully_connected\s+with\s+(\d+)\s+inputs\s+and\s+(\d+)\s+outputs'
            ),
            "feature_detector": re.compile(
                r'add\s+feature\s+detector\s+with\s+(\d+)\s+input\s+channels,\s+(\d+)\s+output\s+channels(?:\s+and\s+(\d+)x(\d+)\s+filter\s+size)?'
            ),
            "downsampling": re.compile(
                r'add\s+downsampling\s+using\s+(max|average)\s+method\s+with\s+size\s+(\d+)x(\d+)'
            )
        }
        
        # Pattern for activation functions
        self.activation_pattern = re.compile(
            r'apply\s+(\w+)\s+activation'
        )
        
        # Pattern for training definition
        self.training_pattern = re.compile(
            r'train\s+(\w+)\s+on\s+(\w+)'
        )
        
        # Pattern for error measure
        self.error_pattern = re.compile(
            r'measure\s+error\s+using\s+(\w+)'
        )
        
        # Pattern for improvement strategy
        self.improvement_pattern = re.compile(
            r'improve\s+using\s+(\w+)\s+with\s+learning\s+rate\s+([\d\.]+)'
        )
        
        # Pattern for learning cycles
        self.cycles_pattern = re.compile(
            r'repeat\s+for\s+(\d+)\s+learning\s+cycles'
        )
    
    def translate(self, nova_code: str) -> str:
        """
        Translate Nova code to PyTorch code.
        
        Args:
            nova_code: The Nova code to translate
            
        Returns:
            str: The corresponding PyTorch code
        """
        # Parse the Nova code to extract components
        components = self._parse_nova_code(nova_code)
        
        # Generate imports section
        imports = self._generate_imports(components)
        
        # Generate model class
        model_class = self._generate_model_class(components)
        
        # Generate training code
        training_code = self._generate_training_code(components)
        
        # Combine all sections
        pytorch_code = "\n".join([imports, model_class, training_code])
        
        return pytorch_code
    
    def _parse_nova_code(self, nova_code: str) -> Dict[str, Any]:
        """
        Parse Nova code to extract key components.
        
        Args:
            nova_code: The Nova code to parse
            
        Returns:
            Dict: A dictionary of parsed components
        """
        components = {
            "model_name": None,
            "layers": [],
            "activations": [],
            "training": {
                "data_stream": None,
                "error_measure": None,
                "improvement_strategy": None,
                "learning_rate": 0.01,
                "cycles": 10
            }
        }
        
        # Split the code into lines for processing
        lines = nova_code.strip().split('\n')
        
        # Process each line
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith("#") or not line:
                continue
            
            # Parse model definition
            model_match = self.model_pattern.search(line)
            if model_match:
                components["model_name"] = model_match.group(1)
                continue
            
            # Parse fully connected layer
            fc_match = self.layer_patterns["fully_connected"].search(line)
            if fc_match:
                components["layers"].append({
                    "type": "fully_connected",
                    "inputs": int(fc_match.group(1)),
                    "outputs": int(fc_match.group(2))
                })
                continue
            
            # Parse feature detector (convolutional layer)
            conv_match = self.layer_patterns["feature_detector"].search(line)
            if conv_match:
                filter_size = (3, 3)  # Default filter size
                if conv_match.group(3) and conv_match.group(4):
                    filter_size = (int(conv_match.group(3)), int(conv_match.group(4)))
                
                components["layers"].append({
                    "type": "feature_detector",
                    "in_channels": int(conv_match.group(1)),
                    "out_channels": int(conv_match.group(2)),
                    "filter_size": filter_size
                })
                continue
                
            # Parse downsampling (pooling layer)
            pool_match = self.layer_patterns["downsampling"].search(line)
            if pool_match:
                components["layers"].append({
                    "type": "downsampling",
                    "method": pool_match.group(1),
                    "size": (int(pool_match.group(2)), int(pool_match.group(3)))
                })
                continue
            
            # Parse activation function
            act_match = self.activation_pattern.search(line)
            if act_match:
                components["activations"].append(act_match.group(1))
                continue
            
            # Parse training definition
            train_match = self.training_pattern.search(line)
            if train_match:
                components["training"]["model_name"] = train_match.group(1)
                components["training"]["data_stream"] = train_match.group(2)
                continue
            
            # Parse error measure
            error_match = self.error_pattern.search(line)
            if error_match:
                components["training"]["error_measure"] = error_match.group(1)
                continue
            
            # Parse improvement strategy
            improve_match = self.improvement_pattern.search(line)
            if improve_match:
                components["training"]["improvement_strategy"] = improve_match.group(1)
                components["training"]["learning_rate"] = float(improve_match.group(2))
                continue
            
            # Parse learning cycles
            cycles_match = self.cycles_pattern.search(line)
            if cycles_match:
                components["training"]["cycles"] = int(cycles_match.group(1))
                continue
        
        return components
    
    def _generate_imports(self, components: Dict[str, Any]) -> str:
        """
        Generate the imports section of the PyTorch code.
        
        Args:
            components: The parsed components from Nova code
            
        Returns:
            str: The imports section
        """
        imports = [
            "import torch",
            "import torch.nn as nn",
            "import torch.optim as optim"
        ]
        
        # Add specific imports based on the components
        has_conv = any(layer["type"] == "feature_detector" for layer in components["layers"])
        if has_conv:
            imports.append("import torch.nn.functional as F")
        
        return "\n".join(imports)
    
    def _generate_model_class(self, components: Dict[str, Any]) -> str:
        """
        Generate the model class definition in PyTorch.
        
        Args:
            components: The parsed components from Nova code
            
        Returns:
            str: The model class definition
        """
        model_name = components["model_name"]
        if not model_name:
            model_name = "NovaModel"
        
        # Class header
        class_lines = [
            f"\n\nclass {model_name.capitalize()}(nn.Module):",
            "    def __init__(self):",
            f"        super({model_name.capitalize()}, self).__init__()"
        ]
        
        # Initialize layers
        for i, layer in enumerate(components["layers"]):
            if layer["type"] == "fully_connected":
                class_lines.append(f"        self.fc{i+1} = nn.Linear({layer['inputs']}, {layer['outputs']})")
            
            elif layer["type"] == "feature_detector":
                k_size = layer["filter_size"][0]
                class_lines.append(f"        self.conv{i+1} = nn.Conv2d({layer['in_channels']}, {layer['out_channels']}, kernel_size={k_size})")
            
            elif layer["type"] == "downsampling":
                k_size = layer["size"][0]
                method = "nn.MaxPool2d" if layer["method"] == "max" else "nn.AvgPool2d"
                class_lines.append(f"        self.pool{i+1} = {method}(kernel_size={k_size})")
        
        # Initialize activations
        activation_count = 0
        for act in components["activations"]:
            if act in self.activations:
                activation_count += 1
                class_lines.append(f"        self.{act}{activation_count} = {self.activations[act]}")
        
        # Forward method
        class_lines.extend([
            "",
            "    def forward(self, x):"
        ])
        
        # Layer forward passes
        act_index = 0
        for i, layer in enumerate(components["layers"]):
            if layer["type"] == "fully_connected":
                # If it's the first layer and we have a CNN before, flatten
                if i == 0 and any(l["type"] in ["feature_detector", "downsampling"] for l in components["layers"]):
                    class_lines.append("        x = x.view(x.size(0), -1)  # Flatten")
                class_lines.append(f"        x = self.fc{i+1}(x)")
            
            elif layer["type"] == "feature_detector":
                class_lines.append(f"        x = self.conv{i+1}(x)")
            
            elif layer["type"] == "downsampling":
                class_lines.append(f"        x = self.pool{i+1}(x)")
            
            # Apply activation if available
            if act_index < len(components["activations"]):
                act = components["activations"][act_index]
                if act in self.activations:
                    class_lines.append(f"        x = self.{act}{act_index+1}(x)")
                    act_index += 1
        
        # Return statement
        class_lines.append("        return x")
        
        # Model instantiation
        class_lines.extend([
            "",
            f"model = {model_name.capitalize()}()"
        ])
        
        return "\n".join(class_lines)
    
    def _generate_training_code(self, components: Dict[str, Any]) -> str:
        """
        Generate the training code section in PyTorch.
        
        Args:
            components: The parsed components from Nova code
            
        Returns:
            str: The training code
        """
        training = components["training"]
        
        # If no training information is provided, return empty string
        if not training["data_stream"]:
            return ""
        
        # Get error measure
        error_measure = self.error_measures.get(
            training["error_measure"], 
            "nn.CrossEntropyLoss()"
        )
        
        # Get improvement strategy
        strategy = self.improvement_strategies.get(
            training["improvement_strategy"], 
            "optim.Adam"
        )
        
        lines = [
            f"\n# Define loss function and optimizer",
            f"criterion = {error_measure}",
            f"optimizer = {strategy}(model.parameters(), lr={training['learning_rate']})"
        ]
        
        # Training loop
        lines.extend([
            "",
            "# Training loop",
            f"for epoch in range({training['cycles']}):",
            "    running_loss = 0.0",
            f"    for inputs, labels in {training['data_stream']}:",
            "        # Zero the parameter gradients",
            "        optimizer.zero_grad()",
            "",
            "        # Forward + backward + optimize",
            "        outputs = model(inputs)",
            "        loss = criterion(outputs, labels)",
            "        loss.backward()",
            "        optimizer.step()",
            "",
            "        # Print statistics",
            "        running_loss += loss.item()",
            "",
            f"    print(f'Epoch {{{}}}, Loss: {{{}}}'.format(epoch + 1, running_loss))"
        ])
        
        return "\n".join(lines)
    
    def explain_translation(self, nova_code: str, pytorch_code: str) -> str:
        """
        Generate an explanation of the translation from Nova to PyTorch.
        
        Args:
            nova_code: The original Nova code
            pytorch_code: The translated PyTorch code
            
        Returns:
            str: An explanation of the translation
        """
        components = self._parse_nova_code(nova_code)
        
        explanation = ["Translation Explanation:"]
        
        # Explain model creation
        if components["model_name"]:
            explanation.append(f"1. Created a PyTorch model class called '{components['model_name'].capitalize()}'")
        
        # Explain layers
        for i, layer in enumerate(components["layers"]):
            if layer["type"] == "fully_connected":
                explanation.append(f"{i+2}. Added a fully connected layer with {layer['inputs']} inputs and {layer['outputs']} outputs")
            elif layer["type"] == "feature_detector":
                explanation.append(f"{i+2}. Added a convolutional layer with {layer['in_channels']} input channels, {layer['out_channels']} output channels, and {layer['filter_size'][0]}x{layer['filter_size'][1]} kernel size")
            elif layer["type"] == "downsampling":
                method = "max pooling" if layer["method"] == "max" else "average pooling"
                explanation.append(f"{i+2}. Added a {method} layer with kernel size {layer['size'][0]}x{layer['size'][1]}")
        
        # Explain activations
        for i, act in enumerate(components["activations"]):
            explanation.append(f"{len(components['layers'])+i+2}. Applied {act} activation function")
        
        # Explain training
        if components["training"]["data_stream"]:
            next_num = len(components["layers"]) + len(components["activations"]) + 2
            explanation.append(f"{next_num}. Set up training with:")
            explanation.append(f"   - Loss function: {components['training']['error_measure'] or 'cross_entropy'}")
            explanation.append(f"   - Optimizer: {components['training']['improvement_strategy'] or 'adam'} with learning rate {components['training']['learning_rate']}")
            explanation.append(f"   - Training for {components['training']['cycles']} epochs")
        
        return "\n".join(explanation)