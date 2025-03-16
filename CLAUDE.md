# CLAUDE.md - Agent Instructions

You are a Pytorch and Python programming and machine learning expert, who can use the brave MCP to browse the latest pytorch documentation. Our goal is to create an easier alternative to program in Python and Pytorch, by using your LLM as a natural language interpreter.

The current issue with machine learning learning is the vagueness of the terms, nomenclature, methods and so forth in using these if the programmer is not well versed in mathematics or machine learning.

Our goal is to fix this issue by creating a natural language interpreter in the form of a system prompt, with accompany documentation website, that can translate from this natural language to pytorch and python code.

Here is an example of the pipeline approach:

```
Data Pipeline Framework
We want to reimagine neural networks as data transformation pipelines, a concept familiar to most programmers who have worked with ETL processes, Unix pipes, or functional programming chains.
Core Concept: ML as configurable data pipelines with feedback loops

from easyml.pipeline import Pipeline
from easyml.transforms import Linear, Activation, Pooling
from easyml.feedback import MeanSquaredError
from easyml.tuner import GradientTuner

# Create a neural network
image_processor = Pipeline("image_classifier")
image_processor.add(Linear(inputs=784, outputs=128))
image_processor.add(Activation(type="relu"))
image_processor.add(Linear(inputs=128, outputs=10))

# Train the model
trainer = image_processor.create_trainer(
    quality_checker=MeanSquaredError(),
    parameter_tuner=GradientTuner(step_size=0.01)
)
trainer.improve(training_data, expected_outputs, iterations=10)
```

Mappings:

Tensors → DataArrays (multi-dimensional data structures)
Layers → Transforms (data transformation units)
Weights → TransformParameters (configurable settings)
Forward pass → Process (running data through pipeline)
Backward pass → Feedback (adjusting based on results)
Loss functions → QualityCheckers (measuring output correctness)
Optimizers → ParameterTuners (systematic parameter adjustment)

It uses familiar data processing concepts; clear relationship between input, processing, and output; natural explanation for backpropagation as feedback

Please read the ABOUT.md for an overview of the starting point to build from.

We want to expand on this, and build a documentation website.

Begin by reading the ABOUT.md, make sure you are familiar with the Pytorch library and all of its methods and functions, etc, and then begin writing documentations for the project, building the website, expanding on everything and making sure every aspect of the library and all functionality is covered by our interpreter, and well defined in our website docs.
