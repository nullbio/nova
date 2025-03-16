# Contributing to Nova

Thank you for your interest in contributing to Nova! This document provides guidelines and instructions for contributing to the project. Nova is an open-source project, and we welcome contributions of all kinds, from bug reports to feature implementations.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We expect all contributors to be respectful, inclusive, and considerate of others.

## Ways to Contribute

There are many ways to contribute to Nova:

### 1. Report Bugs

If you find a bug, please report it by creating an issue on GitHub. When reporting bugs, please include:

- A clear, descriptive title
- A detailed description of the issue
- Steps to reproduce the bug
- Expected behavior vs. actual behavior
- System information (OS, Python version, PyTorch version)
- Error messages or screenshots if applicable

### 2. Suggest Features

If you have an idea for a new feature or enhancement, please create an issue on GitHub:

- Clearly describe the feature
- Explain why it would be valuable
- Provide examples of how it would be used
- If possible, outline how it might be implemented

### 3. Improve Documentation

Documentation improvements are always welcome:

- Fixing typos or grammar
- Clarifying confusing explanations
- Adding examples or tutorials
- Translating documentation to other languages

### 4. Write Code

Contributing code is a great way to help improve Nova:

- Implement new features
- Fix bugs
- Optimize existing code
- Add test coverage

### 5. Share Examples

Creating examples showing how to use Nova for various tasks is very valuable:

- Create examples for different domains (computer vision, NLP, etc.)
- Show how to use Nova for specific tasks
- Compare Nova with other approaches

## Development Workflow

### Setting Up Your Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/nova.git
   cd nova
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Making Changes

1. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**: Implement your feature or bug fix.

3. **Follow the code style**:
   - Use PEP 8 style guidelines
   - Add type hints to new functions and methods
   - Write docstrings in the Google style
   - Add appropriate comments for complex code

4. **Write tests**:
   - Add tests for new features
   - Fix or update tests for bug fixes
   - Ensure all tests pass

5. **Update documentation**:
   - Update or add docstrings
   - Update relevant documentation files
   - Add examples if applicable

### Submitting a Pull Request

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add a descriptive commit message"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request**: Go to the Nova repository on GitHub and create a pull request from your fork.

4. **Describe your changes**:
   - Provide a clear title and description
   - Reference any related issues
   - Explain the motivation for the changes
   - Describe how you tested the changes

5. **Address review feedback**: If maintainers suggest changes, make them and push the updates to your branch.

## Development Guidelines

### Code Style

We use [Black](https://black.readthedocs.io/) and [isort](https://pycqa.github.io/isort/) for code formatting. You can format your code with:

```bash
black .
isort .
```

We use [flake8](https://flake8.pycqa.org/) for linting. You can check your code with:

```bash
flake8
```

We use [mypy](https://mypy.readthedocs.io/) for type checking. You can check your code with:

```bash
mypy nova
```

### Testing

We use [pytest](https://docs.pytest.org/) for testing. You can run the tests with:

```bash
pytest
```

For test coverage, we use [pytest-cov](https://pytest-cov.readthedocs.io/). You can check coverage with:

```bash
pytest --cov=nova
```

### Documentation

We use [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/) for documentation. You can build and serve the documentation locally with:

```bash
cd website
pip install -r requirements.txt
mkdocs serve
```

The documentation will be available at [http://localhost:8000](http://localhost:8000).

### Continuous Integration

We use GitHub Actions for continuous integration. Pull requests will be automatically tested, and you will be notified of any failures.

## Project Structure

Understanding the project structure will help you contribute effectively:

```
nova/
├── docs/            # Documentation
├── examples/        # Example usage
├── src/             # Source code
│   └── nova/        # Main package
│       ├── __init__.py
│       ├── interpreter.py
│       ├── parser.py
│       ├── translator.py
│       ├── components/
│       ├── adapters/
│       └── plugins/
├── tests/           # Test suite
├── website/         # Documentation website
└── setup.py         # Package configuration
```

## Extending Nova

If you're interested in extending Nova with new components or features, see the [Extensions API](../api/extensions.md) documentation.

### Adding a New Component

1. Create a new component class in the appropriate module:
   ```python
   # src/nova/components/my_component.py
   from nova.components.base import Component

   class MyComponent(Component):
       def __init__(self, name, params):
           super().__init__(name, params)
           
       def to_pytorch(self):
           # Generate PyTorch code
           return "..."
   ```

2. Register the component in the registry:
   ```python
   # src/nova/components/__init__.py
   from nova.registry import register_component
   from .my_component import MyComponent

   register_component("my_component", MyComponent)
   ```

3. Add tests for the component:
   ```python
   # tests/components/test_my_component.py
   def test_my_component():
       # Test the component
       ...
   ```

4. Add documentation for the component:
   ```markdown
   <!-- docs/api/components.md -->
   ## MyComponent

   Description of the component and how to use it.

   ```

### Adding a New Feature

1. Implement the feature:
   ```python
   # src/nova/feature.py
   def my_feature():
       # Implement the feature
       ...
   ```

2. Add tests for the feature:
   ```python
   # tests/test_feature.py
   def test_my_feature():
       # Test the feature
       ...
   ```

3. Add documentation for the feature:
   ```markdown
   <!-- docs/features.md -->
   ## My Feature

   Description of the feature and how to use it.
   ```

## Community

Join our community to discuss Nova development, ask questions, and share ideas:

- **GitHub Discussions**: For questions, ideas, and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Slack Channel**: For real-time discussion (coming soon)

## Acknowledgements

Your contributions are greatly appreciated. Contributors will be acknowledged in the project's README and release notes.

## License

By contributing to Nova, you agree that your contributions will be licensed under the project's [MIT License](https://github.com/nova-team/nova/blob/main/LICENSE).