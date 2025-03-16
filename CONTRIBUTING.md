# Contributing to Nova

Thank you for your interest in contributing to Nova! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We expect all contributors to be respectful, inclusive, and considerate of others.

## How to Contribute

There are many ways to contribute to Nova:

1. **Report bugs and suggest features**: Open issues on GitHub to report bugs or suggest new features
2. **Improve documentation**: Help improve or translate the documentation
3. **Write code**: Implement new features or fix bugs
4. **Review code**: Review pull requests from other contributors
5. **Share examples**: Create examples showing how to use Nova for various tasks

## Development Workflow

1. **Fork the repository**: Create your own fork of the Nova repository
2. **Clone the repository**: Clone your fork to your local machine
3. **Create a branch**: Create a new branch for your changes
4. **Make changes**: Implement your changes or fixes
5. **Run tests**: Make sure all tests pass with your changes
6. **Commit and push**: Commit your changes and push them to your fork
7. **Open a pull request**: Open a pull request to merge your changes into the main repository

## Development Environment

1. **Set up your environment**:
   ```bash
   git clone https://github.com/yourusername/nova.git
   cd nova
   pip install -e .
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Coding Style

We follow PEP 8 style guidelines for Python code. Additionally:

- Use docstrings for all modules, classes, and functions
- Write clear, descriptive variable and function names
- Include type hints for function parameters and return values
- Add comments for complex code sections

## Testing

Before submitting a pull request, make sure your code passes all tests:

```bash
pytest
```

If you add new features, please include appropriate tests.

## Documentation

When adding new features or making changes, please update the documentation accordingly. Documentation is written in Markdown and is located in the `docs` directory.

## Submitting a Pull Request

1. Update your branch with the latest changes from the main repository
2. Make sure all tests pass
3. Push your changes to your fork
4. Submit a pull request with a clear description of your changes
5. Address any feedback from code reviews

## License

By contributing to Nova, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).