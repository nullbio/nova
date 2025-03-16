# CLAUDE.md - Agent Instructions

You are a PyTorch and Python programming and machine learning expert, who can use the brave MCP to browse the latest PyTorch documentation. Our goal is to create an easier alternative to program in Python and PyTorch, by using your LLM as a natural language interpreter called Nova.

The current issue with machine learning is the vagueness of the terms, nomenclature, methods and so forth if the programmer is not well versed in mathematics or machine learning.

Our goal is to fix this issue by creating a natural language interpreter in the form of a system prompt, with accompanying documentation website, that can translate from natural language to PyTorch and Python code.

It uses familiar data processing concepts; clear relationship between input, processing, and output; natural explanation for backpropagation as feedback.

## System Architecture

The Nova system consists of the following key components:

1. **Nova Language** - A natural language syntax for describing ML models
2. **Interpreter** - Translates Nova language to PyTorch code
3. **Documentation Website** - Explains the system and provides examples

## Key Files and Their Purposes

When modifying the system, these are the main files to inspect and/or modify:

### Language Definition
- `/website/docs/language-guide/*.md` - Contains the language specifications
- `/website/docs/examples/*.md` - Contains examples of Nova syntax and equivalent PyTorch code

### Website Components
- `/website/docs/stylesheets/nova-highlight.css` - Defines the syntax highlighting styles
- `/website/docs/javascripts/nova-highlight.js` - Contains the syntax highlighting logic
- `/website/docs/assets/vscode/` - VS Code extension files for Nova syntax highlighting
- `/website/mkdocs.yml` - MkDocs configuration file for the documentation site

### Interpreter Components
- [Add paths to actual interpreter files when they are implemented]

## Extending the System

When adding new features or modifying the system, follow these steps:

1. **Update Language Definition**:
   - Add new syntax rules to `/website/docs/language-guide/syntax.md`
   - Ensure coherence with existing language features
   - Test with simple examples first

2. **Add Examples**:
   - Create examples demonstrating the new functionality
   - Include both Nova code and equivalent PyTorch code
   - Place examples in `/website/docs/examples/`

3. **Update Syntax Highlighting**:
   - Add new keywords or patterns to `/website/docs/javascripts/nova-highlight.js`
   - Update CSS styles in `/website/docs/stylesheets/nova-highlight.css` if needed
   - Update VS Code extension grammar in `/website/docs/assets/vscode/nova.tmLanguage.json`

4. **Update Interpreter**:
   - [Add specific steps for updating the interpreter when implemented]

5. **Test Thoroughly**:
   - Test new features with various inputs and edge cases
   - Ensure backward compatibility with existing code
   - Verify syntax highlighting works correctly in the documentation and VS Code

6. **Document Changes**:
   - Update the documentation to explain new features
   - Add to the appropriate sections in the language guide
   - Consider adding a blog post or changelog entry for significant changes

Please expand on this documentation as the system evolves.
