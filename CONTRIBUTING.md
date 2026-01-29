# Contributing to DimViz

Thank you for your interest in contributing to DimViz! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/dimviz.git
   cd dimviz
   ```
3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Setting Up Your Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,rich]"
```

### Making Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and ensure they follow our coding standards

3. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Run code formatters**:
   ```bash
   black dimviz tests examples
   isort dimviz tests examples
   ```

5. **Run linters**:
   ```bash
   flake8 dimviz tests
   mypy dimviz
   ```

### Testing

We use pytest for testing. Please ensure all tests pass before submitting a PR.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=dimviz --cov-report=html

# Run specific test file
pytest tests/test_dimviz.py -v
```

#### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive test names
- Include docstrings explaining what you're testing

Example:
```python
def test_basic_tracking():
    """Test that basic operation tracking works correctly."""
    with DimViz(show_summary=False) as viz:
        x = torch.randn(2, 10)
        y = x + 1
    
    assert len(viz.get_log()) > 0
```

### Code Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use Black for formatting
- Use isort for import sorting

### Commit Messages

Write clear, descriptive commit messages:
- Use present tense ("Add feature" not "Added feature")
- Start with a verb ("Fix", "Add", "Update", "Remove")
- Keep first line under 50 characters
- Add detailed explanation in the body if needed

Good examples:
```
Add memory tracking feature

Implement memory allocation tracking per operation.
Includes new column in output table and summary statistics.
```

```
Fix shape detection for scalar tensors

Previously scalar tensors would show empty shapes.
Now they display as "scalar" for clarity.
```

### Pull Request Process

1. **Update documentation** if you've added features
2. **Add tests** for new functionality
3. **Update CHANGELOG.md** with your changes
4. **Ensure all tests pass**
5. **Submit the PR** with a clear description

PR Description Template:
```markdown
## Description
Brief description of what this PR does

## Motivation
Why is this change needed?

## Changes
- List of changes made
- Another change

## Testing
How have you tested these changes?

## Checklist
- [ ] Tests pass
- [ ] Code formatted with Black
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Types of Contributions

### Bug Reports

Submit bug reports through GitHub Issues:
- Use a clear, descriptive title
- Describe the expected behavior
- Describe the actual behavior
- Provide a minimal code example to reproduce
- Include your environment (Python version, PyTorch version, OS)

### Feature Requests

We welcome feature requests! Please:
- Use a clear, descriptive title
- Explain the feature and its benefits
- Provide examples of how it would be used
- Consider implementation complexity

### Code Contributions

Areas where contributions are especially welcome:
- Additional export formats
- Performance optimizations
- Better error messages
- Documentation improvements
- More examples
- Integration with other tools

## Code Review Process

1. A maintainer will review your PR
2. They may request changes or ask questions
3. Make requested changes and push to your branch
4. Once approved, a maintainer will merge your PR

## Community Guidelines

- Be respectful and constructive
- Help others in issues and discussions
- Follow the code of conduct
- Share knowledge and learn together

## Questions?

- Open an issue for questions about the code
- Check existing issues and PRs first
- Ask in discussions for general questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in the CHANGELOG and README.

Thank you for contributing to DimViz! 🎉
