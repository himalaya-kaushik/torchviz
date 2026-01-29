# Changelog

All notable changes to DimViz will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-29

### Added
- Initial release of DimViz
- Core tracking functionality with `DimViz` context manager
- `@visualize` decorator for easy function wrapping
- Support for verbose and non-verbose modes
- Memory tracking per operation
- Operation filtering by name
- Maximum entries limit
- Multiple export formats (JSON, CSV, TXT)
- Log comparison functionality
- Rich terminal output (optional)
- Fallback to tabulate for environments without Rich
- Friendly operation names for common PyTorch operations
- Summary statistics (total operations, unique operations, top operations)
- Comprehensive test suite
- Documentation and examples
- Support for:
  - All standard PyTorch tensor operations
  - Multi-input operations
  - In-place operations
  - Scalar tensors
  - Complex model architectures (CNNs, Transformers, etc.)

### Features
- Zero-code-change tracking via context manager
- Automatic shape detection for all tensor inputs/outputs
- Smart filtering of metadata operations
- Performance-conscious design with configurable verbosity
- Export logs for offline analysis
- Compare shape flows between model versions

### Documentation
- Comprehensive README with examples
- API documentation in docstrings
- Contributing guidelines
- Example scripts for common use cases
- Test coverage

[Unreleased]: https://github.com/yourusername/dimviz/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/dimviz/releases/tag/v0.1.0
