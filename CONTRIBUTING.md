# Contributing to PySpark Code Analyzer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/codellama.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment and install dependencies: `pip install -r requirements.txt`
5. Create a feature branch: `git checkout -b feature/your-feature-name`

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions, classes, and modules
- Keep functions focused and single-purpose
- Maximum line length: 100 characters

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Run tests: `python -m unittest discover tests`
- Aim for high test coverage

### Documentation
- Update README.md if adding new features
- Add inline comments for complex logic
- Update docstrings when modifying functions
- Include usage examples for new features

## 🔧 Areas for Contribution

### High Priority
- [ ] Add batch file processing capability
- [ ] Implement web UI for easier interaction
- [ ] Add support for additional LLM providers (OpenAI, Anthropic)
- [ ] Enhance error handling and recovery
- [ ] Add progress bars for long-running operations

### Medium Priority
- [ ] Implement caching for repeated queries
- [ ] Add export functionality (PDF, JSON reports)
- [ ] Create visualization for code structure
- [ ] Add support for other languages (Java, Scala)
- [ ] Implement query history and favorites

### Lower Priority
- [ ] Add Docker containerization
- [ ] Create CI/CD pipeline
- [ ] Add performance benchmarks
- [ ] Implement multi-user support
- [ ] Add API endpoint for programmatic access

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages and stack traces
- Relevant log files

## ✨ Feature Requests

When requesting features, please include:
- Use case description
- Expected behavior
- Why this feature would be valuable
- Any implementation ideas

## 📥 Pull Request Process

1. Update documentation as needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md with your changes
5. Submit PR with clear description
6. Link related issues in PR description

### PR Title Format
- `feat: Add new feature`
- `fix: Fix bug description`
- `docs: Update documentation`
- `test: Add tests`
- `refactor: Code refactoring`
- `style: Code style changes`

## 🤝 Code Review

All PRs require review before merging:
- Code quality and style
- Test coverage
- Documentation completeness
- Performance considerations
- Security implications

## 📧 Contact

Questions? Open an issue or discussion on GitHub.

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project.
