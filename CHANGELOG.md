# Changelog

All notable changes to the PySpark Code Analyzer project will be documented in this file.

## [1.0.0] - 2025-11-23

### Added
- Initial release of PySpark Code Analyzer
- Token-based code chunking with tiktoken (25k tokens, 5k overlap)
- ChromaDB integration for vector storage and retrieval
- DSPy-based prompt optimization for code analysis
- LangGraph workflow orchestration for Stage 1 and Stage 2
- CodeLlama-7B integration via Ollama
- Interactive query mode for code analysis
- Command-line interface with multiple modes (process, query, interactive, info, clear)
- Comprehensive configuration system (YAML + environment variables)
- Example PySpark ETL pipeline
- Unit tests for core modules
- Quick start script for easy setup
- Detailed README with usage instructions
- Logging system with file and console output

### Features
- **Stage 1**: File reading, chunking, and embedding in ChromaDB
- **Stage 2**: Semantic search and AI-powered code analysis
- **DSPy Signatures**: CodeAnalysis, BusinessLogicExtractor, CodeExplanation
- **LangGraph Workflows**: Separate orchestration for each stage
- **Configuration Management**: Flexible YAML and .env configuration

### Documentation
- Comprehensive README.md
- CONTRIBUTING.md guidelines
- Example PySpark file with business logic
- API documentation in docstrings

### Testing
- Unit tests for CodeChunker
- Unit tests for ChromaDBManager
- Test data generation utilities

## [Unreleased]

### Planned Features
- Batch file processing
- Web UI interface
- Additional LLM provider support
- Export functionality (PDF, JSON)
- Query caching
- Progress indicators
- Docker containerization
