# PySpark Code Analyzer - Project Summary

## 📋 Project Overview

A complete Python application for analyzing PySpark code using AI technologies. The system processes Python/PySpark files, extracts business logic, and provides human-readable insights through natural language queries.

## 🎯 Key Requirements Implemented

✅ **Stage 1: Chunking and Embedding**
- Reads PySpark file contents
- Chunks code with 25,000 tokens per chunk
- 5,000 token overlap between chunks
- Stores chunks in ChromaDB with vector embeddings

✅ **Stage 2: Query and Analysis**
- Accepts user queries about code
- Retrieves relevant chunks from ChromaDB
- Passes chunks to CodeLlama:7b for analysis
- Returns human-readable explanations

✅ **DSPy Integration**
- Incremental prompt evolution
- Chain-of-Thought reasoning
- Few-shot learning capabilities
- Bootstrap optimization

✅ **LangGraph Orchestration**
- Separate workflows for Stage 1 and Stage 2
- State management and error handling
- Modular node-based architecture

## 📁 Project Structure

```
codellama/
├── main.py                          # Main application entry point
├── quickstart.py                    # Quick setup script
├── run_tests.py                     # Test runner
├── setup.ps1                        # Windows PowerShell setup
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules
├── readme.md                        # Documentation
├── CONTRIBUTING.md                  # Contribution guidelines
├── CHANGELOG.md                     # Version history
│
├── config/
│   └── config.yaml                 # Application configuration
│
├── src/
│   ├── __init__.py
│   ├── chunker.py                  # Token-based code chunking
│   ├── chromadb_manager.py         # Vector storage operations
│   ├── dspy_analyzer.py            # DSPy prompt optimization
│   ├── langgraph_workflows.py      # Workflow orchestration
│   └── config_loader.py            # Configuration management
│
├── examples/
│   └── example_pyspark_etl.py      # Sample PySpark file
│
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py             # Chunker tests
│   ├── test_chromadb_manager.py    # ChromaDB tests
│   └── generate_test_data.py       # Test data generator
│
├── data/
│   ├── chromadb/                   # Vector database storage
│   ├── input/                      # Input files
│   └── output/                     # Processing results
│
└── logs/                           # Application logs
```

## 🔧 Key Components

### 1. CodeChunker (`src/chunker.py`)
- Uses tiktoken for accurate token counting
- Configurable chunk size (default: 25k tokens)
- Overlapping chunks (default: 5k tokens)
- Preserves code context across chunks

### 2. ChromaDBManager (`src/chromadb_manager.py`)
- Persistent vector storage
- Semantic similarity search
- Automatic embedding generation
- Collection management operations

### 3. DSPyOptimizer (`src/dspy_analyzer.py`)
- CodeLlama integration via Ollama
- Multiple signature types (Analysis, Logic Extraction, Explanation)
- Prompt optimization with training examples
- Bootstrap few-shot learning

### 4. LangGraph Workflows (`src/langgraph_workflows.py`)
- **Stage1Workflow**: File → Chunk → Embed → Store
- **Stage2Workflow**: Query → Retrieve → Analyze → Return
- State management with TypedDict
- Error handling and logging

### 5. Main Application (`main.py`)
- Command-line interface
- Multiple operation modes
- Interactive query mode
- Configuration integration

## 🚀 Usage Modes

### 1. Process Mode (Stage 1)
```bash
python main.py --mode process --file examples/example_pyspark_etl.py
```
Chunks and embeds a PySpark file into ChromaDB.

### 2. Query Mode (Stage 2)
```bash
python main.py --mode query --query "What business rules are in this code?"
```
Analyzes previously processed code.

### 3. Interactive Mode
```bash
python main.py --mode interactive --file examples/example_pyspark_etl.py
```
Process a file and enter interactive query loop.

### 4. Info Mode
```bash
python main.py --mode info
```
Display ChromaDB collection information.

### 5. Clear Mode
```bash
python main.py --mode clear
```
Clear all documents from ChromaDB.

## 📦 Dependencies

### Core Libraries
- **chromadb**: Vector database for embeddings
- **dspy-ai**: Prompt optimization framework
- **langgraph**: Workflow orchestration
- **tiktoken**: Token counting and chunking
- **ollama**: Local LLM interface

### Supporting Libraries
- **pyspark**: Example code domain
- **python-dotenv**: Environment management
- **pyyaml**: Configuration parsing
- **sentence-transformers**: Embeddings

## 🎓 Example Workflow

1. **Setup**
   ```bash
   # Windows PowerShell
   .\setup.ps1
   
   # Or manual setup
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Start Ollama**
   ```bash
   ollama serve
   ollama pull codellama:7b
   ```

3. **Process a File**
   ```bash
   python main.py --mode process --file examples/example_pyspark_etl.py
   ```

4. **Query the Code**
   ```bash
   python main.py --mode query --query "Explain the customer segmentation logic"
   ```

## 💡 Technical Highlights

### Token-Based Chunking
- Uses tiktoken's `cl100k_base` encoding
- Maintains code context with overlaps
- Metadata tracking for each chunk

### Vector Similarity Search
- ChromaDB handles embedding automatically
- Semantic search for relevant code sections
- Distance-based ranking

### DSPy Prompt Optimization
- Signatures define input/output structure
- Chain-of-Thought for better reasoning
- Incremental improvement with examples

### LangGraph State Management
- TypedDict for type-safe states
- Node-based workflow design
- Built-in error handling

## 🧪 Testing

Run all tests:
```bash
python run_tests.py
```

Individual test files:
```bash
python -m unittest tests.test_chunker
python -m unittest tests.test_chromadb_manager
```

## 🔐 Configuration

### Environment Variables (.env)
- `OLLAMA_BASE_URL`: Ollama server URL
- `OLLAMA_MODEL`: LLM model name
- `CHUNK_SIZE`: Tokens per chunk
- `CHUNK_OVERLAP`: Overlapping tokens

### YAML Config (config/config.yaml)
- Application settings
- Chunking parameters
- ChromaDB configuration
- LLM settings
- DSPy optimization parameters

## 📈 Future Enhancements

- [ ] Batch file processing
- [ ] Web UI interface
- [ ] Additional LLM providers
- [ ] Export functionality (PDF, JSON)
- [ ] Query caching
- [ ] Docker containerization
- [ ] Multi-language support

## 🎉 Quick Start

Run the automated quick start:
```bash
python quickstart.py
```

This will:
1. Check Ollama installation
2. Setup environment files
3. Create necessary directories
4. Run an example analysis

## 📚 Documentation

- **README.md**: Comprehensive user guide
- **CONTRIBUTING.md**: Development guidelines
- **CHANGELOG.md**: Version history
- **Code docstrings**: API documentation

## ✨ Success Criteria

All requirements met:
✅ Stage 1 chunking and embedding implemented
✅ Stage 2 query and analysis implemented  
✅ DSPy prompt optimization integrated
✅ LangGraph workflow orchestration
✅ 25k token chunks with 5k overlap
✅ ChromaDB vector storage
✅ CodeLlama:7b integration
✅ Interactive query mode
✅ Comprehensive documentation
✅ Example PySpark file included
✅ Unit tests provided
✅ Easy setup scripts

## 🎯 Project Status

**Status**: ✅ Complete and Ready to Use

The PySpark Code Analyzer is fully functional with all mandatory requirements implemented. The system successfully analyzes Python/PySpark code using a two-stage pipeline orchestrated by LangGraph, with DSPy handling prompt optimization for CodeLlama-7B.
