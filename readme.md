# PySpark Code Analyzer

An intelligent AI-powered tool for analyzing Python/PySpark code to extract detailed insights including embedded business logic. The system uses advanced techniques like semantic chunking, vector embeddings, and LLM-based analysis to provide human-readable explanations of complex code.

## 🎯 Features

- **Intelligent Code Chunking**: Splits large PySpark files into 25k token chunks with 5k overlap using tiktoken
- **Vector Storage**: Persists code chunks in ChromaDB with semantic embeddings for efficient retrieval
- **AI-Powered Analysis**: Uses CodeLlama-7B for deep code understanding and business logic extraction
- **Prompt Optimization**: Leverages DSPy for incremental prompt evolution and improvement
- **Workflow Orchestration**: LangGraph manages two separate stages (chunking/embedding and query/analysis)
- **Interactive Mode**: Query your code multiple times after initial processing

## 🏗️ Architecture

### Stage 1: Chunking and Embedding
1. Reads PySpark file content
2. Chunks code with 25k tokens per chunk (5k overlap)
3. Vectorizes and embeds chunks in ChromaDB

### Stage 2: Query and Analysis
1. Receives user query about the code
2. Retrieves relevant chunks from ChromaDB
3. Passes chunks to CodeLlama-7B via DSPy
4. Returns human-readable analysis including business logic

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- Ollama installed and running locally
- At least 8GB RAM recommended

### Install Ollama and CodeLlama

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Pull the CodeLlama-7B model:
```bash
ollama pull codellama:7b
```

3. Verify Ollama is running:
```bash
ollama list
```

## 🚀 Installation

### 1. Clone the repository
```bash
cd codellama
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure environment
```bash
cp .env.example .env
```

Edit `.env` file with your settings if needed (default values work for local Ollama setup).

## 📖 Usage

### Process a PySpark File (Stage 1)
Chunk and embed a PySpark file into ChromaDB:

```bash
python main.py --mode process --file examples/example_pyspark_etl.py
```

### Query the Code (Stage 2)
Ask questions about the processed code:

```bash
python main.py --mode query --query "What business rules are implemented in this code?"
```

### Interactive Mode
Process a file and enter interactive query mode:

```bash
python main.py --mode interactive --file examples/example_pyspark_etl.py
```

Then ask multiple queries:
```
Query: What are the customer segmentation rules?
Query: How are discounts calculated?
Query: What validations are applied to customer data?
```

### View Collection Info
Check ChromaDB collection status:

```bash
python main.py --mode info
```

### Clear Collection
Remove all documents from ChromaDB:

```bash
python main.py --mode clear
```

## 🔧 Configuration

Configuration is managed through `config/config.yaml` and environment variables in `.env`.

### Key Settings

**Chunking:**
- `chunk_size`: 25000 tokens per chunk
- `overlap_size`: 5000 overlapping tokens

**ChromaDB:**
- `collection_name`: Name of the vector collection
- `persist_directory`: Where to store embeddings

**LLM:**
- `model`: codellama:7b
- `base_url`: http://localhost:11434

**DSPy:**
- `cache_dir`: Cache directory for DSPy
- `max_bootstrapped_demos`: 4
- `max_labeled_demos`: 16

## 📂 Project Structure

```
codellama/
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── readme.md                   # This file
├── config/
│   └── config.yaml            # Application configuration
├── src/
│   ├── __init__.py
│   ├── chunker.py             # Code chunking module
│   ├── chromadb_manager.py    # ChromaDB integration
│   ├── dspy_analyzer.py       # DSPy prompt optimization
│   ├── langgraph_workflows.py # LangGraph orchestration
│   └── config_loader.py       # Configuration management
├── examples/
│   └── example_pyspark_etl.py # Sample PySpark file
├── data/
│   └── chromadb/              # ChromaDB persistence
├── logs/                       # Application logs
└── tests/                      # Unit tests
```

## 💡 Example Queries

Once you've processed a PySpark file, try these queries:

- "What business rules are implemented in this code?"
- "Explain the customer segmentation logic"
- "How are discounts calculated for different customer types?"
- "What data validations are performed?"
- "List all the transformations applied to the customer data"
- "What are the key business metrics being calculated?"
- "Explain the ETL pipeline flow"

## 🧪 Testing

Run the example to verify everything is working:

```bash
# Process the example file
python main.py --mode process --file examples/example_pyspark_etl.py

# Query it
python main.py --mode query --query "Explain the main business rules"
```

## 🔍 How It Works

### Stage 1: Chunking and Embedding (LangGraph)
```
Read File → Chunk Content → Store in ChromaDB → Finalize
```

**Components:**
- **CodeChunker**: Uses tiktoken for token-aware chunking
- **ChromaDBManager**: Handles vector storage and retrieval
- **Stage1Workflow**: Orchestrates the chunking pipeline

### Stage 2: Query and Analysis (LangGraph)
```
Retrieve Chunks → Analyze with DSPy → Return Result → Finalize
```

**Components:**
- **ChromaDBManager**: Retrieves semantically similar chunks
- **DSPyOptimizer**: Optimizes prompts and analyzes code
- **Stage2Workflow**: Orchestrates the analysis pipeline

### DSPy Prompt Evolution
DSPy automatically improves prompts through:
- Chain-of-Thought reasoning
- Few-shot learning
- Bootstrap optimization
- Incremental refinement

## 🛠️ Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve
```

### ChromaDB Errors
```bash
# Clear the collection and try again
python main.py --mode clear
```

### Memory Issues
Reduce chunk size in `config/config.yaml`:
```yaml
chunking:
  chunk_size: 15000  # Reduced from 25000
  overlap_size: 3000  # Reduced from 5000
```

## 📝 Notes

- **First run**: Initial processing may take time as embeddings are generated
- **ChromaDB**: Data persists in `data/chromadb/` directory
- **Logs**: Check `logs/app.log` for detailed execution logs
- **DSPy Cache**: Cached in `.dspy_cache/` for faster subsequent runs

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add support for more LLM providers
- Implement batch file processing
- Add web UI for easier interaction
- Enhance DSPy training with more examples
- Add unit tests

## 📄 License

This project is provided as-is for educational and research purposes.

## 🔗 Technologies Used

- **[ChromaDB](https://www.trychroma.com/)**: Vector database for embeddings
- **[DSPy](https://github.com/stanfordnlp/dspy)**: Prompt optimization framework
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Workflow orchestration
- **[Ollama](https://ollama.ai)**: Local LLM runtime
- **[CodeLlama](https://ai.meta.com/blog/code-llama-large-language-model-coding/)**: Meta's code-specialized LLM
- **[Tiktoken](https://github.com/openai/tiktoken)**: Token counting and chunking