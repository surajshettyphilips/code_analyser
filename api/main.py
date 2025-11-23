"""
FastAPI backend for PySpark Code Analyzer
Provides endpoints for file upload (Stage 1) and question answering (Stage 2)
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
import logging

# Disable PostHog telemetry to avoid SSL errors
os.environ['POSTHOG_DISABLED'] = '1'
os.environ['DISABLE_TELEMETRY'] = '1'

# Suppress PostHog and SSL error logging
logging.getLogger('posthog').setLevel(logging.CRITICAL)
logging.getLogger('backoff').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

# Import existing components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.chunker import CodeChunker
from src.chromadb_manager import ChromaDBManager
from src.dspy_analyzer import DSPyOptimizer
from src.config_loader import Config

# Import RAGAs evaluator
sys.path.insert(0, str(Path(__file__).parent.parent / "ragas"))
from evaluate_questions import RAGasEvaluator

# Initialize FastAPI app
app = FastAPI(
    title="PySpark Code Analyzer API",
    description="Upload code files for analysis and query with natural language",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = Config()
chunker = CodeChunker(
    chunk_size=config.get('chunking.chunk_size', 25000),
    overlap_size=config.get('chunking.overlap_size', 5000)
)
chromadb_manager = ChromaDBManager(
    persist_directory=config.get('chromadb.persist_directory', './chromadb'),
    collection_name=config.get('chromadb.collection_name', 'pyspark_code_chunks')
)
dspy_optimizer = DSPyOptimizer(
    model_name=config.get('llm.model', 'codellama:7b'),
    cache_dir=config.get('dspy.cache_dir', './.dspy_cache')
)
# Setup the language model for DSPy
dspy_optimizer.setup_lm(
    base_url=config.get('llm.base_url', 'http://localhost:11434'),
    temperature=config.get('llm.temperature', 0.1),
    max_tokens=config.get('llm.max_tokens', 2048)
)

# Response models
class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int
    total_chunks: int

class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[dict]
    num_contexts: int


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "PySpark Code Analyzer API",
        "version": "1.0.0"
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Stage 1: Upload a code file and index it in ChromaDB
    
    Args:
        file: Python code file to analyze
        
    Returns:
        Upload status with chunk count
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.py', '.txt')):
            raise HTTPException(
                status_code=400,
                detail="Only Python (.py) and text (.txt) files are supported"
            )
        
        # Read file content
        content = await file.read()
        code_content = content.decode('utf-8')
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(code_content)
            tmp_path = tmp_file.name
        
        try:
            # Stage 1: Chunk the code
            print(f"Chunking file: {file.filename}")
            chunks = chunker.chunk_file(tmp_path)
            
            if not chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No valid chunks could be extracted from the file"
                )
            
            # Index chunks in ChromaDB
            print(f"Indexing {len(chunks)} chunks to ChromaDB")
            chromadb_manager.add_chunks(chunks, source_file=file.filename)
            
            return UploadResponse(
                message="File uploaded and indexed successfully",
                filename=file.filename,
                chunks_indexed=len(chunks),
                total_chunks=len(chunks)
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded"
        )
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_code(request: QueryRequest):
    """
    Stage 2: Query the indexed code with a natural language question
    
    Args:
        request: Query request with question and optional n_results
        
    Returns:
        Answer with relevant code contexts
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Query ChromaDB for relevant chunks
        print(f"Querying ChromaDB: {request.question}")
        relevant_chunks = chromadb_manager.query_relevant_chunks(
            query=request.question,
            n_results=request.n_results
        )
        
        if not relevant_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant code found for this question"
            )
        
        # Use DSPy to analyze and answer
        print("Analyzing with DSPy")
        analysis = dspy_optimizer.analyze_code(
            relevant_chunks=relevant_chunks,
            query=request.question
        )
        
        # Format contexts for response
        contexts = [
            {
                "text": chunk.get("content", chunk.get("text", "")),
                "metadata": chunk.get("metadata", {}),
                "distance": chunk.get("distance", None)
            }
            for chunk in relevant_chunks
        ]
        
        return QueryResponse(
            question=request.question,
            answer=analysis,
            contexts=contexts,
            num_contexts=len(contexts)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/collection/info")
async def collection_info():
    """Get information about the ChromaDB collection"""
    try:
        info = chromadb_manager.get_collection_info()
        return {
            "collection_name": info.get("name", config.get('chromadb.collection_name')),
            "total_chunks": info.get("count", 0),
            "persist_directory": info.get("persist_directory", config.get('chromadb.persist_directory'))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collection info: {str(e)}"
        )


@app.delete("/collection/reset")
async def reset_collection():
    """Clear all data from the ChromaDB collection"""
    try:
        chromadb_manager.reset_collection()
        return {
            "message": "Collection reset successfully",
            "collection_name": config.get('chromadb.collection_name')
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting collection: {str(e)}"
        )


class EvalResponse(BaseModel):
    message: str
    results_file: str
    total_questions: int
    results: list[dict]


@app.post("/eval", response_model=EvalResponse)
async def evaluate_questions(file: UploadFile = File(...)):
    """
    Run RAGAs evaluation on uploaded Excel file with questions and ground truth.
    
    Args:
        file: Excel file with columns: question, ground_truth, answer (optional)
        
    Returns:
        Evaluation results with RAGAs metrics
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Only Excel files (.xlsx, .xls) are supported"
            )
        
        # Read and save uploaded file
        content = await file.read()
        input_file = Path("ragas") / "temp_input.xlsx"
        output_file = Path("ragas") / "temp_output_evaluated.xlsx"
        
        # Save uploaded file
        with open(input_file, 'wb') as f:
            f.write(content)
        
        try:
            # Run the evaluation directly using RAGasEvaluator
            print(f"Running RAGAs evaluation on {file.filename}")
            evaluator = RAGasEvaluator(
                input_excel=str(input_file),
                output_excel=str(output_file)
            )
            evaluator.run()
            
            # Read the evaluated results
            if not output_file.exists():
                raise HTTPException(
                    status_code=500,
                    detail="Evaluation completed but output file not found"
                )
            
            df = pd.read_excel(output_file)
            
            # Convert to list of dicts for response
            results = df.to_dict('records')
            
            # Clean up temp input file
            if input_file.exists():
                os.unlink(input_file)
            
            return EvalResponse(
                message="Evaluation completed successfully",
                results_file=str(output_file),
                total_questions=len(results),
                results=results
            )
            
        except Exception as eval_error:
            print(f"Evaluation error: {str(eval_error)}")
            # Clean up on error
            if input_file.exists():
                os.unlink(input_file)
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation failed: {str(eval_error)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing evaluation: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

