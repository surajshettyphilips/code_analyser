"""
FastAPI backend for PySpark Code Analyzer
Provides endpoints for file upload (Stage 1) and question answering (Stage 2)
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.chunker import CodeChunker
from src.chromadb_manager import ChromaDBManager
from src.dspy_analyzer import DSPyOptimizer
from src.config_loader import Config

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
dspy_optimizer = DSPyOptimizer(config)

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
            query_text=request.question,
            n_results=request.n_results
        )
        
        if not relevant_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant code found for this question"
            )
        
        # Use DSPy to analyze and answer
        print(f"Analyzing with DSPy")
        analysis = dspy_optimizer.analyze_code(
            code_snippets=relevant_chunks,
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
        count = chromadb_manager.get_collection_count()
        return {
            "collection_name": config.chromadb['collection_name'],
            "total_chunks": count,
            "persist_directory": config.chromadb['persist_directory']
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
            "collection_name": config.chromadb['collection_name']
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting collection: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
