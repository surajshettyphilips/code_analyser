"""
LangGraph workflows for orchestrating the code analysis pipeline.
Stage 1: Chunking and embedding
Stage 2: Query and analysis
"""
from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
import logging
import operator

logger = logging.getLogger(__name__)


# State definitions for Stage 1 (Chunking and Embedding)
class Stage1State(TypedDict):
    """State for Stage 1: File processing and chunking."""
    file_path: str
    file_content: str
    chunks: List[Dict]
    embedded_chunks: List[Dict]
    chromadb_status: str
    error: str
    chunk_count: int
    embedding_status: str


# State definitions for Stage 2 (Query and Analysis)
class Stage2State(TypedDict):
    """State for Stage 2: Query processing and analysis."""
    query: str
    relevant_chunks: List[Dict]
    analysis_result: str
    error: str
    chunk_count: int


class Stage1Workflow:
    """LangGraph workflow for Stage 1: Chunking and Embedding."""
    
    def __init__(self, chunker, chromadb_manager, ollama_embedder=None):
        """
        Initialize Stage 1 workflow.
        
        Args:
            chunker: CodeChunker instance
            chromadb_manager: ChromaDBManager instance
            ollama_embedder: Optional Ollama embedder for custom embeddings
        """
        self.chunker = chunker
        self.chromadb_manager = chromadb_manager
        self.ollama_embedder = ollama_embedder
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the Stage 1 workflow graph."""
        workflow = StateGraph(Stage1State)
        
        # Add nodes
        workflow.add_node("read_file", self._read_file)
        workflow.add_node("chunk_content", self._chunk_content)
        workflow.add_node("embed_chunks", self._embed_chunks)
        workflow.add_node("store_in_chromadb", self._store_in_chromadb)
        workflow.add_node("finalize", self._finalize)
        
        # Add edges
        workflow.set_entry_point("read_file")
        workflow.add_edge("read_file", "chunk_content")
        workflow.add_edge("chunk_content", "embed_chunks")
        workflow.add_edge("embed_chunks", "store_in_chromadb")
        workflow.add_edge("store_in_chromadb", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _read_file(self, state: Stage1State) -> Stage1State:
        """Read file content."""
        try:
            logger.info(f"Reading file: {state['file_path']}")
            content = self.chunker.read_file(state['file_path'])
            state['file_content'] = content
            state['error'] = ""
            logger.info("File read successfully")
        except Exception as e:
            state['error'] = f"Error reading file: {str(e)}"
            logger.error(state['error'])
        return state
    
    def _chunk_content(self, state: Stage1State) -> Stage1State:
        """Chunk the file content."""
        try:
            if state.get('error'):
                return state
            
            logger.info("Chunking file content")
            chunks = self.chunker.chunk_file(state['file_path'])
            state['chunks'] = chunks
            state['chunk_count'] = len(chunks)
            logger.info(f"Created {len(chunks)} chunks")
        except Exception as e:
            state['error'] = f"Error chunking content: {str(e)}"
            logger.error(state['error'])
        return state
    
    def _embed_chunks(self, state: Stage1State) -> Stage1State:
        """Generate Ollama-based embeddings for chunks to improve retrieval relevance."""
        try:
            if state.get('error'):
                return state
            
            logger.info("Generating Ollama embeddings for chunks")
            
            if self.ollama_embedder:
                # Use custom Ollama embedder if provided
                embedded_chunks = []
                for chunk in state['chunks']:
                    try:
                        # Generate embedding using Ollama
                        embedding = self.ollama_embedder.embed(chunk.get('text', ''))
                        chunk['embedding'] = embedding
                        embedded_chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Failed to embed chunk {chunk.get('metadata', {}).get('chunk_id', 'unknown')}: {e}")
                        # Keep chunk without custom embedding - ChromaDB will use default
                        embedded_chunks.append(chunk)
                
                state['embedded_chunks'] = embedded_chunks
                state['embedding_status'] = "success"
                logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks using Ollama")
            else:
                # No custom embedder - use ChromaDB's default embedding
                state['embedded_chunks'] = state['chunks']
                state['embedding_status'] = "default"
                logger.info("Using ChromaDB default embeddings (no custom Ollama embedder provided)")
                
        except Exception as e:
            state['embedding_status'] = "failed"
            state['error'] = f"Error generating embeddings: {str(e)}"
            logger.error(state['error'])
            # Fall back to original chunks
            state['embedded_chunks'] = state['chunks']
        
        return state
    
    def _store_in_chromadb(self, state: Stage1State) -> Stage1State:
        """Store chunks in ChromaDB."""
        try:
            if state.get('error'):
                return state
            
            logger.info("Storing chunks in ChromaDB")
            # Use embedded chunks if available, otherwise fall back to original chunks
            chunks_to_store = state.get('embedded_chunks', state['chunks'])
            success = self.chromadb_manager.add_chunks(chunks_to_store)
            
            if success:
                state['chromadb_status'] = "success"
                logger.info(f"Chunks stored successfully with {state.get('embedding_status', 'default')} embeddings")
            else:
                state['chromadb_status'] = "failed"
                state['error'] = "Failed to store chunks in ChromaDB"
                logger.error(state['error'])
        except Exception as e:
            state['chromadb_status'] = "failed"
            state['error'] = f"Error storing in ChromaDB: {str(e)}"
            logger.error(state['error'])
        return state
    
    def _finalize(self, state: Stage1State) -> Stage1State:
        """Finalize Stage 1."""
        if not state.get('error'):
            logger.info(f"Stage 1 completed successfully. Processed {state['chunk_count']} chunks")
        else:
            logger.error(f"Stage 1 completed with errors: {state['error']}")
        return state
    
    def execute(self, file_path: str) -> Stage1State:
        """
        Execute Stage 1 workflow.
        
        Args:
            file_path: Path to the PySpark file
            
        Returns:
            Final state
        """
        initial_state = Stage1State(
            file_path=file_path,
            file_content="",
            chunks=[],
            embedded_chunks=[],
            chromadb_status="",
            error="",
            chunk_count=0,
            embedding_status=""
        )
        
        final_state = self.graph.invoke(initial_state)
        return final_state


class Stage2Workflow:
    """LangGraph workflow for Stage 2: Query and Analysis."""
    
    def __init__(self, chromadb_manager, dspy_optimizer):
        """
        Initialize Stage 2 workflow.
        
        Args:
            chromadb_manager: ChromaDBManager instance
            dspy_optimizer: DSPyOptimizer instance
        """
        self.chromadb_manager = chromadb_manager
        self.dspy_optimizer = dspy_optimizer
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the Stage 2 workflow graph."""
        workflow = StateGraph(Stage2State)
        
        # Add nodes
        workflow.add_node("retrieve_chunks", self._retrieve_chunks)
        workflow.add_node("analyze_code", self._analyze_code)
        workflow.add_node("finalize", self._finalize)
        
        # Add edges
        workflow.set_entry_point("retrieve_chunks")
        workflow.add_edge("retrieve_chunks", "analyze_code")
        workflow.add_edge("analyze_code", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _retrieve_chunks(self, state: Stage2State) -> Stage2State:
        """Retrieve relevant chunks from ChromaDB."""
        try:
            logger.info(f"Retrieving relevant chunks for query: {state['query']}")
            chunks = self.chromadb_manager.query_relevant_chunks(
                query=state['query'],
                n_results=1
            )
            state['relevant_chunks'] = chunks
            state['chunk_count'] = len(chunks)
            state['error'] = ""
            logger.info(f"Retrieved {len(chunks)} relevant chunks")
        except Exception as e:
            state['error'] = f"Error retrieving chunks: {str(e)}"
            logger.error(state['error'])
        return state
    
    def _analyze_code(self, state: Stage2State) -> Stage2State:
        """Analyze code using DSPy."""
        try:
            if state.get('error'):
                return state
            
            logger.info("Analyzing code with DSPy")
            analysis = self.dspy_optimizer.analyze_code(
                relevant_chunks=state['relevant_chunks'],
                query=state['query']
            )
            state['analysis_result'] = analysis
            logger.info("Analysis completed")
        except Exception as e:
            state['error'] = f"Error during analysis: {str(e)}"
            logger.error(state['error'])
        return state
    
    def _finalize(self, state: Stage2State) -> Stage2State:
        """Finalize Stage 2."""
        if not state.get('error'):
            logger.info("Stage 2 completed successfully")
        else:
            logger.error(f"Stage 2 completed with errors: {state['error']}")
        return state
    
    def execute(self, query: str) -> Stage2State:
        """
        Execute Stage 2 workflow.
        
        Args:
            query: User's query about the code
            
        Returns:
            Final state
        """
        initial_state = Stage2State(
            query=query,
            relevant_chunks=[],
            analysis_result="",
            error="",
            chunk_count=0
        )
        
        final_state = self.graph.invoke(initial_state)
        return final_state


def main():
    """Example usage of workflows."""
    logging.basicConfig(level=logging.INFO)
    print("LangGraph workflows defined successfully")


if __name__ == "__main__":
    main()
