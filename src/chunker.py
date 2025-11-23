"""
Chunking module for splitting PySpark code files into manageable chunks.
Uses tiktoken for token-based chunking with configurable overlap.
"""
import tiktoken
from typing import List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CodeChunker:
    """Handles chunking of code files with token-based splitting."""
    
    def __init__(self, chunk_size: int = 25000, overlap_size: int = 5000, encoding: str = "cl100k_base"):
        """
        Initialize the CodeChunker.
        
        Args:
            chunk_size: Maximum tokens per chunk
            overlap_size: Number of overlapping tokens between chunks
            encoding: Tiktoken encoding to use
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.encoding = tiktoken.get_encoding(encoding)
        
    def read_file(self, file_path: str) -> str:
        """
        Read content from a PySpark file.
        
        Args:
            file_path: Path to the PySpark file
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"Successfully read file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text into overlapping segments based on token count.
        
        Args:
            text: The text to chunk
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Encode the entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)
        
        logger.info(f"Total tokens in text: {total_tokens}")
        
        chunks = []
        start_idx = 0
        chunk_num = 0
        
        while start_idx < total_tokens:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk_metadata = {
                "chunk_id": chunk_num,
                "start_token": start_idx,
                "end_token": end_idx,
                "total_tokens": len(chunk_tokens),
                "total_file_tokens": total_tokens
            }
            
            # Add any additional metadata
            if metadata:
                chunk_metadata.update(metadata)
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
            logger.debug(f"Created chunk {chunk_num}: tokens {start_idx} to {end_idx}")
            
            # Move to next chunk with overlap
            start_idx += (self.chunk_size - self.overlap_size)
            chunk_num += 1
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def chunk_file(self, file_path: str) -> List[Dict]:
        """
        Read and chunk a PySpark file.
        
        Args:
            file_path: Path to the PySpark file
            
        Returns:
            List of chunk dictionaries
        """
        # Read file content
        content = self.read_file(file_path)
        
        # Create metadata
        path_obj = Path(file_path)
        metadata = {
            "file_path": str(path_obj.absolute()),
            "file_name": path_obj.name,
            "file_extension": path_obj.suffix
        }
        
        # Chunk the content
        chunks = self.chunk_text(content, metadata)
        
        logger.info(f"Chunked file {file_path} into {len(chunks)} chunks")
        return chunks


def main():
    """Example usage of the CodeChunker."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize chunker
    chunker = CodeChunker(chunk_size=25000, overlap_size=5000)
    
    # Example: chunk a file
    # chunks = chunker.chunk_file("path/to/your/pyspark_file.py")
    
    # Print chunk information
    # for chunk in chunks:
    #     print(f"Chunk {chunk['metadata']['chunk_id']}: {chunk['metadata']['total_tokens']} tokens")


if __name__ == "__main__":
    main()
