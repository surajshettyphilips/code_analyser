"""
ChromaDB integration module for persisting and retrieving code chunks.
Handles vectorization and embedding of code chunks.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Manages ChromaDB operations for code chunk storage and retrieval."""
    
    def __init__(
        self, 
        collection_name: str = "pyspark_code_chunks",
        persist_directory: str = "c:\\Users\\320196443\\codellama\\chromadb",
        embedding_function: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_directory: str = "c:\\Users\\320196443\\codellama\\output"
    ):
        """
        Initialize ChromaDB manager.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_function: Embedding model to use
            output_directory: Directory to save output JSON files
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.output_directory = output_directory
        
        # Create directories if they don't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "PySpark code chunks for analysis"}
            )
            logger.info(f"Connected to collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB collection: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict], source_file: str = None) -> bool:
        """
        Add code chunks to ChromaDB with embeddings.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            source_file: Optional source file path to add to metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                
                # Add source_file to metadata if provided
                if source_file:
                    metadata["source_file"] = source_file
                
                # Create unique ID for chunk
                file_name = metadata.get("file_name", "unknown")
                chunk_id = metadata.get("chunk_id", 0)
                unique_id = f"{file_name}_chunk_{chunk_id}"
                
                documents.append(text)
                
                # Convert metadata values to strings for ChromaDB
                str_metadata = {k: str(v) for k, v in metadata.items()}
                metadatas.append(str_metadata)
                ids.append(unique_id)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            
            # Save indexed data to JSON file
            self._save_indexed_data(documents, metadatas, ids, source_file)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            return False
    
    def _save_indexed_data(self, documents: List[str], metadatas: List[Dict], 
                          ids: List[str], source_file: str = None):
        """
        Save indexed data to JSON file.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of document IDs
            source_file: Optional source file path
        """
        try:
            output_file = Path(self.output_directory) / "chromadb_indexed_data.json"
            
            # Create new data structure
            indexed_chunks = []
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                chunk_data = {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                    "source_file": source_file,
                    "indexed_at": datetime.now().isoformat()
                }
                indexed_chunks.append(chunk_data)
            
            # Create output data (overwrite mode)
            output_data = {
                "collection_name": self.collection_name,
                "total_chunks": len(indexed_chunks),
                "last_updated": datetime.now().isoformat(),
                "indexed_chunks": indexed_chunks
            }
            
            # Save to file (overwrite)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved indexed data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving indexed data to JSON: {e}")
    
    def query_relevant_chunks(
        self, 
        query: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Query ChromaDB for relevant code chunks.
        
        Args:
            query: User query about the code
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of relevant chunks with their metadata
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata if filter_metadata else None
            )
            
            # Format results
            relevant_chunks = []
            if results['documents'] and len(results['documents']) > 0:
                for i in range(len(results['documents'][0])):
                    chunk = {
                        "content": results['documents'][0][i],
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else None
                    }
                    relevant_chunks.append(chunk)
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query")
            
            # Save query results to JSON file
            self._save_query_results(query, relevant_chunks, n_results, filter_metadata)
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []
    
    def _save_query_results(self, query: str, relevant_chunks: List[Dict], 
                           n_results: int, filter_metadata: Optional[Dict]):
        """
        Save query results to JSON file.
        
        Args:
            query: The query text
            relevant_chunks: List of relevant chunks retrieved
            n_results: Number of results requested
            filter_metadata: Metadata filter used
        """
        try:
            output_file = Path(self.output_directory) / "chromadb_query_relevant_chunks.json"
            
            # Create new query result (overwrite mode)
            query_result = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "n_results_requested": n_results,
                "n_results_retrieved": len(relevant_chunks),
                "filter_metadata": filter_metadata,
                "relevant_chunks": relevant_chunks
            }
            
            # Create output data (overwrite mode)
            output_data = {
                "collection_name": self.collection_name,
                "total_queries": 1,
                "last_query_at": datetime.now().isoformat(),
                "queries": [query_result]
            }
            
            # Save to file (overwrite)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved query results to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving query results to JSON: {e}")
    
    def delete_collection(self) -> bool:
        """
        Delete the current collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all IDs
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Cleared {len(results['ids'])} documents from collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all IDs
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Cleared {len(results['ids'])} documents from collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False


def main():
    """Example usage of ChromaDBManager."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize manager
    manager = ChromaDBManager()
    
    # Get collection info
    info = manager.get_collection_info()
    print(f"Collection info: {info}")


if __name__ == "__main__":
    main()
