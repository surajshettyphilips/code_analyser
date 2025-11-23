"""
Unit tests for ChromaDBManager module.
"""
import unittest
import tempfile
import shutil
from src.chromadb_manager import ChromaDBManager


class TestChromaDBManager(unittest.TestCase):
    """Test cases for ChromaDBManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ChromaDBManager(
            collection_name="test_collection",
            persist_directory=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_chunks(self):
        """Test adding chunks to ChromaDB."""
        chunks = [
            {
                "text": "def test_function(): pass",
                "metadata": {"chunk_id": 0, "file_name": "test.py"}
            }
        ]
        
        result = self.manager.add_chunks(chunks)
        self.assertTrue(result)
    
    def test_query_chunks(self):
        """Test querying chunks."""
        chunks = [
            {
                "text": "def calculate_discount(price): return price * 0.1",
                "metadata": {"chunk_id": 0, "file_name": "test.py"}
            }
        ]
        
        self.manager.add_chunks(chunks)
        results = self.manager.query_relevant_chunks("discount calculation", n_results=1)
        
        self.assertGreater(len(results), 0)
    
    def test_collection_info(self):
        """Test getting collection info."""
        info = self.manager.get_collection_info()
        self.assertIn('name', info)
        self.assertIn('count', info)


if __name__ == '__main__':
    unittest.main()
