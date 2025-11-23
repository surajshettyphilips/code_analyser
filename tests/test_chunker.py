"""
Unit tests for the CodeChunker module.
"""
import unittest
import tempfile
import os
from src.chunker import CodeChunker


class TestCodeChunker(unittest.TestCase):
    """Test cases for CodeChunker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = CodeChunker(chunk_size=100, overlap_size=20)
        
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test. " * 50  # Create some text
        chunks = self.chunker.chunk_text(text)
        
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all('text' in chunk for chunk in chunks))
        self.assertTrue(all('metadata' in chunk for chunk in chunks))
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = "word " * 200
        chunks = self.chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check overlap exists
            self.assertIsNotNone(chunks[0]['metadata']['end_token'])
            self.assertIsNotNone(chunks[1]['metadata']['start_token'])
    
    def test_chunk_file(self):
        """Test chunking a file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write("# Test Python file\n" * 100)
            temp_path = f.name
        
        try:
            chunks = self.chunker.chunk_file(temp_path)
            self.assertGreater(len(chunks), 0)
            self.assertTrue(all('file_name' in chunk['metadata'] for chunk in chunks))
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
