"""
Main application entry point for PySpark Code Analyzer.
Orchestrates both Stage 1 (chunking and embedding) and Stage 2 (query and analysis).
"""
import argparse
import logging
import sys
from pathlib import Path

from src.config_loader import get_config
from src.chunker import CodeChunker
from src.chromadb_manager import ChromaDBManager
from src.dspy_analyzer import DSPyOptimizer
from src.langgraph_workflows import Stage1Workflow, Stage2Workflow


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log')
    ]
)

logger = logging.getLogger(__name__)


class PySparkCodeAnalyzer:
    """Main application class for analyzing PySpark code."""
    
    def __init__(self):
        """Initialize the analyzer with all components."""
        logger.info("Initializing PySpark Code Analyzer")
        
        # Load configuration
        self.config = get_config()
        
        # Initialize components
        self._initialize_components()
        
        # Setup workflows
        self._setup_workflows()
        
        logger.info("PySpark Code Analyzer initialized successfully")
    
    def _initialize_components(self):
        """Initialize all components."""
        # Chunking configuration
        chunking_config = self.config.get_chunking_config()
        self.chunker = CodeChunker(
            chunk_size=chunking_config.get('chunk_size', 25000),
            overlap_size=chunking_config.get('overlap_size', 5000),
            encoding=chunking_config.get('encoding', 'cl100k_base')
        )
        logger.info("CodeChunker initialized")
        
        # ChromaDB configuration
        chromadb_config = self.config.get_chromadb_config()
        self.chromadb_manager = ChromaDBManager(
            collection_name=chromadb_config.get('collection_name', 'pyspark_code_chunks'),
            persist_directory=chromadb_config.get('persist_directory', 'c:\\Users\\320196443\\codellama\\chromadb'),
            embedding_function=chromadb_config.get('embedding_function', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        logger.info("ChromaDBManager initialized")
        
        # DSPy configuration
        llm_config = self.config.get_llm_config()
        self.dspy_optimizer = DSPyOptimizer(
            model_name=llm_config.get('model', 'codellama:7b'),
            cache_dir=self.config.get('dspy.cache_dir', './.dspy_cache')
        )
        self.dspy_optimizer.setup_lm(
            base_url=llm_config.get('base_url', 'http://localhost:11434')
        )
        logger.info("DSPy Optimizer initialized")
    
    def _setup_workflows(self):
        """Setup LangGraph workflows."""
        self.stage1_workflow = Stage1Workflow(
            chunker=self.chunker,
            chromadb_manager=self.chromadb_manager
        )
        logger.info("Stage 1 Workflow initialized")
        
        self.stage2_workflow = Stage2Workflow(
            chromadb_manager=self.chromadb_manager,
            dspy_optimizer=self.dspy_optimizer
        )
        logger.info("Stage 2 Workflow initialized")
    
    def process_file(self, file_path: str) -> bool:
        """
        Process a PySpark file (Stage 1).
        
        Args:
            file_path: Path to the PySpark file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting Stage 1: Processing file {file_path}")
        
        # Check if file exists
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Execute Stage 1 workflow
        result = self.stage1_workflow.execute(file_path)
        
        # Check for errors
        if result.get('error'):
            logger.error(f"Stage 1 failed: {result['error']}")
            return False
        
        logger.info(f"Stage 1 completed successfully. Processed {result['chunk_count']} chunks")
        return True
    
    def query_code(self, query: str) -> str:
        """
        Query the processed code (Stage 2).
        
        Args:
            query: User's question about the code
            
        Returns:
            Analysis result
        """
        logger.info(f"Starting Stage 2: Processing query - {query}")
        
        # Execute Stage 2 workflow
        result = self.stage2_workflow.execute(query)
        
        # Check for errors
        if result.get('error'):
            logger.error(f"Stage 2 failed: {result['error']}")
            return f"Error: {result['error']}"
        
        logger.info("Stage 2 completed successfully")
        return result.get('analysis_result', 'No analysis result generated')
    
    def get_collection_info(self):
        """Get information about the ChromaDB collection."""
        return self.chromadb_manager.get_collection_info()
    
    def clear_collection(self):
        """Clear all documents from the ChromaDB collection."""
        return self.chromadb_manager.clear_collection()


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='PySpark Code Analyzer - Analyze Python/PySpark code with AI'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['process', 'query', 'interactive', 'info', 'clear'],
        required=True,
        help='Operation mode: process (Stage 1), query (Stage 2), interactive (both), info (collection info), clear (clear collection)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Path to the PySpark file to process (for process mode)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Query about the code (for query mode)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)
        
        # Initialize analyzer
        analyzer = PySparkCodeAnalyzer()
        
        if args.mode == 'process':
            if not args.file:
                print("Error: --file argument is required for process mode")
                sys.exit(1)
            
            success = analyzer.process_file(args.file)
            if success:
                print(f"\n✓ Successfully processed file: {args.file}")
                print(f"  Collection info: {analyzer.get_collection_info()}")
            else:
                print(f"\n✗ Failed to process file: {args.file}")
                sys.exit(1)
        
        elif args.mode == 'query':
            if not args.query:
                print("Error: --query argument is required for query mode")
                sys.exit(1)
            
            result = analyzer.query_code(args.query)
            print("\n" + "="*80)
            print("ANALYSIS RESULT:")
            print("="*80)
            print(result)
            print("="*80 + "\n")
        
        elif args.mode == 'interactive':
            print("\n" + "="*80)
            print("PySpark Code Analyzer - Interactive Mode")
            print("="*80 + "\n")
            
            # Process file
            if args.file:
                file_path = args.file
            else:
                file_path = input("Enter path to PySpark file: ").strip()
            
            success = analyzer.process_file(file_path)
            if not success:
                print("Failed to process file. Exiting.")
                sys.exit(1)
            
            print(f"\n✓ File processed successfully!")
            print(f"  {analyzer.get_collection_info()}\n")
            
            # Interactive query loop
            print("Enter your queries (type 'exit' to quit):\n")
            while True:
                query = input("Query: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not query:
                    continue
                
                result = analyzer.query_code(query)
                print("\n" + "-"*80)
                print(result)
                print("-"*80 + "\n")
        
        elif args.mode == 'info':
            info = analyzer.get_collection_info()
            print("\n" + "="*80)
            print("ChromaDB Collection Info:")
            print("="*80)
            print(f"  Name: {info.get('name', 'N/A')}")
            print(f"  Document Count: {info.get('count', 0)}")
            print(f"  Persist Directory: {info.get('persist_directory', 'N/A')}")
            print("="*80 + "\n")
        
        elif args.mode == 'clear':
            confirm = input("Are you sure you want to clear the collection? (yes/no): ").strip().lower()
            if confirm == 'yes':
                analyzer.clear_collection()
                print("\n✓ Collection cleared successfully!")
            else:
                print("\nOperation cancelled.")
    
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
