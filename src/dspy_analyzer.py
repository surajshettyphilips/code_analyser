"""
DSPy module for code analysis with prompt optimization.
Uses DSPy signatures and modules for evolving prompts incrementally.
"""
import dspy
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CodeAnalysisSignature(dspy.Signature):
    """Signature for analyzing code and extracting business logic.
    
    You are an expert SQL and code analyst with deep knowledge of:
    - PySpark, Python, SQL, and data engineering patterns
    - Reading and interpreting PySpark code and DataFrame operations
    - Business logic extraction and documentation
    - Data transformation pipelines and ETL workflows
    
    Provide detailed, accurate, and professional analysis.
    """
    
    context = dspy.InputField(desc="Relevant code chunks from the PySpark file")
    query = dspy.InputField(desc="User's question about the code")
    analysis = dspy.OutputField(desc="Detailed human-readable analysis of the code including business logic, written by an expert analyst")


class BusinessLogicExtractor(dspy.Signature):
    """Signature for extracting specific business logic from code.
    
    You are an expert business analyst and code reviewer specializing in:
    - Identifying business rules and logic patterns in code
    - Translating technical implementation to business requirements
    - Documenting data validation and transformation rules
    
    Extract and explain business logic clearly and comprehensively.
    """
    
    code = dspy.InputField(desc="Code snippet to analyze")
    business_logic = dspy.OutputField(desc="Business logic and rules embedded in the code, explained by an expert business analyst")


class CodeExplanation(dspy.Signature):
    """Signature for explaining code functionality.
    
    You are an expert software engineer and technical educator with expertise in:
    - Code comprehension and documentation
    - Explaining complex technical concepts clearly
    - PySpark, SQL, Python, and distributed computing
    
    Provide clear, accurate, and educational explanations.
    """
    
    code = dspy.InputField(desc="Code to explain")
    context = dspy.InputField(desc="Additional context about the codebase")
    explanation = dspy.OutputField(desc="Clear, expert-level explanation of what the code does and why")


class CodeAnalyzer(dspy.Module):
    """DSPy Module for analyzing code with optimized prompts."""
    
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(CodeAnalysisSignature)
        self.extract_logic = dspy.ChainOfThought(BusinessLogicExtractor)
        self.explain = dspy.ChainOfThought(CodeExplanation)
    
    def forward(self, context: str, query: str) -> dspy.Prediction:
        """
        Analyze code based on context and query.
        
        Args:
            context: Relevant code chunks
            query: User's question
            
        Returns:
            Analysis prediction
        """
        return self.analyze(context=context, query=query)
    
    def extract_business_logic(self, code: str) -> dspy.Prediction:
        """
        Extract business logic from code.
        
        Args:
            code: Code snippet
            
        Returns:
            Business logic prediction
        """
        return self.extract_logic(code=code)
    
    def explain_code(self, code: str, context: str = "") -> dspy.Prediction:
        """
        Explain code functionality.
        
        Args:
            code: Code to explain
            context: Additional context
            
        Returns:
            Explanation prediction
        """
        return self.explain(code=code, context=context)


class DSPyOptimizer:
    """Handles DSPy prompt optimization and evolution."""
    
    def __init__(
        self, 
        model_name: str = "codellama:7b",
        cache_dir: str = "./.dspy_cache"
    ):
        """
        Initialize DSPy optimizer.
        
        Args:
            model_name: Name of the LLM model
            cache_dir: Directory for caching
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.analyzer = None
        
    def setup_lm(self, base_url: str = "http://localhost:11434", temperature: float = 0.1, max_tokens: int = 2048):
        """
        Setup the language model for DSPy.
        
        Args:
            base_url: Base URL for Ollama
            temperature: Temperature for LLM sampling
            max_tokens: Maximum tokens in response
        """
        try:
            # Configure DSPy with Ollama
            lm = dspy.LM(
                model=f"ollama/{self.model_name}",
                api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens
            )
            dspy.settings.configure(lm=lm)
            
            # Initialize analyzer
            self.analyzer = CodeAnalyzer()
            
            logger.info(f"DSPy configured with Ollama model: ollama/{self.model_name} at {base_url}")
            
        except Exception as e:
            logger.error(f"Error setting up DSPy LM: {e}")
            raise
    
    def analyze_code(
        self, 
        relevant_chunks: List[Dict], 
        query: str
    ) -> str:
        """
        Analyze code using optimized prompts.
        
        Args:
            relevant_chunks: List of relevant code chunks
            query: User's question
            
        Returns:
            Analysis result
        """
        if not self.analyzer:
            raise ValueError("Language model not setup. Call setup_lm() first.")
        
        try:
            # Combine chunks into context
            context = "\n\n".join([
                f"--- Chunk {chunk.get('metadata', {}).get('chunk_id', 'N/A')} ---\n{chunk.get('text', '')}"
                for chunk in relevant_chunks
            ])
            
            # Run analysis
            result = self.analyzer(context=context, query=query)
            
            logger.info("Code analysis completed")
            return result.analysis
            
        except Exception as e:
            logger.error(f"Error during code analysis: {e}")
            raise
    
    def optimize_with_examples(
        self, 
        trainset: List[dspy.Example],
        metric: Optional[callable] = None
    ):
        """
        Optimize prompts using training examples.
        
        Args:
            trainset: List of training examples
            metric: Optional metric function for evaluation
        """
        try:
            # Use BootstrapFewShot for optimization
            from dspy.teleprompt import BootstrapFewShot
            
            config = dict(max_bootstrapped_demos=4, max_labeled_demos=16)
            
            optimizer = BootstrapFewShot(
                metric=metric if metric else self._default_metric,
                **config
            )
            
            # Compile the optimized program
            self.analyzer = optimizer.compile(
                self.analyzer,
                trainset=trainset
            )
            
            logger.info("DSPy prompts optimized with training examples")
            
        except Exception as e:
            logger.error(f"Error optimizing prompts: {e}")
            raise
    
    def _default_metric(self, example, pred, trace=None):
        """Default metric for optimization."""
        return len(pred.analysis) > 50  # Simple metric: ensure substantial response


def create_training_example(context: str, query: str, expected_analysis: str) -> dspy.Example:
    """
    Create a training example for DSPy optimization.
    
    Args:
        context: Code context
        query: User query
        expected_analysis: Expected analysis output
        
    Returns:
        DSPy Example
    """
    return dspy.Example(
        context=context,
        query=query,
        analysis=expected_analysis
    ).with_inputs("context", "query")


def main():
    """Example usage of DSPy optimizer."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize optimizer
    optimizer = DSPyOptimizer(model_name="codellama:7b")
    
    # Setup language model
    optimizer.setup_lm()
    
    print("DSPy optimizer initialized successfully")


if __name__ == "__main__":
    main()
