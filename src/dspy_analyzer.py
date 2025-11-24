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
    
    You are an expert code analyst. Your ONLY job is to analyze the EXACT code provided to you.
    
    CRITICAL RULES - VIOLATION OF THESE RULES IS UNACCEPTABLE:
    1. READ THE CODE: The code is provided in the context below. READ IT COMPLETELY.
    2. ANALYZE ONLY WHAT YOU SEE: Explain ONLY code that is explicitly shown in the context
    3. DO NOT HALLUCINATE: DO NOT make up functionality, column names, or logic not in the code
    4. QUOTE ACTUAL NAMES: When you mention a column, table, or variable, it MUST be from the code
    5. IF/WHEN conditions: List every actual IF/WHEN/WHERE condition from the code
    6. VALUES: Quote actual values being compared (e.g., 'open', 'ZPO', 'Y', 'N')
    7. NO ASSUMPTIONS: If something is not in the code, explicitly state it's not present
    8. IGNORE COMMENTS: Ignore # Python, -- SQL, /* */ multi-line comments
    9. NO SUGGESTIONS: DO NOT provide code improvements, fixes, or syntax corrections
    10. ACTUAL BEHAVIOR: Explain what the code DOES, not what it should/could do
    11. WRONG ANSWER = FAILURE: Talking about code not present is a critical error
    
    BEFORE YOU ANSWER - VERIFY:
    - Did I read all the code provided in the context?
    - Are the column/table names I'm mentioning actually in the code?
    - Am I making up any logic not explicitly shown?
    - Am I talking about Python sets/lists when the code shows SQL?
    
    FOR SQL QUERIES - MANDATORY STEPS:
    1. Identify SELECT clause: What columns are selected? (use exact names)
    2. Identify FROM clause: What tables? (use exact names)
    3. Identify JOINs: What tables join and on what conditions?
    4. List EVERY WHERE condition with exact column names and values
    5. For IF/CASE: Break down outer condition, then each nested branch
    6. For IN clauses: List ALL values present
    7. For functions (DATEADD, COALESCE, etc.): Explain with exact parameters
    
    FOR ALERT QUERIES - USE THIS EXACT FORMAT:
    "[AlertName] is an alert that triggers when ALL of these conditions are met:
    - field1 = 'value' (or equals 'value')
    - field2 is one of ('val1', 'val2', 'val3')
    - field3 is 'value'
    - field4 [operator] [value/expression]"
    
    List EVERY condition with AND between them. Use exact column names and values from the query.
    
    YOUR RESPONSE MUST:
    - Start by identifying the type of code (SQL alert query, Python function, etc.)
    - If it's an alert query, identify the alert name from the SELECT clause
    - List ALL conditions in a clear bullet point format
    - Use plain, natural language for business users
    - Quote exact field/column names from the code
    - Quote exact values being compared
    - State AND/OR logic explicitly
    - Reference only what exists in the provided code
    
    WRONG EXAMPLES (HALLUCINATION):
    ❌ "uses Python sets to remove duplicates" (when code shows SQL query)
    ❌ "checks storage_location_id" (when that column isn't in the code)
    ❌ "validates vendor is third-party" (when vendor isn't mentioned)
    
    RIGHT EXAMPLES (GROUNDED IN CODE):
    ✓ "PO_DUE_3rd_Party is an alert that triggers when:"
    ✓ "- orderlinestatus = 'open'"
    ✓ "- Sales_Doc_Type is one of ('ZPO','ZSO','ZDM')"
    ✓ "- PORelevantFlag = 'Y'"
    ✓ "- Acct_Grp is in ('0001')"
    ✓ "- Created_On is null"
    ✓ "- RDD < DATEADD(MONTH, 6, current_date())"
    
    Remember: Your job is to read and explain the ACTUAL code provided. Nothing else.
    """
    
    context = dspy.InputField(desc="The ACTUAL CODE to analyze. This contains SQL queries, Python code, or other code. READ EVERY LINE. ONLY use column names, table names, functions, and values that appear in THIS CODE. If you mention something, it MUST be in this context.")
    query = dspy.InputField(desc="User's specific question about the code")
    analysis = dspy.OutputField(desc="Detailed explanation of what the ACTUAL CODE does, using exact names and values from the code, step-by-step breakdown of all conditions")


class BusinessLogicExtractor(dspy.Signature):
    """Signature for extracting specific business logic from code.
    
    You are a business analyst. Extract business logic ONLY from what you can actually see in the code.
    
    STRICT RULES:
    1. Analyze ONLY the code that is explicitly provided
    2. DO NOT invent or assume business rules not visible in the code
    3. DO NOT hallucinate functionality or logic that isn't shown
    4. QUOTE actual field/column names from the code when explaining
    5. If you mention a condition, it MUST appear in the code word-for-word
    6. If a business rule is not clear from the code, say so explicitly
    7. Ignore all comments (# Python, -- SQL, /* */ multi-line)
    8. DO NOT suggest improvements or corrections
    9. Explain ONLY what the code actually does, not what it should do
    10. If the code doesn't check for something, don't say it does
    
    FOR BUSINESS LOGIC EXTRACTION:
    - Identify EACH business rule/condition that exists in the code
    - QUOTE the exact field names being checked (e.g., "checks Name_Of_Orderer field")
    - Explain what EACH filter or condition is validating
    - List EACH value being compared (e.g., 'open', 'ZPO', market names)
    - Describe the business purpose of EACH validation you can see
    - List EACH data transformation or calculation present
    - Break down complex nested IF/CASE logic level by level
    - Explain date/time validations with exact column names
    - For IN clauses, list all the actual values
    
    YOUR EXPLANATION SHOULD:
    - Describe business rules and logic you can see in the code
    - Use plain language understandable by business users
    - Reference actual column/field names from the code in quotes
    - Quote actual values being compared
    - Stay factual and grounded in the actual code
    - Avoid technical jargon when possible
    
    WRONG: "validates vendor is not third-party" (when code doesn't mention vendor)
    RIGHT: "checks if Name_Of_Orderer is empty or blank" (using actual column from code)
    
    Base your analysis entirely on the provided code. Do not guess or invent conditions.
    """
    
    code = dspy.InputField(desc="The actual code to analyze. Read every condition carefully. Only use field/column names that appear in this code.")
    business_logic = dspy.OutputField(desc="Business rules from the code, with each condition explained using actual field names and values from the code")


class CodeExplanation(dspy.Signature):
    """Signature for explaining code functionality.
    
    You are a code explainer. Explain ONLY what you can actually see in the provided code.
    
    MANDATORY RULES:
    1. Explain ONLY the code that is explicitly shown
    2. DO NOT make up or assume functionality not visible in the code
    3. DO NOT hallucinate features, methods, or logic that aren't present
    4. QUOTE actual variable/column names from the code
    5. If you mention a field or condition, it MUST be in the provided code
    6. If something is unclear or not shown, state that explicitly
    7. Ignore all comments (# Python, -- SQL, /* */ multi-line)
    8. DO NOT provide suggestions, fixes, or improvements
    9. Describe what the code ACTUALLY does, based solely on what you see
    10. DO NOT mention what it should do, could do, or might do
    11. If code doesn't check something, don't claim it does
    
    FOR CODE WITH LOGIC/CONDITIONS:
    - List EVERY condition present in the code
    - QUOTE the exact column/variable names (e.g., "Sales_Document", "orderlinestatus")
    - Explain EACH logical operator (AND, OR, IF) with actual conditions
    - Detail EACH function call that appears in the code
    - Break down EACH calculation using exact field names
    - For IN clauses, list ALL the actual values
    - For nested IF/CASE, explain outer condition then each inner branch
    - Quote actual comparison values (e.g., 'open', 'ZPO', 'Y', 'N')
    
    STEP-BY-STEP:
    1. Read all column/variable names in the code
    2. List each condition one by one using exact names
    3. Explain what each condition checks using actual values
    4. For nested logic, break down level by level
    5. Don't add any conditions not in the code
    
    YOUR EXPLANATION MUST:
    - Use simple, natural language
    - Be understandable to non-technical users
    - Reference specific field/column names from the code
    - Quote actual values being compared
    - Describe the actual behavior visible in the code
    - Stay completely within the bounds of the provided code
    
    WRONG: "validates storage_location_id" (when that field isn't in the code)
    RIGHT: "checks if Sales_Doc_Type is 'ZPO' or 'ZSO'" (quoting actual field and values)
    
    Remember: If it's not in the code, don't mention it. Quote actual names and values.
    """
    
    code = dspy.InputField(desc="The actual code to explain. Read every word. Only use names and values that appear in this code.")
    context = dspy.InputField(desc="Additional context about the codebase")
    explanation = dspy.OutputField(desc="Step-by-step explanation using exact field names and values from the code, breaking down each actual condition")


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
            # Create explicit instruction prefix
            instruction = (
                "IMPORTANT: Below is the ACTUAL CODE you must analyze. "
                "Read every line carefully. Use ONLY the column names, table names, conditions, and values "
                "that appear in this code. DO NOT make up or assume anything not explicitly shown.\n\n"
            )
            
            # Combine chunks into context with clear delimiters
            code_chunks = "\n\n".join([
                f"=== CODE CHUNK {chunk.get('metadata', {}).get('chunk_id', 'N/A')} ===\n"
                f"{chunk.get('text', '')}\n"
                f"=== END OF CHUNK {chunk.get('metadata', {}).get('chunk_id', 'N/A')} ==="
                for chunk in relevant_chunks
            ])
            
            # Combine instruction with code
            context = instruction + code_chunks
            
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
