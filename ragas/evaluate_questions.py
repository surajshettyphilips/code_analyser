"""
RAGAs evaluation script for batch question processing.
Reads questions from Excel, queries Stage 2, and evaluates with RAGAs.
"""
import os
# Disable PostHog telemetry to avoid SSL errors
os.environ['POSTHOG_DISABLED'] = '1'
os.environ['DISABLE_TELEMETRY'] = '1'

import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

# Disable additional telemetry
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='posthog')

# Suppress SSL and PostHog errors BEFORE any other imports
logging.basicConfig(level=logging.INFO)
logging.getLogger('posthog').setLevel(logging.CRITICAL)
logging.getLogger('posthog').disabled = True
logging.getLogger('backoff').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('urllib3.connectionpool').setLevel(logging.CRITICAL)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chromadb_manager import ChromaDBManager
from src.dspy_analyzer import DSPyOptimizer
from src.config_loader import Config
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGasEvaluator:
    """Evaluates questions using RAGAs framework."""
    
    def __init__(self, input_excel: str, output_excel: str = None):
        """
        Initialize RAGAs evaluator.
        
        Args:
            input_excel: Path to input Excel file with questions
            output_excel: Path to output Excel file (optional)
        """
        self.input_excel = input_excel
        self.output_excel = output_excel or input_excel.replace('.xlsx', '_evaluated.xlsx')
        
        # Load configuration
        self.config = Config()
        
        # Initialize components
        self.chromadb = ChromaDBManager(
            collection_name=self.config.get("chromadb.collection_name", "pyspark_code_chunks"),
            persist_directory=self.config.get("chromadb.persist_directory", "./data/chromadb")
        )
        
        self.dspy_optimizer = DSPyOptimizer(
            model_name=self.config.get("llm.model", "codellama:7b"),
            cache_dir=self.config.get("dspy.cache_dir", "./.dspy_cache")
        )
        
        # Setup language model
        self.dspy_optimizer.setup_lm(
            base_url=self.config.get("llm.base_url", "http://localhost:11434"),
            temperature=self.config.get("llm.temperature", 0.1),
            max_tokens=self.config.get("llm.max_tokens", 2048)
        )
        
        # Initialize LLM and embeddings for RAGAs
        self.llm = Ollama(
            model=self.config.get("llm.model", "codellama:7b"),
            base_url=self.config.get("llm.base_url", "http://localhost:11434")
        )
        
        self.embeddings = OllamaEmbeddings(
            model=self.config.get("llm.model", "codellama:7b"),
            base_url=self.config.get("llm.base_url", "http://localhost:11434")
        )
        
        logger.info("RAGAs evaluator initialized successfully")
    
    def load_questions(self) -> pd.DataFrame:
        """
        Load questions from Excel file.
        
        Returns:
            DataFrame with questions and ground truth
        """
        try:
            df = pd.read_excel(self.input_excel)
            logger.info(f"Loaded {len(df)} questions from {self.input_excel}")
            
            # Validate required columns
            required_cols = ['question', 'ground_truth']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Add answer column if not exists (for compatibility)
            if 'answer' not in df.columns:
                df['answer'] = df['ground_truth']
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def query_stage2(self, question: str, n_results: int = 3) -> Dict:
        """
        Query Stage 2 (retrieval + analysis) for a question.
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve
            
        Returns:
            Dictionary with AI response and contexts
        """
        try:
            # Retrieve relevant chunks from ChromaDB
            relevant_chunks = self.chromadb.query_relevant_chunks(
                query=question,
                n_results=n_results
            )
            
            # Extract context
            contexts = [chunk["content"] for chunk in relevant_chunks]
            
            # Generate answer using DSPy
            analysis_result = self.dspy_optimizer.analyze_code(
                relevant_chunks=[{"text": ctx, "metadata": {"chunk_id": i}} 
                               for i, ctx in enumerate(contexts)],
                query=question
            )
            
            ai_response = analysis_result if isinstance(analysis_result, str) else \
                         analysis_result.get("explanation", "No answer generated")
            
            return {
                "ai_response": ai_response,
                "contexts": contexts
            }
            
        except Exception as e:
            logger.error(f"Error querying Stage 2 for question '{question}': {e}")
            return {
                "ai_response": f"Error: {str(e)}",
                "contexts": []
            }
    
    def process_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all questions through Stage 2.
        
        Args:
            df: DataFrame with questions
            
        Returns:
            DataFrame with AI responses and contexts
        """
        logger.info("Processing questions through Stage 2...")
        
        ai_responses = []
        contexts_list = []
        
        for idx, row in df.iterrows():
            question = row['question']
            logger.info(f"Processing question {idx + 1}/{len(df)}: {question}")
            
            result = self.query_stage2(question)
            ai_responses.append(result['ai_response'])
            
            # Convert contexts list to string for Excel storage
            contexts_str = "\n---\n".join(result['contexts'])
            contexts_list.append(contexts_str)
        
        # Add new columns
        df['AI_response'] = ai_responses
        df['contexts'] = contexts_list
        
        logger.info("All questions processed successfully")
        return df
    
    def save_intermediate_results(self, df: pd.DataFrame):
        """
        Save intermediate results to Excel.
        
        Args:
            df: DataFrame with all columns
        """
        try:
            df.to_excel(self.output_excel, index=False)
            logger.info(f"Intermediate results saved to {self.output_excel}")
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
            raise
    
    def run_ragas_evaluation(self, df: pd.DataFrame) -> Dict:
        """
        Run RAGAs evaluation on the processed data.
        
        Args:
            df: DataFrame with all columns including AI_response and contexts
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting RAGAs evaluation...")
        
        try:
            # Prepare data for RAGAs
            questions = df['question'].tolist()
            ground_truths = df['ground_truth'].tolist()
            answers = df['AI_response'].tolist()
            
            # Convert contexts string back to list
            contexts_list = []
            for contexts_str in df['contexts']:
                if pd.notna(contexts_str) and contexts_str:
                    contexts = contexts_str.split("\n---\n")
                    contexts_list.append(contexts)
                else:
                    contexts_list.append([])
            
            # Create dataset for RAGAs
            eval_dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts_list,
                "ground_truth": ground_truths
            })
            
            # Evaluate using RAGAs metrics
            result = evaluate(
                eval_dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy
                ],
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Extract metric values
            cp = result['context_precision']
            cp_val = float(cp[0]) if isinstance(cp, list) else float(cp)
            cr = result['context_recall']
            cr_val = float(cr[0]) if isinstance(cr, list) else float(cr)
            f = result['faithfulness']
            f_val = float(f[0]) if isinstance(f, list) else float(f)
            ar = result['answer_relevancy']
            ar_val = float(ar[0]) if isinstance(ar, list) else float(ar)
            
            metrics = {
                'context_precision': cp_val,
                'context_recall': cr_val,
                'faithfulness': f_val,
                'answer_relevancy': ar_val
            }
            
            # Log results
            logger.info("RAGAs Evaluation Results:")
            logger.info(f"Context Precision: {cp_val:.4f}")
            logger.info(f"Context Recall: {cr_val:.4f}")
            logger.info(f"Faithfulness: {f_val:.4f}")
            logger.info(f"Answer Relevancy: {ar_val:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during RAGAs evaluation: {e}")
            raise
    
    def save_final_results(self, df: pd.DataFrame, metrics: Dict):
        """
        Save final results with RAGAs metrics to Excel.
        
        Args:
            df: DataFrame with all data
            metrics: RAGAs evaluation metrics
        """
        try:
            # Add metrics as new columns (same value for all rows)
            df['ragas_context_precision'] = metrics['context_precision']
            df['ragas_context_recall'] = metrics['context_recall']
            df['ragas_faithfulness'] = metrics['faithfulness']
            df['ragas_answer_relevancy'] = metrics['answer_relevancy']
            
            # Save to Excel
            df.to_excel(self.output_excel, index=False)
            logger.info(f"Final results with RAGAs metrics saved to {self.output_excel}")
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
            raise
    
    def run(self):
        """Execute the complete evaluation pipeline."""
        try:
            start_time = datetime.now()
            logger.info("="*80)
            logger.info("Starting RAGAs Evaluation Pipeline")
            logger.info(f"Input: {self.input_excel}")
            logger.info(f"Output: {self.output_excel}")
            logger.info("="*80)
            
            # Step 1: Load questions
            df = self.load_questions()
            
            # Step 2: Process questions through Stage 2
            df = self.process_questions(df)
            
            # Step 3: Save intermediate results
            self.save_intermediate_results(df)
            
            # Step 4: Run RAGAs evaluation
            metrics = self.run_ragas_evaluation(df)
            
            # Step 5: Save final results with metrics
            self.save_final_results(df, metrics)
            
            # Summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("="*80)
            logger.info("RAGAs Evaluation Pipeline Completed Successfully")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Questions processed: {len(df)}")
            logger.info(f"Results saved to: {self.output_excel}")
            logger.info("="*80)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main entry point for RAGAs evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGAs Evaluation Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Excel file with questions and ground_truth columns"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output Excel file (optional, defaults to input_evaluated.xlsx)"
    )
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = RAGasEvaluator(
        input_excel=args.input,
        output_excel=args.output
    )
    
    evaluator.run()


if __name__ == "__main__":
    main()
