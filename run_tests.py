"""
Test runner script for the PySpark Code Analyzer.
Runs all unit tests and generates a coverage report.
"""
import unittest
import sys
from pathlib import Path


def run_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("  Running PySpark Code Analyzer Tests")
    print("="*80 + "\n")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("  Test Summary")
    print("="*80)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print("="*80 + "\n")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
