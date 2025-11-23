#!/usr/bin/env python
"""
Quick start script for the PySpark Code Analyzer.
Automates the setup and first-run experience.
"""
import subprocess
import sys
import os
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"→ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(f"  {e.stderr}")
        return False


def check_ollama():
    """Check if Ollama is installed and running."""
    print_header("Checking Ollama Installation")
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Ollama is installed and running")
        
        # Check for codellama model
        if "codellama:7b" in result.stdout:
            print("✓ CodeLlama:7b model is available")
            return True
        else:
            print("⚠ CodeLlama:7b model not found")
            print("\nTo install CodeLlama, run:")
            print("  ollama pull codellama:7b")
            return False
            
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        print("\nPlease install Ollama from: https://ollama.ai")
        return False
    except subprocess.CalledProcessError:
        print("✗ Ollama is not running")
        print("\nPlease start Ollama:")
        print("  ollama serve")
        return False


def setup_environment():
    """Set up the environment file."""
    print_header("Setting Up Environment")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if env_example.exists():
        try:
            env_example.read_text()
            with open(env_file, 'w') as f:
                f.write(env_example.read_text())
            print("✓ Created .env file from .env.example")
            return True
        except Exception as e:
            print(f"✗ Error creating .env file: {e}")
            return False
    else:
        print("⚠ .env.example not found")
        return False


def create_directories():
    """Create necessary directories."""
    print_header("Creating Directories")
    
    directories = [
        "logs",
        "data/input",
        "data/output",
        "data/chromadb"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}/")
    
    return True


def run_example():
    """Run the example workflow."""
    print_header("Running Example Workflow")
    
    example_file = "examples/example_pyspark_etl.py"
    
    if not Path(example_file).exists():
        print(f"✗ Example file not found: {example_file}")
        return False
    
    print("\n--- Stage 1: Processing File ---")
    success = run_command(
        f"python main.py --mode process --file {example_file}",
        "Processing example PySpark file"
    )
    
    if not success:
        return False
    
    print("\n--- Stage 2: Querying Code ---")
    query = "What business rules are implemented in this code?"
    success = run_command(
        f'python main.py --mode query --query "{query}"',
        "Analyzing code with query"
    )
    
    return success


def main():
    """Main quick start process."""
    print_header("PySpark Code Analyzer - Quick Start")
    
    print("This script will help you get started with the PySpark Code Analyzer.\n")
    
    # Step 1: Check Ollama
    if not check_ollama():
        print("\n⚠ Please install and configure Ollama before continuing.")
        sys.exit(1)
    
    # Step 2: Setup environment
    if not setup_environment():
        print("\n⚠ Environment setup failed.")
        sys.exit(1)
    
    # Step 3: Create directories
    if not create_directories():
        print("\n⚠ Directory creation failed.")
        sys.exit(1)
    
    # Step 4: Ask about running example
    print_header("Ready to Run Example")
    print("Would you like to run an example analysis now?")
    print("This will:")
    print("  1. Process the example PySpark ETL file")
    print("  2. Run a sample query to demonstrate the analysis")
    print()
    
    response = input("Run example? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        if run_example():
            print_header("Quick Start Complete!")
            print("✓ Example completed successfully!")
            print("\nNext steps:")
            print("  1. Try interactive mode:")
            print("     python main.py --mode interactive --file examples/example_pyspark_etl.py")
            print()
            print("  2. Process your own files:")
            print("     python main.py --mode process --file your_file.py")
            print()
            print("  3. View the README.md for more information")
        else:
            print("\n⚠ Example execution encountered errors.")
            print("Check logs/app.log for details.")
    else:
        print_header("Setup Complete!")
        print("✓ Environment is ready!")
        print("\nTo get started:")
        print("  python main.py --mode process --file examples/example_pyspark_etl.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nQuick start interrupted. Exiting.")
        sys.exit(0)
