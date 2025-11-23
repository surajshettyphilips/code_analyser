"""
Sample test data and customer CSV for the example PySpark ETL.
"""
import csv
from pathlib import Path


def generate_sample_customers():
    """Generate sample customer data for testing."""
    customers = [
        # customer_id, name, age, country, purchase_amount, loyalty_status
        ("C001", "John Doe", 45, "USA", 1500.50, "Gold"),
        ("C002", "Jane Smith", 28, "UK", 750.25, "Silver"),
        ("C003", "Bob Johnson", 65, "Canada", 2100.00, "Gold"),
        ("C004", "Alice Brown", 23, "USA", 350.75, "Bronze"),
        ("C005", "Charlie Wilson", 55, "USA", 980.00, "Silver"),
        ("C006", "Diana Davis", 72, "UK", 1800.50, "Gold"),
        ("C007", "Edward Miller", 19, "Canada", 420.00, "Bronze"),
        ("C008", "Fiona Garcia", 38, "USA", 1250.75, "Silver"),
        ("C009", "George Martinez", 61, "USA", 3200.00, "Gold"),
        ("C010", "Helen Rodriguez", 29, "UK", 680.50, "Silver"),
        ("C011", "Ivan Lopez", 45, "Canada", 1450.00, "Gold"),
        ("C012", "Julia Hernandez", 52, "USA", 890.25, "Silver"),
        ("C013", "Kevin Gonzalez", 25, "UK", 520.00, "Bronze"),
        ("C014", "Laura Wilson", 68, "USA", 2500.75, "Gold"),
        ("C015", "Michael Anderson", 33, "Canada", 1100.00, "Silver"),
    ]
    
    return customers


def create_sample_csv(output_path: str = "data/input/customers.csv"):
    """Create a sample CSV file for testing."""
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    customers = generate_sample_customers()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([
            "customer_id", "name", "age", "country", 
            "purchase_amount", "loyalty_status"
        ])
        # Write data
        writer.writerows(customers)
    
    print(f"Created sample customer data at: {output_path}")
    print(f"Total customers: {len(customers)}")


if __name__ == "__main__":
    create_sample_csv()
