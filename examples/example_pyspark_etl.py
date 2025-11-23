"""
Example PySpark script for data processing.
This demonstrates a typical PySpark ETL pipeline with business logic.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as spark_sum, avg, count
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from datetime import datetime


def create_spark_session(app_name: str = "DataProcessingApp"):
    """
    Create and configure a Spark session.
    
    Business Rule: Application runs in local mode for development.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_customer_data(spark, file_path: str):
    """
    Load customer data from CSV file.
    
    Business Rule: Customer data must have specific schema for validation.
    """
    schema = StructType([
        StructField("customer_id", StringType(), False),
        StructField("name", StringType(), False),
        StructField("age", IntegerType(), True),
        StructField("country", StringType(), True),
        StructField("purchase_amount", DoubleType(), True),
        StructField("loyalty_status", StringType(), True)
    ])
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    return df


def apply_business_rules(df):
    """
    Apply business rules for customer segmentation.
    
    Business Rules:
    1. Customers with purchase_amount > 1000 are "Premium"
    2. Customers with purchase_amount between 500-1000 are "Standard"
    3. Customers with purchase_amount < 500 are "Basic"
    4. Customers aged > 60 get a 15% senior discount
    5. Customers aged 18-30 get a 10% youth discount
    6. Premium customers in loyalty program get additional 5% discount
    """
    # Rule 1-3: Customer Segmentation
    df = df.withColumn(
        "customer_segment",
        when(col("purchase_amount") > 1000, "Premium")
        .when((col("purchase_amount") >= 500) & (col("purchase_amount") <= 1000), "Standard")
        .otherwise("Basic")
    )
    
    # Rule 4-5: Age-based discounts
    df = df.withColumn(
        "discount_rate",
        when(col("age") > 60, 0.15)
        .when((col("age") >= 18) & (col("age") <= 30), 0.10)
        .otherwise(0.0)
    )
    
    # Rule 6: Additional loyalty discount for premium customers
    df = df.withColumn(
        "discount_rate",
        when(
            (col("customer_segment") == "Premium") & (col("loyalty_status") == "Gold"),
            col("discount_rate") + 0.05
        ).otherwise(col("discount_rate"))
    )
    
    # Calculate final amount after discount
    df = df.withColumn(
        "final_amount",
        col("purchase_amount") * (1 - col("discount_rate"))
    )
    
    return df


def filter_valid_customers(df):
    """
    Filter customers based on validation rules.
    
    Business Rules:
    1. Age must be between 18 and 120
    2. Purchase amount must be positive
    3. Country must not be null
    """
    valid_df = df.filter(
        (col("age").between(18, 120)) &
        (col("purchase_amount") > 0) &
        (col("country").isNotNull())
    )
    
    return valid_df


def calculate_metrics_by_country(df):
    """
    Calculate key business metrics grouped by country.
    
    Business Metrics:
    - Total revenue (sum of final amounts)
    - Average customer age
    - Customer count
    - Average discount rate
    """
    metrics = df.groupBy("country").agg(
        spark_sum("final_amount").alias("total_revenue"),
        avg("age").alias("avg_customer_age"),
        count("customer_id").alias("customer_count"),
        avg("discount_rate").alias("avg_discount_rate")
    )
    
    # Business Rule: Only include countries with at least 10 customers
    metrics = metrics.filter(col("customer_count") >= 10)
    
    return metrics.orderBy(col("total_revenue").desc())


def identify_high_value_customers(df):
    """
    Identify high-value customers for special campaigns.
    
    Business Rule: High-value customers are Premium segment with 
    purchase amount > 2000 OR Gold loyalty status
    """
    high_value = df.filter(
        ((col("customer_segment") == "Premium") & (col("purchase_amount") > 2000)) |
        (col("loyalty_status") == "Gold")
    )
    
    return high_value.select(
        "customer_id",
        "name",
        "customer_segment",
        "purchase_amount",
        "final_amount",
        "loyalty_status"
    )


def save_results(df, output_path: str, format: str = "parquet"):
    """
    Save processed data to specified format.
    
    Business Rule: Data is partitioned by country for efficient querying.
    """
    df.write \
        .mode("overwrite") \
        .partitionBy("country") \
        .format(format) \
        .save(output_path)


def main():
    """
    Main ETL pipeline orchestration.
    
    Business Process:
    1. Load customer data
    2. Apply business rules and segmentation
    3. Filter valid customers
    4. Calculate country-level metrics
    5. Identify high-value customers
    6. Save results
    """
    print(f"Starting ETL pipeline at {datetime.now()}")
    
    # Initialize Spark
    spark = create_spark_session("CustomerAnalyticsPipeline")
    
    try:
        # Load data
        print("Loading customer data...")
        customer_df = load_customer_data(spark, "data/input/customers.csv")
        print(f"Loaded {customer_df.count()} customer records")
        
        # Apply business rules
        print("Applying business rules...")
        processed_df = apply_business_rules(customer_df)
        
        # Filter valid customers
        print("Filtering valid customers...")
        valid_df = filter_valid_customers(processed_df)
        print(f"Valid customers: {valid_df.count()}")
        
        # Calculate metrics
        print("Calculating country metrics...")
        country_metrics = calculate_metrics_by_country(valid_df)
        country_metrics.show()
        
        # Identify high-value customers
        print("Identifying high-value customers...")
        high_value_customers = identify_high_value_customers(valid_df)
        print(f"High-value customers: {high_value_customers.count()}")
        high_value_customers.show(10)
        
        # Save results
        print("Saving results...")
        save_results(valid_df, "data/output/processed_customers")
        save_results(country_metrics, "data/output/country_metrics")
        save_results(high_value_customers, "data/output/high_value_customers")
        
        print(f"ETL pipeline completed successfully at {datetime.now()}")
        
    except Exception as e:
        print(f"Error in ETL pipeline: {e}")
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
