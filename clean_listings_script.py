import os
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = "data/raw"
CITIES = ["nyc", "london", "amsterdam", "barcelona"]
OUTPUT_DIR = "output/cleaned_data"
REPARTITION_COUNT = 8  # Tweak based on system specs

# ── Initialize Spark Session ──────────────────────────────────────────────────
spark = (
    SparkSession.builder
        .appName("Clean Airbnb Listings – All Cities")
        .master("local[*]")
        .config("spark.driver.memory", "6g")  # Adjust as needed
        .getOrCreate()
)

# ── Price Cleaning Logic (No UDFs) ─────────────────────────────────────────────
def add_clean_price(df):
    df = df.withColumn("price_tmp", F.regexp_replace("price", "[$€£\\s]", ""))
    df = df.withColumn("price_tmp", F.when(
        (F.instr("price_tmp", ".") < F.instr("price_tmp", ",")) & (F.instr("price_tmp", ".") > 0),
        F.regexp_replace(F.regexp_replace("price_tmp", "\\.", ""), ",", ".")
    ).otherwise(
        F.regexp_replace("price_tmp", ",", "")
    ))
    return df.withColumn("price_clean", F.col("price_tmp").cast(DoubleType())).drop("price_tmp")

# ── Cleaning Function ─────────────────────────────────────────────────────────
def clean_city(city):
    input_path = os.path.join(BASE_DIR, city, "listings.csv")
    output_path = os.path.join(OUTPUT_DIR, f"{city}_listings_cleaned.parquet")

    if not os.path.exists(input_path):
        print(f"❌ {city.upper()}: File not found at {input_path}")
        return

    df = (
        spark.read
            .option("header", "true")
            .option("multiLine", "true")
            .option("quote", '"')
            .option("escape", '"')
            .option("mode", "PERMISSIVE")
            .csv(input_path)
    )

    before = df.count()
    df = add_clean_price(df)
    df = df.na.drop(subset=["price_clean", "latitude", "longitude"])
    df = df.dropDuplicates(["id"])

    # Remove price outliers (IQR filter)
    q1, q3 = df.approxQuantile("price_clean", [0.25, 0.75], 0.05)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df.filter((F.col("price_clean") >= lower) & (F.col("price_clean") <= upper))

    df = df.repartition(REPARTITION_COUNT)
    after = df.count()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.write.mode("overwrite").parquet(output_path)

    print(f"✅ {city.upper()}: kept {after:,} / {before:,} rows ({after / before * 100:.1f}%)")

# ── Execute for All Cities ────────────────────────────────────────────────────
for city in CITIES:
    clean_city(city)

spark.stop()
