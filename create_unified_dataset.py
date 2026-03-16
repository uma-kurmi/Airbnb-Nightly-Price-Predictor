import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, min, max, lit

# ── Initialize Spark ─────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("Create Unified Dataset") \
    .config("spark.driver.memory", "6g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

print("🚀 Creating unified dataset from all cleaned cities...")

# ── Define cities ────────────────────────────────────────────────────────────
cities = ["nyc", "london", "amsterdam", "barcelona"]
input_dir = "output/cleaned_data"
output_dir = "output/unified"
os.makedirs(output_dir, exist_ok=True)

# ── Load cleaned data ────────────────────────────────────────────────────────
all_dfs = []
for city in cities:
    file_path = f"{input_dir}/{city}_listings_cleaned.parquet"
    if os.path.exists(file_path):
        print(f"🔄 Loading {city}...")
        df = spark.read.parquet(file_path)
        if "city" not in df.columns:
            df = df.withColumn("city", lit(city))
        print(f"   - {df.count()} listings loaded")
        all_dfs.append(df)
    else:
        print(f"⚠️  WARNING: {file_path} not found!")

# ── Validate load ────────────────────────────────────────────────────────────
if not all_dfs:
    print("❌ ERROR: No cleaned city data found. Exiting.")
    spark.stop()
    exit(1)

# ── Keep only common columns across all cities ───────────────────────────────
common_columns = set(all_dfs[0].columns)
for df in all_dfs[1:]:
    common_columns &= set(df.columns)

print(f"\n✅ Common columns across cities: {len(common_columns)}")

standardized_dfs = [df.select(*sorted(common_columns)) for df in all_dfs]

# ── Union all cities ─────────────────────────────────────────────────────────
print("\n🔗 Combining datasets...")
unified_df = standardized_dfs[0]
for df in standardized_dfs[1:]:
    unified_df = unified_df.union(df)

# ── Summary & Stats ──────────────────────────────────────────────────────────
print("\n📊 Unified dataset summary:")
print(f"Total rows: {unified_df.count()}")
print("\nListings per city:")
unified_df.groupBy("city").count().orderBy("city").show()

print("\n💰 Price statistics (using price_clean):")
unified_df.groupBy("city").agg(
    count("price_clean").alias("count"),
    avg("price_clean").alias("avg_price"),
    min("price_clean").alias("min_price"),
    max("price_clean").alias("max_price")
).orderBy("city").show()

# ── Save to disk ─────────────────────────────────────────────────────────────
output_path = f"{output_dir}/all_cities_unified.parquet"
print(f"\n💾 Saving unified dataset to: {output_path}")
unified_df.coalesce(4).write.mode("overwrite").parquet(output_path)

print("\n✅ Unified dataset created successfully!")
print(f"Total columns: {len(unified_df.columns)}")
print(f"Total rows: {unified_df.count()}")

# ── Show sample ──────────────────────────────────────────────────────────────
print("\n🔍 Sample data:")
unified_df.select("city", "price_clean", "room_type", "latitude", "longitude").show(5)

spark.stop()
