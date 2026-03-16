import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Initialize Spark
spark = SparkSession.builder \
    .appName("Explore Features") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Loading engineered features...")

# Load features
features_path = "output/features/engineered_features.parquet"
if not os.path.exists(features_path):
    print("❌ ERROR: Engineered features not found! Run feature_engineering.py first")
    spark.stop()
    exit(1)

df = spark.read.parquet(features_path)

# Use price_clean if available
target_col = "price_clean" if "price_clean" in df.columns else "price"

# ── 1. BASIC STATS ───────────────────────────────────────────────────────────
print("\n" + "="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Total records: {df.count()}")
print(f"Total features: {len(df.columns)}")

print("\nData types summary:")
dtype_counts = {}
for col_name, dtype in df.dtypes:
    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
for dtype, cnt in sorted(dtype_counts.items()):
    print(f"  {dtype}: {cnt} columns")

# ── 2. MISSING VALUE CHECK ───────────────────────────────────────────────────
print("\n" + "="*50)
print("MISSING VALUES ANALYSIS")
print("="*50)
total_rows = df.count()
null_counts = [
    (c, df.filter(col(c).isNull()).count())
    for c in df.columns
]
null_counts = [(c, cnt, cnt / total_rows * 100) for c, cnt in null_counts if cnt > 0]

if null_counts:
    print("\nColumns with missing values (Top 10):")
    for c, cnt, pct in sorted(null_counts, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  {c}: {cnt} ({pct:.2f}%)")
else:
    print("No missing values found!")

# ── 3. TARGET VARIABLE ANALYSIS ──────────────────────────────────────────────
print("\n" + "="*50)
print("TARGET VARIABLE ANALYSIS")
print("="*50)
if target_col in df.columns:
    stats = df.select(
        mean(target_col).alias("mean"),
        stddev(target_col).alias("std"),
        min(target_col).alias("min"),
        max(target_col).alias("max"),
        expr(f"percentile_approx({target_col}, 0.25)").alias("q1"),
        expr(f"percentile_approx({target_col}, 0.5)").alias("median"),
        expr(f"percentile_approx({target_col}, 0.75)").alias("q3")
    ).collect()[0]

    if stats["mean"] is not None:
        print(f"Mean {target_col}: ${stats['mean']:.2f}")
        print(f"Std deviation: ${stats['std']:.2f}")
        print(f"Min: ${stats['min']:.2f}")
        print(f"Max: ${stats['max']:.2f}")
        print(f"25th percentile: ${stats['q1']:.2f}")
        print(f"Median: ${stats['median']:.2f}")
        print(f"75th percentile: ${stats['q3']:.2f}")
    else:
        print(f"⚠️ Column '{target_col}' contains no valid numeric values.")
else:
    print(f"❌ Column '{target_col}' not found in dataset.")

# ── 4. CORRELATIONS ──────────────────────────────────────────────────────────
print("\n" + "="*50)
print("TOP FEATURES CORRELATED WITH PRICE")
print("="*50)
numeric_cols = [
    c for c, t in df.dtypes
    if t in ['int', 'bigint', 'float', 'double'] and c != target_col
]
correlations = []
for c in numeric_cols[:30]:
    try:
        corr_val = df.stat.corr(target_col, c)
        if corr_val is not None:
            correlations.append((c, corr_val))
    except:
        continue

print("\nTop 10 positive correlations:")
for c, val in sorted(correlations, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {c}: {val:.3f}")

print("\nTop 10 negative correlations:")
for c, val in sorted(correlations, key=lambda x: x[1])[:10]:
    print(f"  {c}: {val:.3f}")

# ── 5. CATEGORICAL FEATURES ──────────────────────────────────────────────────
print("\n" + "="*50)
print("CATEGORICAL FEATURES ANALYSIS")
print("="*50)

if "room_type" in df.columns:
    print("\nRoom type distribution and avg price:")
    df.groupBy("room_type").agg(
        count("*").alias("listings"),
        avg(target_col).alias("avg_price")
    ).orderBy("listings", ascending=False).show()

if "property_type" in df.columns:
    print("\nTop 10 property types:")
    df.groupBy("property_type").count().orderBy("count", ascending=False).show(10, truncate=False)


# ── 6. FEATURE ENGINEERING RESULTS ───────────────────────────────────────────
print("\n" + "="*50)
print("FEATURE ENGINEERING RESULTS")
print("="*50)

if "amenities_count" in df.columns:
    am_stats = df.agg(
        avg("amenities_count").alias("avg"),
        max("amenities_count").alias("max")
    ).collect()[0]
    print(f"\nAverage amenities: {am_stats['avg']:.1f}")
    print(f"Max amenities: {am_stats['max']}")

host_cols = ["host_is_superhost_binary", "host_response_rate_clean", "host_days_active"]
for col_name in host_cols:
    if col_name in df.columns:
        if col_name.endswith("_binary"):
            pct = df.filter(col(col_name) == 1).count() / total_rows * 100
            print(f"{col_name}: {pct:.1f}% listings")
        else:
            avg_val = df.agg(avg(col_name)).first()[0]
            print(f"{col_name}: avg = {avg_val:.2f}")

if "distance_to_center" in df.columns:
    dist = df.agg(
        avg("distance_to_center").alias("avg"),
        min("distance_to_center").alias("min"),
        max("distance_to_center").alias("max")
    ).collect()[0]
    print(f"\nDistance to city center:")
    print(f"  Avg: {dist['avg']:.2f}")
    print(f"  Range: {dist['min']:.2f} → {dist['max']:.2f}")

print("\n✅ Feature exploration complete!")

spark.stop()
