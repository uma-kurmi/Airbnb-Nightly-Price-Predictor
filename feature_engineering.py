import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, FloatType, StringType
import re

# Initialize Spark with more memory for feature engineering
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-memory 6g --executor-memory 6g pyspark-shell'

spark = SparkSession.builder \
    .appName("Feature Engineering") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

print("Starting feature engineering...")

# Load unified dataset
unified_path = "output/unified/all_cities_unified.parquet"
if not os.path.exists(unified_path):
    print("ERROR: Unified dataset not found! Run create_unified_dataset.py first")
    spark.stop()
    exit(1)

df = spark.read.parquet(unified_path)
print(f"Loaded {df.count()} listings")

# Ensure numeric price column exists
if "price" in df.columns and "price_clean" not in df.columns:
    df = df.withColumn("price_clean", regexp_replace(col("price"), "[^0-9.]", "").cast(FloatType()))

# ============================================
# HELPER FUNCTIONS
# ============================================
def clean_percentage(col_name):
    return when(col(col_name).isNotNull(),
                regexp_replace(col(col_name), "%", "").cast(FloatType()) / 100).otherwise(0)

def clean_boolean(col_name):
    return when(col(col_name) == "t", 1).when(col(col_name) == "f", 0).otherwise(0)

# ============================================
# 1. HOST FEATURES
# ============================================
print("\nCreating host features...")
if "host_response_rate" in df.columns:
    df = df.withColumn("host_response_rate_clean", clean_percentage("host_response_rate"))
if "host_acceptance_rate" in df.columns:
    df = df.withColumn("host_acceptance_rate_clean", clean_percentage("host_acceptance_rate"))
if "host_is_superhost" in df.columns:
    df = df.withColumn("host_is_superhost_binary", clean_boolean("host_is_superhost"))
if "host_has_profile_pic" in df.columns:
    df = df.withColumn("host_has_profile_pic_binary", clean_boolean("host_has_profile_pic"))
if "host_identity_verified" in df.columns:
    df = df.withColumn("host_identity_verified_binary", clean_boolean("host_identity_verified"))
if "host_since" in df.columns:
    df = df.withColumn("host_since_date", to_date(col("host_since"), "yyyy-MM-dd")) \
           .withColumn("host_days_active", 
                      when(col("host_since_date").isNotNull(), 
                           datediff(current_date(), col("host_since_date"))).otherwise(0))

# ============================================
# 2. PROPERTY FEATURES
# ============================================
print("\nCreating property features...")
for col_name in ["accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights"]:
    if col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))

if "bathrooms_text" in df.columns:
    df = df.withColumn("bathrooms_numeric", 
        regexp_extract(col("bathrooms_text"), r"(\d+\.?\d*)", 1).cast(FloatType()))
elif "bathrooms" in df.columns:
    df = df.withColumn("bathrooms_numeric", col("bathrooms").cast(FloatType()))

# ============================================
# 3. AMENITIES FEATURES
# ============================================
print("\nCreating amenities features...")
if "amenities" in df.columns:
    df = df.withColumn("amenities_cleaned", regexp_replace(col("amenities"), r'[\[\]{}"]', ""))
    df = df.withColumn("amenities_count", size(split(col("amenities_cleaned"), ",")))
    key_amenities = [
        ("has_wifi", "wifi|internet|wi-fi"),
        ("has_kitchen", "kitchen"),
        ("has_ac", "air conditioning|ac|a/c"),
        ("has_heating", "heating|heat"),
        ("has_washer", "washer|laundry"),
        ("has_parking", "parking"),
        ("has_tv", "tv|television"),
        ("has_pool", "pool"),
        ("has_gym", "gym|fitness")
    ]
    for amenity_name, pattern in key_amenities:
        df = df.withColumn(amenity_name, 
            when(lower(col("amenities_cleaned")).rlike(pattern), 1).otherwise(0))

# ============================================
# 4. AVAILABILITY FEATURES
# ============================================
print("\nCreating availability features...")
for col_name in ["availability_30", "availability_60", "availability_90", "availability_365"]:
    if col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))

if "availability_30" in df.columns:
    df = df.withColumn("availability_rate_30", col("availability_30") / 30)
if "availability_365" in df.columns:
    df = df.withColumn("availability_rate_365", col("availability_365") / 365)
if "instant_bookable" in df.columns:
    df = df.withColumn("instant_bookable_binary", clean_boolean("instant_bookable"))

# ============================================
# 5. REVIEW FEATURES
# ============================================
print("\nCreating review features...")
review_count_cols = ["number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d"]
for col_name in review_count_cols:
    if col_name in df.columns:
        df = df.withColumn(col_name, col(col_name).cast(IntegerType()))

review_score_cols = ["review_scores_rating", "review_scores_accuracy", 
                    "review_scores_cleanliness", "review_scores_checkin",
                    "review_scores_communication", "review_scores_location", 
                    "review_scores_value"]

existing_review_cols = [c for c in review_score_cols if c in df.columns]
for col_name in existing_review_cols:
    df = df.withColumn(col_name + "_clean", col(col_name).cast(FloatType()))

if existing_review_cols:
    review_sum = coalesce(col(existing_review_cols[0] + "_clean"), lit(0))
    for c in existing_review_cols[1:]:
        review_sum += coalesce(col(c + "_clean"), lit(0))
    df = df.withColumn("avg_review_score", review_sum / len(existing_review_cols))

if "reviews_per_month" in df.columns:
    df = df.withColumn("reviews_per_month", col("reviews_per_month").cast(FloatType()))
if "number_of_reviews" in df.columns:
    df = df.withColumn("has_reviews", when(col("number_of_reviews") > 0, 1).otherwise(0))

# ============================================
# 6. TEXT FEATURES
# ============================================
print("\nCreating text features...")
if "name" in df.columns:
    df = df.withColumn("name_length", length(col("name")))
    df = df.withColumn("name_word_count", size(split(col("name"), " ")))
if "description" in df.columns:
    df = df.withColumn("description_length", length(col("description")))

# ============================================
# 7. LOCATION FEATURES
# ============================================
print("\nCreating location features...")
city_centers = {
    "nyc": (40.7580, -73.9855),
    "london": (51.5074, -0.1278),
    "amsterdam": (52.3676, 4.9041),
    "barcelona": (41.3851, 2.1734),
    "paris": (48.8566, 2.3522)
}
for city, (lat, lng) in city_centers.items():
    df = df.withColumn(f"{city}_center_lat", lit(lat))
    df = df.withColumn(f"{city}_center_lng", lit(lng))

df = df.withColumn("city_center_lat",
    when(col("city") == "nyc", col("nyc_center_lat"))
    .when(col("city") == "london", col("london_center_lat"))
    .when(col("city") == "amsterdam", col("amsterdam_center_lat"))
    .when(col("city") == "barcelona", col("barcelona_center_lat"))
    .when(col("city") == "paris", col("paris_center_lat"))
    .otherwise(0))
df = df.withColumn("city_center_lng",
    when(col("city") == "nyc", col("nyc_center_lng"))
    .when(col("city") == "london", col("london_center_lng"))
    .when(col("city") == "amsterdam", col("amsterdam_center_lng"))
    .when(col("city") == "barcelona", col("barcelona_center_lng"))
    .when(col("city") == "paris", col("paris_center_lng"))
    .otherwise(0))
df = df.withColumn("distance_to_center",
    sqrt(pow((col("latitude") - col("city_center_lat")) * 69.0, 2) + 
         pow((col("longitude") - col("city_center_lng")) * 54.6, 2)))
temp_cols = [f"{city}_center_lat" for city in city_centers] + \
            [f"{city}_center_lng" for city in city_centers] + \
            ["city_center_lat", "city_center_lng"]
df = df.drop(*temp_cols)

# ============================================
# 8. CALCULATED FEATURES
# ============================================
print("\nCreating calculated features...")
if "accommodates" in df.columns:
    df = df.withColumn("price_per_person", 
                      when(col("accommodates") > 0, col("price_clean") / col("accommodates"))
                      .otherwise(col("price_clean")))
if "bedrooms" in df.columns:
    df = df.withColumn("price_per_bedroom", 
                      when(col("bedrooms") > 0, col("price_clean") / col("bedrooms"))
                      .otherwise(col("price_clean")))
if "room_type" in df.columns:
    df = df.withColumn("is_entire_home", when(col("room_type") == "Entire home/apt", 1).otherwise(0))
    df = df.withColumn("is_private_room", when(col("room_type") == "Private room", 1).otherwise(0))
    df = df.withColumn("is_shared_room", when(col("room_type") == "Shared room", 1).otherwise(0))

# ============================================
# 9. FINAL FEATURE SELECTION
# ============================================
print("\nSelecting final features...")
all_columns = df.columns
feature_columns = [c for c in all_columns if c not in ["amenities", "amenities_cleaned"] and not c.endswith("_date")]
important_original = ["id", "latitude", "longitude", "city", "neighbourhood_cleansed", 
                      "property_type", "room_type", "price", "price_clean"]
features_to_keep = list(set(important_original + feature_columns))
df_final = df.select(*[c for c in features_to_keep if c in df.columns])

# Fill numeric nulls with 0
numeric_dtypes = ['int', 'bigint', 'float', 'double']
for col_name, dtype in df_final.dtypes:
    if any(t in dtype for t in numeric_dtypes):
        df_final = df_final.withColumn(col_name, coalesce(col(col_name), lit(0)))

# ============================================
# 10. SAVE FEATURES
# ============================================
output_path = "output/features/engineered_features.parquet"
os.makedirs("output/features", exist_ok=True)

print(f"\nSaving engineered features to {output_path}")
df_final.coalesce(20).write.mode("overwrite").parquet(output_path)

print("\nFeature engineering complete!")
print(f"Total features: {len(df_final.columns)}")
print(f"Total records: {df_final.count()}")

print("\nFeature types:")
numeric_features = [c for c, t in df_final.dtypes if any(nt in t for nt in numeric_dtypes)]
string_features = [c for c, t in df_final.dtypes if t == 'string']
print(f"Numeric features: {len(numeric_features)}")
print(f"String features: {len(string_features)}")

print("\nSample data:")
df_final.select("price_clean", "distance_to_center", "amenities_count", 
                "has_wifi", "is_entire_home").show(5)

spark.stop()
