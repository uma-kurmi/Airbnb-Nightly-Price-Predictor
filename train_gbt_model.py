import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import pickle

# Initialize Spark
spark = SparkSession.builder \
    .appName("Simple GBT Model") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


print("🌳 Training Gradient Boosted Trees Model...")

# ============================================
# 1. LOAD DATA
# ============================================
features_path = "output/features/engineered_features.parquet"
if not os.path.exists(features_path):
    print("❌ ERROR: Run feature_engineering.py first")
    spark.stop()
    exit(1)

df = spark.read.parquet(features_path)
print(f"✅ Loaded {df.count()} listings")

target_col = "price_clean"

# ============================================
# 2. SMART FEATURE SELECTION (CHECK DATA TYPES)
# ============================================
print("\n🎯 Smart feature selection...")

# Get actual numeric columns from the dataset
actual_numeric = []
for col_name, dtype in df.dtypes:
    if col_name == target_col:
        continue
    if dtype in ['int', 'bigint', 'float', 'double']:
        actual_numeric.append(col_name)

# Pick best numeric features that actually exist and are numeric
preferred_numeric = [
    "accommodates", "bedrooms", "bathrooms_numeric", "beds",
    "minimum_nights", "maximum_nights", "availability_30", "availability_60", 
    "availability_90", "availability_365", "number_of_reviews", "distance_to_center", 
    "amenities_count", "host_days_active", "host_response_rate_clean", 
    "host_acceptance_rate_clean", "review_scores_rating_clean", "review_scores_location_clean",
    "review_scores_cleanliness_clean", "host_is_superhost_binary", "instant_bookable_binary",
    "has_wifi", "has_kitchen", "has_ac", "has_parking", "has_reviews"
]

# EXCLUDE LEAKY FEATURES (derived from price)
leaky_features = ["price_per_person", "price_per_bedroom"]

# Use intersection of preferred and actual numeric columns, excluding leaky ones
numeric_features = [f for f in preferred_numeric if f in actual_numeric and f not in leaky_features]

# If we don't have enough, add more actual numeric columns
if len(numeric_features) < 10:
    for col_name in actual_numeric:
        if (col_name not in numeric_features and 
            len(numeric_features) < 15 and
            col_name not in leaky_features):  # Exclude leaky features
            # Skip ID columns and other non-predictive features
            if not any(skip in col_name.lower() for skip in ['id', 'url', 'name', 'description', 'price_per']):
                numeric_features.append(col_name)

# Categorical features - only use simple ones
categorical_features = []
for col_name, dtype in df.dtypes:
    if dtype == 'string' and col_name in ['city', 'room_type', 'property_type']:
        # Check if it has reasonable cardinality
        distinct_count = df.select(col_name).distinct().count()
        if 2 <= distinct_count <= 50:
            categorical_features.append(col_name)

print(f"✅ Selected {len(numeric_features)} legitimate numeric features (excluded price-derived features)")
print(f"✅ Selected {len(categorical_features)} categorical features")
print(f"Numeric: {numeric_features[:5]}...")
print(f"Categorical: {categorical_features}")
# print(f"🚫 Excluded leaky features: {leaky_features}")

# ============================================
# 3. BASIC DATA CLEANING
# ============================================
# Better outlier removal
price_quantiles = df.approxQuantile(target_col, [0.01, 0.99], 0.01)  # More aggressive
df_clean = df.filter(
    (col(target_col) >= price_quantiles[0]) & 
    (col(target_col) <= price_quantiles[1]) &
    col(target_col).isNotNull()
)
print(f"After cleaning: {df_clean.count()} listings")

# ============================================
# 4. SIMPLE PREPROCESSING
# ============================================
# Handle categorical features
indexers = []
indexed_categorical = []

for cat_feature in categorical_features:
    indexer = StringIndexer(
        inputCol=cat_feature,
        outputCol=f"{cat_feature}_indexed",
        handleInvalid="keep"
    )
    indexers.append(indexer)
    indexed_categorical.append(f"{cat_feature}_indexed")

# Combine features
all_features = numeric_features + indexed_categorical
assembler = VectorAssembler(
    inputCols=all_features,
    outputCol="features",
    handleInvalid="skip"
)

# Better GBT - Optimized hyperparameters for performance
gbt = GBTRegressor(
    featuresCol="features",
    labelCol=target_col,
    predictionCol="prediction",
    maxDepth=8,          # Deeper trees (was 5)
    maxIter=100,         # More iterations (was 50)  
    stepSize=0.05,       # Slower learning (was 0.1)
    maxBins=64,          # More bins (was 32)
    subsamplingRate=0.8, # Add subsampling for better generalization
    seed=42
)

# Create simple pipeline
pipeline = Pipeline(stages=indexers + [assembler, gbt])

# ============================================
# 5. SIMPLE TRAIN/TEST SPLIT (no cross-validation)
# ============================================
print("\n📊 Simple train/test split...")
train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)
print(f"Training: {train_df.count()}, Test: {test_df.count()}")

# ============================================
# 6. TRAIN MODEL (fast)
# ============================================
print("\n🚀 Training model...")
model = pipeline.fit(train_df)
print("✅ Model trained!")

# ============================================
# 7. EVALUATE
# ============================================
print("\n📈 Evaluating...")

# Predictions
train_pred = model.transform(train_df)
test_pred = model.transform(test_df)

# Metrics
rmse_eval = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
mae_eval = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")
r2_eval = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")

train_rmse = rmse_eval.evaluate(train_pred)
train_r2 = r2_eval.evaluate(train_pred)
test_rmse = rmse_eval.evaluate(test_pred)
test_r2 = r2_eval.evaluate(test_pred)

print(f"\nTraining RMSE: ${train_rmse:.2f}, R²: {train_r2:.4f}")
print(f"Test RMSE: ${test_rmse:.2f}, R²: {test_r2:.4f}")

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================
print("\n🔍 Feature importance...")
gbt_model = model.stages[-1]
importances = gbt_model.featureImportances.toArray()

# Create importance list
importance_data = [(str(feature), float(imp)) for feature, imp in zip(all_features, importances)]
importance_df = spark.createDataFrame(importance_data, ["feature", "importance"])

print("\nTop 10 Features:")
importance_df.orderBy(col("importance").desc()).show(10, truncate=False)

# ============================================
# 9. SAVE MODEL
# ============================================
os.makedirs("output/models", exist_ok=True)
model_path = "output/models/gbt_regressor"
model.write().overwrite().save(model_path)

# Save metrics
metrics = {
    "model_type": "Legitimate GBT (No Data Leakage)",
    "test_metrics": {
        "rmse": float(test_rmse),
        "r2": float(test_r2)
    },
    "train_metrics": {
        "rmse": float(train_rmse), 
        "r2": float(train_r2)
    },
    "feature_count": len(all_features),
    "excluded_features": leaky_features
}

with open("output/models/gbt_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

# Save feature importance
importance_df.toPandas().to_csv("output/models/gbt_feature_importance.csv", index=False)

print(f"\n✅ Legitimate GBT results (no data leakage)!")
print(f"📁 Model saved to: {model_path}")
print(f"📊 Test RMSE: ${test_rmse:.2f}")
print(f"📊 Test R²: {test_r2:.4f}")
print(f"🎯 These are realistic, defensible results for academic evaluation!")

# Show sample predictions
print("\n🔍 Sample predictions:")
test_pred.select(target_col, "prediction", "city", "room_type").show(10)

spark.stop()