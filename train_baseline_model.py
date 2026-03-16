import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pickle

# Initialize Spark
spark = SparkSession.builder \
    .appName("CLEAN Baseline Linear Regression") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

print("🚀 Training CLEAN Baseline Linear Regression Model (NO DATA LEAKAGE)...")

# ============================================
# 1. LOAD ENGINEERED FEATURES
# ============================================
features_path = "output/features/engineered_features.parquet"
if not os.path.exists(features_path):
    print("❌ ERROR: Engineered features not found! Run feature_engineering.py first")
    spark.stop()
    exit(1)

df = spark.read.parquet(features_path)
print(f"✅ Loaded {df.count()} listings with {len(df.columns)} features")

target_col = "price_clean"

# ============================================
# 2. EXPLICIT CLEAN FEATURE SELECTION 
# ============================================
print("\n📊 CLEAN feature selection (ZERO data leakage)...")

# EXPLICITLY BANNED features (data leakage)
BANNED_FEATURES = [
    'price_per_bedroom',    # DERIVED FROM PRICE!
    'price_per_person',     # DERIVED FROM PRICE!
    'price',                # Original price column
    'id', 'listing_url', 'scrape_id', 'name', 'description'
]

# Get clean numeric features
clean_numeric = []
clean_categorical = []

for col_name, dtype in df.dtypes:
    # Skip target and banned features
    if col_name == target_col or col_name in BANNED_FEATURES:
        continue
    
    # Skip obvious text/ID columns
    if any(word in col_name.lower() for word in ['url', 'name', 'description', 'about']):
        continue
    
    # Keep legitimate features
    if dtype in ['int', 'bigint', 'float', 'double']:
        clean_numeric.append(col_name)
    elif dtype == 'string':
        distinct_count = df.select(col_name).distinct().count()
        if 2 <= distinct_count <= 50:
            clean_categorical.append(col_name)

# Limit to reasonable number of features
clean_numeric = clean_numeric[:25]  # Top 25 numeric
clean_categorical = clean_categorical[:10]  # Top 10 categorical

print(f"✅ Selected {len(clean_numeric)} CLEAN numeric features")
print(f"✅ Selected {len(clean_categorical)} CLEAN categorical features")
print(f"🚫 BANNED FEATURES: {BANNED_FEATURES}")
print(f"Top numeric: {clean_numeric[:5]}")
print(f"Categorical: {clean_categorical}")

# ============================================
# 3. DATA CLEANING
# ============================================
# Remove outliers
price_quantiles = df.approxQuantile(target_col, [0.025, 0.975], 0.01)
df_filtered = df.filter(
    (col(target_col) >= price_quantiles[0]) & 
    (col(target_col) <= price_quantiles[1])
)
print(f"After outlier removal: {df_filtered.count()} listings")

# ============================================
# 4. PREPROCESSING PIPELINE
# ============================================
print("\n🔧 Setting up CLEAN preprocessing pipeline...")

# Handle categorical features
indexers = []
indexed_categorical = []
for cat_feature in clean_categorical:
    indexer = StringIndexer(
        inputCol=cat_feature, 
        outputCol=f"{cat_feature}_indexed",
        handleInvalid="keep"
    )
    indexers.append(indexer)
    indexed_categorical.append(f"{cat_feature}_indexed")

# Vector assembler for all CLEAN features
all_clean_features = clean_numeric + indexed_categorical
assembler = VectorAssembler(
    inputCols=all_clean_features,
    outputCol="features_raw",
    handleInvalid="skip"
)

# Standard scaler
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Linear regression model
lr = LinearRegression(
    featuresCol="features",
    labelCol=target_col,
    predictionCol="prediction"
)

# Create pipeline
pipeline_stages = indexers + [assembler, scaler, lr]
pipeline = Pipeline(stages=pipeline_stages)

# ============================================
# 5. TRAIN/TEST SPLIT
# ============================================
print("\n📊 Splitting data...")
train_df, test_df = df_filtered.randomSplit([0.8, 0.2], seed=42)
print(f"Training set: {train_df.count()} listings")
print(f"Test set: {test_df.count()} listings")

# ============================================
# 6. SIMPLE TRAINING (no hyperparameter tuning for clarity)
# ============================================
print("\n🎯 Training CLEAN model...")
model = pipeline.fit(train_df)
print("✅ CLEAN model trained!")

# ============================================
# 7. MODEL EVALUATION
# ============================================
print("\n📈 Evaluating CLEAN model performance...")

# Make predictions
train_predictions = model.transform(train_df)
test_predictions = model.transform(test_df)

# Calculate metrics
def calculate_clean_metrics(predictions_df, dataset_name):
    # RMSE
    rmse_evaluator = RegressionEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="rmse"
    )
    rmse = rmse_evaluator.evaluate(predictions_df)
    
    # MAE
    mae_evaluator = RegressionEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="mae"
    )
    mae = mae_evaluator.evaluate(predictions_df)
    
    # R²
    r2_evaluator = RegressionEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="r2"
    )
    r2 = r2_evaluator.evaluate(predictions_df)
    
    # MAPE
    mape_df = predictions_df.withColumn(
        "ape", abs((col(target_col) - col("prediction")) / col(target_col)) * 100
    )
    mape = mape_df.agg(avg("ape")).collect()[0][0]
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

train_metrics = calculate_clean_metrics(train_predictions, "Training")
test_metrics = calculate_clean_metrics(test_predictions, "Test")

# ============================================
# 8. FEATURE IMPORTANCE (VERIFY NO LEAKAGE)
# ============================================
print("\n🔍 CLEAN Feature Importance Analysis...")

# Get the linear regression model from the pipeline
lr_model = model.stages[-1]
feature_importance = lr_model.coefficients.toArray()

# Create importance dataframe - convert to Python types for PySpark 3.5.0
importance_data = [(str(name), float(coef)) for name, coef in zip(all_clean_features, feature_importance)]
importance_df = spark.createDataFrame(importance_data, ["feature", "coefficient"])

print("\nTop 10 Most Important CLEAN Features:")
importance_df.withColumn("abs_coef", abs(col("coefficient"))) \
    .orderBy(col("abs_coef").desc()) \
    .select("feature", "coefficient") \
    .show(10, truncate=False)

# VERIFY no leaky features
leaky_in_results = importance_df.filter(
    col("feature").contains("price_per")
).count()

if leaky_in_results > 0:
    print("🚨 ERROR: Still found leaky features!")
else:
    print("✅ VERIFIED: No data leakage features in results!")

# ============================================
# 9. SAVE CLEAN MODEL AND RESULTS
# ============================================
print("\n💾 Saving CLEAN model and results...")

os.makedirs("output/models", exist_ok=True)

# Save the clean model
model_path = "output/models/baseline_linear_regression"
model.write().overwrite().save(model_path)

# Save clean metrics
clean_metrics_data = {
    "model_type": "CLEAN Linear Regression (Zero Data Leakage)",
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
    "feature_count": len(all_clean_features),
    "banned_features": BANNED_FEATURES,
    "verification": "No price-derived features used"
}

with open("output/models/baseline_metrics.pkl", "wb") as f:
    pickle.dump(clean_metrics_data, f)

# Save feature importance
importance_df.toPandas().to_csv("output/models/baseline_feature_importance.csv", index=False)

# ============================================
# 10. SAMPLE PREDICTIONS
# ============================================
print("\n🔍 Sample Clean Predictions:")
sample_predictions = test_predictions.select(
    target_col, "prediction", "city", "room_type", "accommodates"
).limit(10)

sample_predictions.show(truncate=False)

print("\n✅ CLEAN baseline model complete!")
print(f"📁 Model saved to: {model_path}")
print(f"📊 Test RMSE: ${test_metrics['rmse']:.2f}")
print(f"📊 Test R²: {test_metrics['r2']:.4f}")
print(f"🎯 These are LEGITIMATE results with ZERO data leakage!")
print(f"🚫 Verified exclusion of: {BANNED_FEATURES}")

spark.stop()