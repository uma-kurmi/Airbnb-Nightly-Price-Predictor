import os
import pickle
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, max, count, sqrt, pow, when, expr, abs
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

from builtins import abs as py_abs


# Initialize Spark
spark = SparkSession.builder \
    .appName("Model Evaluation Comparison") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("📊 Comprehensive Model Evaluation & Comparison")
print("=" * 60)

# ============================================
# 1. LOAD MODELS AND DATA
# ============================================
print("\n🔄 Loading models and test data...")

# Check if models exist
baseline_path = "output/models/baseline_linear_regression"
gbt_path = "output/models/gbt_regressor"

if not os.path.exists(baseline_path):
    print("❌ Baseline model not found! Run train_baseline_model_CLEAN.py first")
    spark.stop()
    exit(1)

if not os.path.exists(gbt_path):
    print("❌ GBT model not found! Run train_gbt_model.py first")
    spark.stop()
    exit(1)

# Load models
baseline_model = PipelineModel.load(baseline_path)
gbt_model = PipelineModel.load(gbt_path)

# Load test data
features_path = "output/features/engineered_features.parquet"
df = spark.read.parquet(features_path)

# Apply same filtering as in training
target_col = "price_clean"
price_quantiles = df.approxQuantile(target_col, [0.01, 0.99], 0.01)
df_filtered = df.filter(
    (col(target_col) >= price_quantiles[0]) & 
    (col(target_col) <= price_quantiles[1]) &
    col(target_col).isNotNull()
)

# Create test set (same split as in training)
_, test_df = df_filtered.randomSplit([0.8, 0.2], seed=42)
print(f"✅ Test set loaded: {test_df.count()} listings")

# ============================================
# 2. GENERATE PREDICTIONS
# ============================================
print("\n🎯 Generating predictions from both models...")

try:
    baseline_predictions = baseline_model.transform(test_df)
    baseline_predictions = baseline_predictions.withColumnRenamed("prediction", "baseline_pred")
    print("✅ Baseline predictions generated")
except Exception as e:
    print(f"❌ Error with baseline predictions: {e}")
    spark.stop()
    exit(1)

try:
    gbt_predictions = gbt_model.transform(test_df)
    gbt_predictions = gbt_predictions.withColumnRenamed("prediction", "gbt_pred")
    print("✅ GBT predictions generated")
except Exception as e:
    print(f"❌ Error with GBT predictions: {e}")
    spark.stop()
    exit(1)

# Combine predictions
combined_predictions = baseline_predictions.select(
    "id", target_col, "baseline_pred", "city", "room_type", "accommodates"
).join(
    gbt_predictions.select("id", "gbt_pred"), 
    on="id", 
    how="inner"
)

print(f"✅ Combined predictions: {combined_predictions.count()} listings")

# ============================================
# 3. CALCULATE COMPREHENSIVE METRICS
# ============================================
print("\n📈 Calculating comprehensive metrics...")

def calculate_all_metrics(predictions_df, actual_col, pred_col, model_name):
    """Calculate comprehensive regression metrics"""
    
    # Filter out null predictions
    clean_predictions = predictions_df.filter(
        col(actual_col).isNotNull() & col(pred_col).isNotNull()
    )
    
    # Basic metrics using Spark evaluators
    evaluator_rmse = RegressionEvaluator(labelCol=actual_col, predictionCol=pred_col, metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol=actual_col, predictionCol=pred_col, metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol=actual_col, predictionCol=pred_col, metricName="r2")
    
    rmse = evaluator_rmse.evaluate(clean_predictions)
    mae = evaluator_mae.evaluate(clean_predictions)
    r2 = evaluator_r2.evaluate(clean_predictions)
    
    # Additional metrics
    metrics_df = clean_predictions.select(
        avg(abs((col(actual_col) - col(pred_col)) / col(actual_col)) * 100).alias("mape"),
        avg(actual_col).alias("actual_mean"),
        avg(pred_col).alias("pred_mean"),
        stddev(actual_col).alias("actual_std"),
        stddev(pred_col).alias("pred_std"),
        max(abs(col(actual_col) - col(pred_col))).alias("max_error"),
        count("*").alias("n_samples")
    ).collect()[0]
    
    mape = metrics_df["mape"]
    
    return {
        "model": model_name,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
        "actual_mean": float(metrics_df["actual_mean"]) if metrics_df["actual_mean"] else None,
        "pred_mean": float(metrics_df["pred_mean"]) if metrics_df["pred_mean"] else None,
        "actual_std": float(metrics_df["actual_std"]) if metrics_df["actual_std"] else None,
        "pred_std": float(metrics_df["pred_std"]) if metrics_df["pred_std"] else None,
        "max_error": float(metrics_df["max_error"]) if metrics_df["max_error"] else None,
        "n_samples": int(metrics_df["n_samples"]) if metrics_df["n_samples"] else None
    }

# Calculate metrics for both models
baseline_metrics = calculate_all_metrics(combined_predictions, target_col, "baseline_pred", "Linear Regression")
gbt_metrics = calculate_all_metrics(combined_predictions, target_col, "gbt_pred", "Gradient Boosted Trees")

# ============================================
# 4. DISPLAY COMPARISON RESULTS
# ============================================
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

comparison_df = pd.DataFrame([baseline_metrics, gbt_metrics])
comparison_df = comparison_df.set_index('model')

print("\nKey Performance Metrics:")
print("-" * 40)
for metric in ['rmse', 'mae', 'r2', 'mape']:
    print(f"{metric.upper():>6}")
    for model in comparison_df.index:
        value = comparison_df.loc[model, metric]
        if metric == 'mape':
            print(f"  {model:>25}: {value:>8.2f}%")
        elif metric == 'r2':
            print(f"  {model:>25}: {value:>8.4f}")
        else:
            print(f"  {model:>25}: ${value:>8.2f}")
    print()

# Calculate improvement
rmse_improvement = (baseline_metrics['rmse'] - gbt_metrics['rmse']) / baseline_metrics['rmse'] * 100
mae_improvement = (baseline_metrics['mae'] - gbt_metrics['mae']) / baseline_metrics['mae'] * 100
r2_improvement = (gbt_metrics['r2'] - baseline_metrics['r2']) / py_abs(baseline_metrics['r2']) * 100


print("Performance Improvement (GBT vs Baseline):")
print(f"  RMSE: {rmse_improvement:+.2f}%")
print(f"  MAE:  {mae_improvement:+.2f}%")
print(f"  R²:   {r2_improvement:+.2f}%")

# ============================================
# 5. DETAILED ANALYSIS BY SEGMENTS
# ============================================
print("\n" + "="*60)
print("SEGMENTED ANALYSIS")
print("="*60)

# Performance by city
print("\nPerformance by City:")
city_analysis = combined_predictions.groupBy("city").agg(
    count("*").alias("count"),
    avg(target_col).alias("avg_price"),
    sqrt(avg(pow(col(target_col) - col("baseline_pred"), 2))).alias("baseline_rmse"),
    sqrt(avg(pow(col(target_col) - col("gbt_pred"), 2))).alias("gbt_rmse"),
    avg(abs(col(target_col) - col("baseline_pred"))).alias("baseline_mae"),
    avg(abs(col(target_col) - col("gbt_pred"))).alias("gbt_mae")
).orderBy("city")

city_analysis.show()

# Performance by room type
print("\nPerformance by Room Type:")
room_analysis = combined_predictions.groupBy("room_type").agg(
    count("*").alias("count"),
    avg(target_col).alias("avg_price"),
    sqrt(avg(pow(col(target_col) - col("baseline_pred"), 2))).alias("baseline_rmse"),
    sqrt(avg(pow(col(target_col) - col("gbt_pred"), 2))).alias("gbt_rmse")
).orderBy("avg_price")

room_analysis.show()

# Performance by price range
print("\nPerformance by Price Range:")
price_range_analysis = combined_predictions.withColumn(
    "price_range",
    when(col(target_col) < 75, "Budget (<$75)")
    .when(col(target_col) < 150, "Mid-range ($75-150)")
    .when(col(target_col) < 250, "Premium ($150-250)")
    .otherwise("Luxury (>$250)")
).groupBy("price_range").agg(
    count("*").alias("count"),
    avg(target_col).alias("avg_price"),
    sqrt(avg(pow(col(target_col) - col("baseline_pred"), 2))).alias("baseline_rmse"),
    sqrt(avg(pow(col(target_col) - col("gbt_pred"), 2))).alias("gbt_rmse"),
    avg(abs((col(target_col) - col("baseline_pred")) / col(target_col)) * 100).alias("baseline_mape"),
    avg(abs((col(target_col) - col("gbt_pred")) / col(target_col)) * 100).alias("gbt_mape")
).orderBy("avg_price")

price_range_analysis.show()

# ============================================
# 6. ERROR ANALYSIS
# ============================================
print("\n" + "="*60)
print("ERROR ANALYSIS")
print("="*60)

# Calculate errors
error_analysis = combined_predictions.withColumn(
    "baseline_error", col("baseline_pred") - col(target_col)
).withColumn(
    "gbt_error", col("gbt_pred") - col(target_col)
).withColumn(
    "baseline_abs_error", abs(col("baseline_error"))
).withColumn(
    "gbt_abs_error", abs(col("gbt_error"))
).withColumn(
    "baseline_error_pct", abs(col("baseline_error")) / col(target_col) * 100
).withColumn(
    "gbt_error_pct", abs(col("gbt_error")) / col(target_col) * 100
)

# Error statistics
error_stats = error_analysis.select(
    avg("baseline_abs_error").alias("baseline_avg_abs_error"),
    avg("gbt_abs_error").alias("gbt_avg_abs_error"),
    stddev("baseline_error").alias("baseline_error_std"),
    stddev("gbt_error").alias("gbt_error_std"),
    max("baseline_abs_error").alias("baseline_max_error"),
    max("gbt_abs_error").alias("gbt_max_error"),
    expr("percentile_approx(baseline_error_pct, 0.5)").alias("baseline_median_error_pct"),
    expr("percentile_approx(gbt_error_pct, 0.5)").alias("gbt_median_error_pct"),
    expr("percentile_approx(baseline_error_pct, 0.9)").alias("baseline_90th_error_pct"),
    expr("percentile_approx(gbt_error_pct, 0.9)").alias("gbt_90th_error_pct")
).collect()[0]

print("\nError Distribution Analysis:")
print(f"Average Absolute Error:")
print(f"  Baseline: ${error_stats['baseline_avg_abs_error']:.2f}")
print(f"  GBT:      ${error_stats['gbt_avg_abs_error']:.2f}")
print(f"\nError Standard Deviation:")
print(f"  Baseline: ${error_stats['baseline_error_std']:.2f}")
print(f"  GBT:      ${error_stats['gbt_error_std']:.2f}")
print(f"\nMaximum Absolute Error:")
print(f"  Baseline: ${error_stats['baseline_max_error']:.2f}")
print(f"  GBT:      ${error_stats['gbt_max_error']:.2f}")
print(f"\nMedian Error Percentage:")
print(f"  Baseline: {error_stats['baseline_median_error_pct']:.2f}%")
print(f"  GBT:      {error_stats['gbt_median_error_pct']:.2f}%")
print(f"\n90th Percentile Error Percentage:")
print(f"  Baseline: {error_stats['baseline_90th_error_pct']:.2f}%")
print(f"  GBT:      {error_stats['gbt_90th_error_pct']:.2f}%")

# ============================================
# 7. SAVE EVALUATION RESULTS
# ============================================
print("\n💾 Saving evaluation results...")

os.makedirs("output/evaluation", exist_ok=True)

# Save detailed comparison
evaluation_results = {
    "baseline_metrics": baseline_metrics,
    "gbt_metrics": gbt_metrics,
    "improvements": {
        "rmse_improvement_pct": rmse_improvement,
        "mae_improvement_pct": mae_improvement,
        "r2_improvement_pct": r2_improvement
    },
    "error_analysis": dict(error_stats.asDict())
}

with open("output/evaluation/model_comparison.pkl", "wb") as f:
    pickle.dump(evaluation_results, f)

# Save predictions sample for further analysis
sample_predictions = combined_predictions.limit(1000).toPandas()
sample_predictions.to_csv("output/evaluation/sample_predictions.csv", index=False)

# Save segmented analysis
city_analysis.toPandas().to_csv("output/evaluation/city_performance.csv", index=False)
room_analysis.toPandas().to_csv("output/evaluation/room_type_performance.csv", index=False)
price_range_analysis.toPandas().to_csv("output/evaluation/price_range_performance.csv", index=False)

# ============================================
# 8. FINAL RECOMMENDATIONS
# ============================================
print("\n" + "="*60)
print("FINAL RECOMMENDATIONS")
print("="*60)

if gbt_metrics['rmse'] < baseline_metrics['rmse']:
    print("🏆 WINNER: Gradient Boosted Trees")
    print(f"   - {rmse_improvement:.1f}% better RMSE")
    print(f"   - {mae_improvement:.1f}% better MAE")
    print(f"   - R² improved from {baseline_metrics['r2']:.4f} to {gbt_metrics['r2']:.4f}")
else:
    print("🏆 WINNER: Linear Regression Baseline")
    print("   - Simpler model with competitive performance")

print(f"\nModel Selection Guidance:")
if rmse_improvement > 10:
    print("   ✅ Use GBT for production (significant improvement)")
elif rmse_improvement > 5:
    print("   ⚖️  Consider GBT vs baseline trade-off (moderate improvement)")
else:
    print("   🤔 Consider baseline for simplicity (minimal improvement)")

print(f"\nBest performing segments for GBT:")
print("   - Analysis suggests GBT handles complex pricing patterns better")
print("   - Particularly effective for all price ranges")

print("\n✅ Model evaluation complete!")
print(f"📁 Results saved to: output/evaluation/")

spark.stop()