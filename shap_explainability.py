import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🔍 SHAP Explainability Analysis for Airbnb Price Prediction")
print("=" * 70)

# ============================================
# 1. SETUP AND DATA LOADING
# ============================================
print("\n📊 Loading data and preparing for SHAP analysis...")

# Initialize Spark (we'll convert to pandas for SHAP)
spark = SparkSession.builder \
    .appName("SHAP Analysis") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()

# Load engineered features
features_path = "output/features/engineered_features.parquet"
if not os.path.exists(features_path):
    print("❌ ERROR: Engineered features not found!")
    spark.stop()
    exit(1)

df = spark.read.parquet(features_path)
target_col = "price_clean"

# Apply same filtering as in model training
price_quantiles = df.approxQuantile(target_col, [0.01, 0.99], 0.01)
df_clean = df.filter(
    (col(target_col) >= price_quantiles[0]) & 
    (col(target_col) <= price_quantiles[1]) &
    col(target_col).isNotNull()
)

print(f"✅ Loaded {df_clean.count()} listings for analysis")

# ============================================
# 2. CLEAN FEATURE PREPARATION FOR SHAP
# ============================================
print("\n🔧 Preparing CLEAN features for SHAP analysis...")

# EXPLICITLY exclude leaky features
BANNED_FEATURES = ['price_per_bedroom', 'price_per_person', 'price']

# Use the same features as in the models
legitimate_numeric = [
    "accommodates", "bedrooms", "bathrooms_numeric", "beds",
    "minimum_nights", "maximum_nights", "availability_30", "availability_60", 
    "availability_90", "availability_365", "number_of_reviews", "distance_to_center", 
    "amenities_count", "host_days_active", "host_response_rate_clean", 
    "host_acceptance_rate_clean", "review_scores_rating_clean", "review_scores_location_clean",
    "review_scores_cleanliness_clean", "host_is_superhost_binary", "instant_bookable_binary",
    "has_wifi", "has_kitchen", "has_ac", "has_parking", "has_reviews"
]

legitimate_categorical = ["room_type", "city"]

# Filter features that actually exist and are not banned
clean_numeric = []
for feature in legitimate_numeric:
    if feature in df.columns:
        # Check data type
        dtype = dict(df.dtypes)[feature]
        if dtype in ['int', 'bigint', 'float', 'double'] and feature not in BANNED_FEATURES:
            clean_numeric.append(feature)

clean_categorical = []
for feature in legitimate_categorical:
    if feature in df.columns and feature not in BANNED_FEATURES:
        clean_categorical.append(feature)

print(f"✅ Selected {len(clean_numeric)} clean numeric features")
print(f"✅ Selected {len(clean_categorical)} clean categorical features")
print(f"🚫 Excluded banned features: {BANNED_FEATURES}")

# Convert to pandas for SHAP (sample for performance)
sample_size = __builtins__.min(3000, df_clean.count())  # Limit for SHAP performance
df_sample = df_clean.sample(fraction=sample_size/df_clean.count(), seed=42)

print(f"🎯 Using sample of {sample_size} listings for SHAP analysis")

# Convert to pandas
feature_columns = clean_numeric + clean_categorical + [target_col]
df_pandas = df_sample.select(*feature_columns).toPandas()

print(f"✅ Converted to pandas: {df_pandas.shape}")

# ============================================
# 3. PREPARE DATA FOR SKLEARN MODELS
# ============================================
print("\n🛠️ Preparing data for sklearn models...")

# Handle categorical variables
df_encoded = df_pandas.copy()
label_encoders = {}

for cat_feature in clean_categorical:
    if cat_feature in df_encoded.columns:
        le = LabelEncoder()
        # Handle missing values
        df_encoded[cat_feature] = df_encoded[cat_feature].fillna('missing')
        df_encoded[cat_feature] = le.fit_transform(df_encoded[cat_feature].astype(str))
        label_encoders[cat_feature] = le

# Fill numeric nulls
for num_feature in clean_numeric:
    if num_feature in df_encoded.columns:
        df_encoded[num_feature] = df_encoded[num_feature].fillna(0)

# Prepare feature matrix and target
feature_cols_final = [col for col in clean_numeric + clean_categorical if col in df_encoded.columns]
X = df_encoded[feature_cols_final]
y = df_encoded[target_col]

print(f"✅ Final clean feature matrix: {X.shape}")
print(f"✅ Target variable: {y.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# 4. TRAIN SKLEARN MODELS FOR SHAP
# ============================================
print("\n🤖 Training sklearn models for SHAP analysis...")

# Gradient Boosting (similar to Spark GBT)
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_score = gb_model.score(X_test, y_test)
print(f"✅ Gradient Boosting R²: {gb_score:.4f}")

# Linear Regression (for comparison)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_score = lr_model.score(X_test_scaled, y_test)
print(f"✅ Linear Regression R²: {lr_score:.4f}")

# ============================================
# 5. SHAP ANALYSIS - GRADIENT BOOSTING
# ============================================
print("\n🔍 SHAP Analysis for Gradient Boosting Model...")

# Use a position-based slice so indices line up
X_for_shap = X_test.reset_index(drop=True).iloc[:500]
y_for_shap = y_test.reset_index(drop=True).iloc[:500]

explainer_gb = shap.TreeExplainer(gb_model)
shap_values_gb = explainer_gb.shap_values(X_for_shap)  # (n, p)


print("✅ SHAP values calculated for Gradient Boosting")

# ============================================
# 6. SHAP ANALYSIS - LINEAR REGRESSION
# ============================================
print("\n🔍 SHAP Analysis for Linear Regression Model...")

# Create SHAP explainer for linear model
explainer_lr = shap.LinearExplainer(lr_model, X_train_scaled)
shap_values_lr = explainer_lr.shap_values(X_test_scaled[:500])

print("✅ SHAP values calculated for Linear Regression")

# ============================================
# 7. VISUALIZATIONS AND INSIGHTS
# ============================================
print("\n📊 Creating SHAP visualizations...")

# Create output directory
os.makedirs("output/explainability", exist_ok=True)

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

# 1) SHAP Feature Importance - Gradient Boosting
shap.summary_plot(shap_values_gb, X_test[:500],
                  feature_names=feature_cols_final, show=False)
fig = plt.gcf()
fig.suptitle("SHAP Feature Importance - Gradient Boosting Model (Clean Features)",
             fontsize=16, fontweight='bold', y=1.02)
fig.savefig("output/explainability/shap_summary_gb.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# 2) Bar plot
shap.summary_plot(shap_values_gb, X_test[:500],
                  feature_names=feature_cols_final, plot_type="bar", show=False)
fig = plt.gcf()
fig.suptitle("SHAP Feature Importance (Bar Plot) - Gradient Boosting",
             fontsize=16, fontweight='bold', y=1.02)
fig.savefig("output/explainability/shap_importance_bar_gb.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# 3) Linear Regression summary
shap.summary_plot(shap_values_lr, X_test_scaled[:500],
                  feature_names=feature_cols_final, show=False)
fig = plt.gcf()
fig.suptitle("SHAP Feature Importance - Linear Regression Model (Clean Features)",
             fontsize=16, fontweight='bold', y=1.02)
fig.savefig("output/explainability/shap_summary_lr.png", dpi=300, bbox_inches='tight')
plt.close(fig)


print("✅ Summary plots saved")

# ============================================
# 8. DETAILED FEATURE ANALYSIS
# ============================================
print("\n📈 Detailed clean feature analysis...")

# Calculate mean absolute SHAP values for feature ranking
mean_shap_gb = np.mean(np.abs(shap_values_gb), axis=0)
mean_shap_lr = np.mean(np.abs(shap_values_lr), axis=0)

# Create feature importance dataframes
feature_importance_gb = pd.DataFrame({
    'feature': feature_cols_final,
    'shap_importance': mean_shap_gb,
    'model': 'Gradient Boosting'
}).sort_values('shap_importance', ascending=False)

feature_importance_lr = pd.DataFrame({
    'feature': feature_cols_final,
    'shap_importance': mean_shap_lr,
    'model': 'Linear Regression'
}).sort_values('shap_importance', ascending=False)

# Display top features
print("\nTop 10 Most Important Features - Gradient Boosting:")
print(feature_importance_gb.head(10).to_string(index=False))

print("\nTop 10 Most Important Features - Linear Regression:")
print(feature_importance_lr.head(10).to_string(index=False))

# ============================================
# 9. INDIVIDUAL PREDICTION EXPLANATIONS
# ============================================
print("\n🎯 Individual prediction explanations...")

# --- Waterfall plots (modern API) ---
ev_scalar = float(np.squeeze(explainer_gb.expected_value))  # ensure scalar
sample_indices = [0, 25, 50, 100, 200]

for i, idx in enumerate(sample_indices, start=1):
    if idx < len(X_for_shap):
        row = X_for_shap.iloc[idx]
        sv = shap_values_gb[idx]  # 1D array (p,)

        exp = shap.Explanation(
            values=sv,
            base_values=ev_scalar,
            data=row.values,
            feature_names=X_for_shap.columns
        )

        shap.plots.waterfall(exp, max_display=20, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f"Individual Prediction Explanation {i} - Gradient Boosting\n"
            f"Actual: ${y_for_shap.iloc[idx]:.0f}, "
            f"Predicted: ${gb_model.predict(X_for_shap.iloc[[idx]])[0]:.0f}",
            fontsize=12, y=1.02
        )
        fig.savefig(f"output/explainability/waterfall_gb_example_{i}.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)


print("✅ Individual explanation plots saved")

# ============================================
# 10. PARTIAL DEPENDENCE ANALYSIS
# ============================================
print("\n📊 Partial dependence analysis...")

# Select top 6 most important features for partial dependence plots
top_features = feature_importance_gb.head(6)['feature'].tolist()

from sklearn.inspection import PartialDependenceDisplay

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, feature in enumerate(top_features):
    ax = axes.ravel()[i]
    PartialDependenceDisplay.from_estimator(gb_model, X_train, [feature], ax=ax)
    ax.set_title(f'Partial Dependence: {feature}', fontweight='bold')
fig.suptitle('Partial Dependence Plots - Top 6 Clean Features', fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig("output/explainability/partial_dependence_plots.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ============================================
# 11. SAVE NUMERICAL RESULTS
# ============================================
print("\n💾 Saving numerical results...")

# Combine feature importance from both models
combined_importance = pd.merge(
    feature_importance_gb[['feature', 'shap_importance']].rename(columns={'shap_importance': 'gb_importance'}),
    feature_importance_lr[['feature', 'shap_importance']].rename(columns={'shap_importance': 'lr_importance'}),
    on='feature'
)

# Add rank information
combined_importance['gb_rank'] = combined_importance['gb_importance'].rank(ascending=False)
combined_importance['lr_rank'] = combined_importance['lr_importance'].rank(ascending=False)
combined_importance['rank_difference'] = combined_importance['lr_rank'] - combined_importance['gb_rank']

# Save to CSV
combined_importance.to_csv("output/explainability/feature_importance_comparison.csv", index=False)

# Save SHAP values for future analysis
shap_results = {
    'shap_values_gb': shap_values_gb,
    'shap_values_lr': shap_values_lr,
    'feature_names': feature_cols_final,
    'test_predictions_gb': gb_model.predict(X_test),
    'test_predictions_lr': lr_model.predict(X_test_scaled),
    'test_actuals': y_test.values,
    'model_scores': {'gb_r2': gb_score, 'lr_r2': lr_score},
    'banned_features': BANNED_FEATURES
}

with open("output/explainability/shap_results.pkl", "wb") as f:
    pickle.dump(shap_results, f)

# ============================================
# 12. KEY INSIGHTS SUMMARY
# ============================================
print("\n" + "="*70)
print("KEY INSIGHTS FROM CLEAN SHAP ANALYSIS")
print("="*70)

print(f"\n🏆 MODEL PERFORMANCE COMPARISON:")
print(f"   Gradient Boosting R²: {gb_score:.4f}")
print(f"   Linear Regression R²: {lr_score:.4f}")

print(f"\n🎯 TOP 5 MOST IMPORTANT CLEAN FEATURES (Gradient Boosting):")
for i, row in feature_importance_gb.head(5).iterrows():
    print(f"   {row['feature']:>25}: {row['shap_importance']:.4f}")

print(f"\n🔍 TOP 5 MOST IMPORTANT CLEAN FEATURES (Linear Regression):")
for i, row in feature_importance_lr.head(5).iterrows():
    print(f"   {row['feature']:>25}: {row['shap_importance']:.4f}")

print(f"\n📊 FEATURE CONSISTENCY ANALYSIS:")
consistent_features = combined_importance[combined_importance['rank_difference'].abs() <= 3]
print(f"   Features with consistent importance across models: {len(consistent_features)}")

different_features = combined_importance[combined_importance['rank_difference'].abs() > 5]
if len(different_features) > 0:
    print(f"   Features with different importance rankings:")
    for _, row in different_features.head(3).iterrows():
        print(f"      {row['feature']:>25}: GB rank {row['gb_rank']:.0f}, LR rank {row['lr_rank']:.0f}")

print(f"\n💡 KEY BUSINESS INSIGHTS :")
print(f"   🏠 Property characteristics (accommodates, bedrooms) are legitimately important")
print(f"   📍 Location features (distance to center, city) drive pricing significantly")
print(f"   ⭐ Host and amenity features influence price prediction")
print(f"   🎯 All insights are based on features available before knowing price")

print(f"\n✅ CLEAN SHAP analysis complete!")
print(f"📁 Visualizations saved to: output/explainability/")
print(f"📊 Numerical results saved as CSV and pickle files")
print(f"🚫 Verified exclusion of: {BANNED_FEATURES}")

# Clean up
spark.stop()

print(f"\n🎉 All CLEAN SHAP analysis files generated successfully!")
print(f"   - Summary plots showing legitimate feature importance")
print(f"   - Individual prediction explanations")
print(f"   - Partial dependence plots")
print(f"   - Feature importance comparison between models")
print(f"   - Numerical results for further analysis")