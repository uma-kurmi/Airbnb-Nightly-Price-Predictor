# Airbnb-Nightly-Price-Predictor
A machine learning project that predicts the nightly price of Airbnb listings based on listing characteristics such as location, property type, room type, amenities, and host attributes.
This project demonstrates a complete end-to-end data science pipeline, including data cleaning, feature engineering, model training, evaluation, explainability, and deployment with a Streamlit dashboard.

# Problem Statement
Airbnb hosts need to determine the right nightly price for their listings. Setting prices too high may reduce bookings, while setting prices too low may reduce potential revenue.
The objective of this project is to build a machine learning model that predicts a reasonable nightly price for Airbnb listings based on various listing features.

# Project Workflow
The project follows a structured machine learning pipeline.

1. Data Cleaning
The raw Airbnb listing dataset is cleaned by:
Removing invalid or missing prices
Handling missing values
Standardizing column formats
Filtering unrealistic listings
Script used:
clean_listings_script.py

2. Neighborhood Feature Construction
Neighborhood-level information is extracted to enrich the dataset.
This helps the model capture location-based pricing patterns.
Script used:
build_neighborhood_lookup.py

3. Feature Engineering
Additional features are created to improve model performance, such as:
Numerical transformations
Encoded categorical variables
Aggregated listing features
Derived host metrics
Script used:
feature_engineering.py

4. Dataset Creation
All processed features are combined into a single unified dataset used for modeling.
Script used:
create_unified_dataset.py

5. Exploratory Data Analysis
Initial analysis is performed to understand feature distributions and relationships with price.
Script used:
explore_features.py

6. Baseline Model Training
A simple regression model is trained to establish a baseline performance benchmark.
Script used:
train_baseline_model.py

7. Gradient Boosting Model Training
A more advanced model (Gradient Boosting) is trained to improve prediction accuracy.
Script used:
train_gbt_model.py

8. Model Evaluation
Models are evaluated using standard regression metrics:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score
Script used:
model_evaluation.py

9. Model Explainability
SHAP values are used to understand how features influence predictions.
This helps identify the most important factors affecting Airbnb pricing.
Script used:
shap_explainability.py

10. Interactive Dashboard
A Streamlit application allows users to input listing features and receive a predicted nightly price.
Script used:
streamlit_dashboard.py

Run the dashboard with:
streamlit run streamlit_dashboard.py

# Project Structure
Airbnb-Nightly-Price-Predictor

├── README.md

├── requirements.txt

├── clean_listings_script.py

├── build_neighborhood_lookup.py

├── feature_engineering.py

├── create_unified_dataset.py

├── explore_features.py

├── train_baseline_model.py

├── train_gbt_model.py

├── model_evaluation.py

├── shap_explainability.py

└── streamlit_dashboard.py

# Clone the repository:
git clone https://github.com/uma-kurmi/Airbnb-Nightly-Price-Predictor.git
cd Airbnb-Nightly-Price-Predictor

# Install required packages:
pip install -r requirements.txt

# Run the scripts in the following order:

python clean_listings_script.py
python build_neighborhood_lookup.py
python feature_engineering.py
python create_unified_dataset.py
python explore_features.py
python train_baseline_model.py
python train_gbt_model.py
python model_evaluation.py
python shap_explainability.py

Launch the dashboard:
streamlit run streamlit_dashboard.py

# Technologies Used
Python
Pandas
NumPy
Scikit-learn
SHAP
Streamlit
Matplotlib / Seaborn

# Future Improvements
Potential improvements for this project include:
Hyperparameter tuning
Testing additional models (XGBoost, LightGBM)
Adding geospatial features
Building a REST API for predictions
Deploying the Streamlit app online

# License
This project is open source and available under the MIT License.

# Streamlit Dashboard
The project includes an interactive Streamlit dashboard where users can input listing details and receive a predicted Airbnb nightly price.


