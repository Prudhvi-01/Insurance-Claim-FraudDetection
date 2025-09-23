Insurance Claims Fraud Detection Project
1. Project Overview

Objective
The objective of this project is to build a machine learning system to detect fraudulent insurance claims.
We combine:

IsolationForest (inverted logic) for anomaly detection.

XGBoost (softprob classifier) for supervised fraud classification.

This hybrid approach improves fraud detection accuracy compared to using a single model.

Dataset
We worked with an augmented insurance fraud dataset containing labeled fraud and non-fraud claims. The dataset includes:

Demographic details (age, occupation, nominee relation)

Policy details (sum assured, premium, tenure)

Claim attributes (claim amount, reason, hospital, etc.)

Programming Language

Python (3.8+)

Tools & Libraries

Data Handling: Pandas, NumPy

Modeling: Scikit-learn, XGBoost, Imbalanced-learn

Visualization: Matplotlib, Seaborn

UI: Streamlit

2. Data Acquisition

Placed raw CSVs into the data/ folder.

Loaded dataset into Pandas for cleaning, preprocessing, and feature engineering.

3. Data Exploration and Preprocessing
3.1 Exploratory Data Analysis (EDA)

Checked shape and data types.

Verified missing values and duplicates.

Visualized fraud vs. non-fraud distribution (highly imbalanced).

Used bar plots, histograms, box plots, and correlation heatmaps to explore relationships between features.

3.2 Data Cleaning

Handled missing values with appropriate strategies.

Detected and treated outliers using IQR method.

Replaced unknown categories with "unknown" class instead of dropping them.

3.3 Feature Engineering

Created ratio-based features (e.g., Premium-to-SumAssured).

Extracted useful info from dates (month, quarter).

Encoded categorical variables using Label Encoding and One-Hot Encoding.

Standardized numerical columns with StandardScaler.

3.4 Feature Selection

Removed columns with low predictive power.

Selected features that contributed most to model performance.

4. Model Building
4.1 Split Data

Data divided into train (80%) and test (20%).

4.2 Train Models

IsolationForest (inverted) → trained on known fraud claims (fraud = inliers).

XGBoost (softprob) → supervised classifier for fraud vs. non-fraud.

4.3 Evaluate Models

Metrics used: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

XGBoost achieved strong classification results.

Hybrid approach (IsolationForest + XGBoost) improved overall detection rate.

5. Model Tuning

Used RandomizedSearchCV and GridSearchCV for hyperparameter tuning.

Optimized XGBoost parameters: learning rate, max depth, number of estimators.

Tuned IsolationForest contamination and max samples.
