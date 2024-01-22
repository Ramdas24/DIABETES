import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import time  # Import the time module
import re

# Load the imputed diabetes dataset
diabetes_data = pd.read_csv('cleaned_diabetic_data.csv')

# Define the target variable (e.g., 'readmitted')
target_variable = diabetes_data['readmitted']

# Encode the target variable into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(target_variable)

# Remove the target variable from the dataset
diabetes_data.drop(columns=['readmitted'], inplace=True)

# Split the dataset into features (X) and target (y)
X = diabetes_data
y = y_encoded

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
# Rename columns to replace invalid characters and ensure valid feature names
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<',' ',','))) else col for col in X.columns.values]
print(X)
# Get the data types of all columns
data_types = diabetes_data.dtypes


# Preprocess the 'age' column
diabetes_data['age'] = diabetes_data['age'].str.replace('[', '').str.replace(')', '')

# Preprocess the 'weight' column
diabetes_data['weight'] = diabetes_data['weight'].str.replace('[', '').str.replace(')', '').str.replace('>','')

# Preprocess the 'diag_3' column to remove 'V' and 'E' prefixes
diabetes_data['diag_3'] = diabetes_data['diag_3'].str.replace('V', '').str.replace('E', '')

# Separate columns into categorical and numerical based on data types
categorical_columns_list = data_types[data_types == 'object'].index.tolist()
numerical_columns = data_types[data_types.isin(['int64', 'int32', 'float64', 'float32'])].index.tolist()

# Encode categorical variables using OneHotEncoder
categorical_columns = categorical_columns_list[:-1]  # Exclude the last column ('readmitted')
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = encoder.fit_transform(X[categorical_columns])

# Combine encoded categorical features with numerical features
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names(categorical_columns))
X = pd.concat([X_encoded_df, X[numerical_columns]], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
random_forest = RandomForestClassifier(random_state=42)
lgb_model = lgb.LGBMClassifier(random_state=42)

# Feature names (assuming column names are feature names)
feature_names = X.columns

#Random Forest
start_time = time.time()
random_forest.fit(X_train, y_train)
random_forest_time = time.time() - start_time
start_time = time.time()
print(f"Random Forest Training Time: {random_forest_time} seconds")
random_forest_importances = random_forest.feature_importances_
print("Random Forest Feature Importance:")
print(random_forest_importances)
random_forest_importance_time = time.time() - start_time
print(f"Random Forest Feature Importance Evaluation Time: {random_forest_importance_time} seconds")
plt.figure(figsize=(8, 6))
plt.barh(feature_names, random_forest_importances)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

#LGBM
start_time = time.time()
lgb_model.fit(X_train, y_train)
lgb_time = time.time() - start_time
print(f"LightGBM Training Time: {lgb_time} seconds")
start_time = time.time()
lgb_importances = lgb_model.feature_importances_
print("Light GBM Feature Importance:")
print(lgb_importances)
lgb_importance_time = time.time() - start_time
print(f"LightGBM Feature Importance Evaluation Time: {lgb_importance_time} seconds")
plt.figure(figsize=(8, 6))
plt.barh(feature_names, lgb_importances)
plt.title("LightGBM Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
