import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('cleaned_diabetic_data.csv')

# Encode categorical columns
categorical_cols = ['race', 'gender', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone',
                    'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
                    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
                    'change', 'diabetesMed', 'readmitted']

# Use LabelEncoder to convert categorical columns to numeric
le = LabelEncoder()
for col in categorical_cols:
    # print(col)
    data[col] = le.fit_transform(data[col])
print("le over")


X = data.drop(columns=['readmitted','payer_code','medical_specialty'])
y = data['readmitted']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lgbm_model = LGBMClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression()

# Train the base models on the training data
rf_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Make predictions on the testing data
rf_pred = rf_model.predict(X_test)
lgbm_pred = lgbm_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Create a new feature matrix using base model predictions as features
X_stack = np.column_stack((rf_pred, lgbm_pred, lr_pred))

# Define the meta-model (Logistic Regression in this example)
meta_model = LogisticRegression()

# Train the meta-model on the base model predictions
meta_model.fit(X_stack, y_test)

# Make predictions using the stacked ensemble model
stacked_pred = meta_model.predict(X_stack)

# Evaluate the stacked ensemble model
accuracy = accuracy_score(y_test, stacked_pred)
print(f"Stacked Ensemble Accuracy: {accuracy}")
