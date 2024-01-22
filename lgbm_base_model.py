import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import joblib  

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
    data[col] = le.fit_transform(data[col])

data.to_csv('transformed_diabetic_data.csv', index=False)

# Split the data into features (X) and target (y)
X = data.drop(columns=['readmitted', 'payer_code', 'medical_specialty'])
y = data['readmitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the trained model and label encoder using joblib
# joblib.dump(lgbm_model, 'lgbm_model.pkl')
# joblib.dump(le, 'label_encoder.pkl')

print(f"Model accuracy: {accuracy}")
print("Model and label encoder saved.")
# After fitting the model, check the number of features used
n_features_used = lgbm_model.n_features_
print(f"Number of features used in the model: {n_features_used}")

# After fitting the model, print the names of the features used
feature_names = lgbm_model.feature_name_
print(f"Feature names used in the model: {feature_names}")

