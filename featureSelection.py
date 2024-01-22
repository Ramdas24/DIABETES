import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import time

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
# Split the data into features (X) and target (y)
X = data.drop(columns=['readmitted','payer_code','medical_specialty'])
y = data['readmitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature Selection with Chi-squared test
# chi2_selector = SelectKBest(chi2, k=40)  # Select the top 10 features
# X_train_chi2 = chi2_selector.fit_transform(X_train, y_train)
# X_test_chi2 = chi2_selector.transform(X_test)  # Transform the test set using the same selector
# print("Chi square completed")

# # Feature Selection with Mutual Information
# mi_selector = SelectKBest(mutual_info_classif, k=40)  # Select the top 10 features
# X_train_mi = mi_selector.fit_transform(X_train, y_train)
# X_test_mi = mi_selector.transform(X_test)  # Transform the test set using the same selector
# print("MI completed")

# # Feature Selection with RFE and Logistic Regression
# logistic_regression = LogisticRegression()
# rfe_selector = RFE(estimator=logistic_regression, n_features_to_select=10, step=1)
# X_train_rfe = rfe_selector.fit_transform(X_train, y_train)
# X_test_rfe = rfe_selector.transform(X_test)  # Transform the test set using the same selector
# print("RFE completed")

# Feature Selection with Random Forest Feature Importance
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order
top_k = 40  # Select the top 10 features
X_train_rf = X_train.iloc[:, indices[:top_k]]
X_test_rf = X_test.iloc[:, indices[:top_k]] 

# # Visualize feature importances from Random Forest
# plt.figure(figsize=(10, 6))
# plt.title("Random Forest Feature Importances")
# plt.bar(range(top_k), importances[indices[:top_k]], align="center")
# plt.xticks(range(top_k), X_train.columns[indices[:top_k]], rotation=90)
# plt.tight_layout()
# plt.show()

# # Evaluate feature selection methods with a classifier (e.g., Random Forest)
# rf_classifier.fit(X_train_chi2, y_train)
# y_pred_chi2 = rf_classifier.predict(X_test_chi2)
# accuracy_chi2 = accuracy_score(y_test, y_pred_chi2)

# rf_classifier.fit(X_train_mi, y_train)
# y_pred_mi = rf_classifier.predict(X_test_mi)
# accuracy_mi = accuracy_score(y_test, y_pred_mi)

# rf_classifier.fit(X_train_rfe, y_train)
# y_pred_rfe = rf_classifier.predict(X_test_rfe)
# accuracy_rfe = accuracy_score(y_test, y_pred_rfe)

# rf_classifier.fit(X_train_rf, y_train)
# y_pred_rf = rf_classifier.predict(X_test_rf)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)

# # Print accuracy scores
# print(f"Accuracy (Chi-squared test): {accuracy_chi2}")
# print(f"Accuracy (Mutual Information): {accuracy_mi}")
# print(f"Accuracy (RFE with Logistic Regression): {accuracy_rfe}")
# print(f"Accuracy (Random Forest Feature Importance): {accuracy_rf}")

# # Plot Chi-squared selected features
# plt.figure(figsize=(10, 6))
# plt.title("Chi-squared Selected Features")
# plt.bar(range(len(chi2_selector.scores_)), chi2_selector.scores_)
# plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=90)
# plt.tight_layout()
# plt.show()

# # Plot RFE selected features
# plt.figure(figsize=(10, 6))
# plt.title("RFE Selected Features")
# plt.bar(range(len(rfe_selector.support_)), rfe_selector.support_)
# plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=90)
# plt.tight_layout()
# plt.show()

# # Plot Mutual Information selected features
# plt.figure(figsize=(10, 6))
# plt.title("Mutual Information Selected Features")
# plt.bar(range(len(mi_selector.scores_)), mi_selector.scores_)
# plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=90)
# plt.tight_layout()
# plt.show()

# Standardize the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_rf)
X_test_std = scaler.transform(X_test_rf)

# Define a dictionary to store model results
results = {
    'Model': [],
    'Training Time (s)': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'ROC-AUC': [],
    'Confusion Matrix': []
}

# Define a function to evaluate a model and store the results
def evaluate_model(model, model_name):
    start_time = time.time()
    model.fit(X_train_std, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    y_pred = model.predict(X_test_std)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Use average='weighted' for multiclass classification
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Specify multi_class parameter for ROC-AUC
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_std), multi_class='ovr')
    confusion = confusion_matrix(y_test, y_pred)
    
    results['Model'].append(model_name)
    results['Training Time (s)'].append(training_time)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1-Score'].append(f1)
    results['ROC-AUC'].append(roc_auc)
    results['Confusion Matrix'].append(confusion)
    print(model_name, " Completed")

# Evaluate different models
# svm_model = SVC()
nb_model = GaussianNB()
knn_model = KNeighborsClassifier()
lr_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lgbm_model = LGBMClassifier()
xgb_model = xgb.XGBClassifier(random_state=42)

# evaluate_model(svm_model, 'SVM')
evaluate_model(nb_model, 'Naive Bayes')
evaluate_model(knn_model, 'K-Nearest Neighbors')
evaluate_model(lr_model, 'Logistic Regression')
evaluate_model(rf_model, 'Random Forest')
evaluate_model(lgbm_model, 'LGBM')
evaluate_model(xgb_model, 'XGBoost Classifier')

# Display the results
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_json('results.json')