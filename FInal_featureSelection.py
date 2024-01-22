import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
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

# Feature Selection with Random Forest Feature Importance
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order
top_k = 25  # Select the top 10 features
X_train_rf = X_train.iloc[:, indices[:top_k]]
X_test_rf = X_test.iloc[:, indices[:top_k]] 

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

base_learners = [
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# Define the meta-learner
meta_learner = LogisticRegression(max_iter=1000, random_state=42)

# Build the stacking classifier
stacking_clf_1 = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

# Define the base learners
base_learners_1 = [
    ('rf_model', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lgbm_model', LGBMClassifier(n_estimators=100, random_state=42)),
    ('xgb_model', XGBClassifier(n_estimators=100, random_state=42))
]

# Define the meta learner
meta_learner_1 = RandomForestClassifier(n_estimators=50, random_state=42)

# Create the stacking classifier
stacker = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

# evaluate_model(svm_model, 'SVM')
evaluate_model(nb_model, 'Naive Bayes')
evaluate_model(knn_model, 'K-Nearest Neighbors')
evaluate_model(lr_model, 'Logistic Regression')
evaluate_model(rf_model, 'Random Forest')
evaluate_model(lgbm_model, 'LGBM')
evaluate_model(xgb_model, 'XGBoost Classifier')
evaluate_model(stacking_clf_1, 'Stacking_NB_KNN_LR')
evaluate_model(stacker, 'Stacking_RF_LGBM_XGB')

# Display the results
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_json('results.json')

# Results obtained from your code
models = results['Model']
accuracy = results['Accuracy']
precision = results['Precision']
recall = results['Recall']
f1_score = results['F1-Score']
roc_auc = results['ROC-AUC']
training_time = results['Training Time (s)']

def plot_metric_with_values(metric, ylabel):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, metric, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} by Model')
    plt.xticks(rotation=45)
    
    # Add data values on top of the bars
    for bar, value in zip(bars, metric):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, value + 0.01, f'{value:.2f}', fontsize=10)

    plt.tight_layout()
    plt.show()

plot_metric_with_values(accuracy, 'Accuracy')
plot_metric_with_values(precision, 'Precision')
plot_metric_with_values(recall, 'Recall')
plot_metric_with_values(f1_score, 'F1-Score')
plot_metric_with_values(roc_auc, 'ROC-AUC')
plot_metric_with_values(training_time, 'Training Time (s)')

# Get the confusion matrices for all evaluated models
confusion_matrices = results['Confusion Matrix']

# Get class labels and their names
class_labels = np.unique(y_test)
class_names = le.inverse_transform(class_labels)  # Assuming you used LabelEncoder earlier

# Create a dictionary to map class labels to class names
class_label_to_name = dict(zip(class_labels, class_names))

# Iterate through each model and plot its confusion matrix with class names
for model_name, confusion_matrix_model in zip(models, confusion_matrices):
    class_names_with_labels = [f"{label}\n({class_label_to_name[label]})" for label in class_labels]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_model, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_with_labels, yticklabels=class_names_with_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()