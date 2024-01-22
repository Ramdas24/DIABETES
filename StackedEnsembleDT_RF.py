from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset from the provided path
df = pd.read_csv('cleaned_diabetic_data.csv')

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
    df[col] = le.fit_transform(df[col])
print("le over")


# Assuming 'readmitted' is the target variable and the rest are features
X = df.drop(columns=['readmitted','payer_code','medical_specialty'])
y = df['readmitted']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
]

# Define meta-learner
meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)

# Build the Stacking classifier
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)

# Train the Stacking classifier
stacking_clf.fit(X_train, y_train)

# Predict test set
y_pred = stacking_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
