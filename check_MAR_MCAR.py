import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load your dataset into a DataFrame (replace 'your_dataset.csv' with your actual dataset file path)
df = pd.read_csv('new_diabetic_data.csv')

# Assuming your DataFrame is named 'df'
df['readmitted'] = df['readmitted'].replace({'<30': 'Early Readmission', '>30': 'Late Readmission', 'NO': 'No Readmission'})

# Define the columns with missing values
columns_with_missing = ['race', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']

# Define a categorical variable from your dataset to test against (e.g., 'gender')
categorical_variable = 'readmitted'

# Loop through the columns with missing values and perform the Chi-Square Test
for column in columns_with_missing:
    contingency_table = pd.crosstab(df[column].isnull(), df[categorical_variable])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    alpha = 0.05  # Set the significance level
    
    print(f"Column: {column}")
    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-Value: {p}")
    
    if p < alpha:
        print(f"The missingness in '{column}' is associated with '{categorical_variable}'.")
        print("It suggests that the missingness may not be completely at random (MAR).\n")
    else:
        print(f"The missingness in '{column}' is not significantly associated with '{categorical_variable}'.")
        print("It suggests that the missingness may be closer to MCAR (Missing Completely at Random).\n")


# Columns to visualize
columns_to_visualize = ['race', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3']

# Create stacked bar plots for each column
for col in columns_to_visualize:
    ct = pd.crosstab(df[col].isna().replace({False: 'Not Missing', True: 'Missing'}), df['readmitted'])
    ct = ct.div(ct.sum(1), axis=0)
    ax = ct.plot(kind='bar', stacked=True, colormap='viridis', figsize=(8, 6))
    
    plt.title(f'Stacked Bar Plot for {col} vs Readmission')
    plt.xlabel(f'Missing {col}')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)  # Rotate X-axis labels to be horizontal
    ax.set_xticklabels(['Not Missing', 'Missing'])  # Set X-axis labels
    plt.legend(title='Readmitted', loc='upper right')
    plt.show()
