import pandas as pd
import numpy as np

# Load your dataset into a DataFrame (replace 'your_dataset.csv' with your actual dataset file path)
df = pd.read_csv('new_diabetic_data.csv')
# Define a function to convert age ranges to numeric values
def convert_age_range_to_numeric(age_range):
    # Split the age range to extract the lower bound
    lower_bound = int(age_range.strip('[]()').split('-')[0])
    # Calculate the midpoint
    midpoint = lower_bound + 5
    return midpoint

# Apply the conversion function to the 'age' column
df['age'] = df['age'].apply(convert_age_range_to_numeric)

def preprocess_and_impute_weight(df):
    # Define a mapping for weight ranges to numeric values
    weight_mapping = {
        '[0-25)': 12.5,
        '[25-50)': 37.5,
        '[50-75)': 62.5,
        '[75-100)': 87.5,
        '[100-125)': 112.5,
        '[125-150)': 137.5,
        '[150-175)': 162.5,
        '[175-200)': 187.5,
        '>200': 210.0,  # Assuming >200 represents a value greater than 200
    }
    
    # Replace weight values with their numeric equivalents
    df['weight'] = df['weight'].replace(weight_mapping)
    
    # Group the data by age and calculate the mean weight for each age group
    age_weight_mean = df.groupby('age')['weight'].mean()
    
    # Iterate through the rows and impute missing weight values based on age group
    for index, row in df.iterrows():
        if pd.isna(row['weight']):  # Check if weight is missing
            age_group = row['age']
            if age_group in age_weight_mean.index:
                df.at[index, 'weight'] = age_weight_mean[age_group]
    
    return df

# Call the function to preprocess and impute missing weight values
df = preprocess_and_impute_weight(df)

df['readmitted'] = df['readmitted'].replace({'<30': 'EarlyReadmission', '>30': 'LateReadmission', 'NO': 'NoReadmission'})

# Define the columns to impute based on their associations with 'Readmitted'
mar_columns = ['race', 'payer_code', 'medical_specialty', 'diag_2', 'diag_3']
mcar_columns = ['diag_1']

# Apply imputation methods for MAR columns
for column in mar_columns:
    if df[column].dtype == 'object':
        # For object (categorical) columns, impute with mode
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)
    else:
        # For numeric columns, impute with mean
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)

# Apply imputation methods for MCAR columns
for column in mcar_columns:
    if df[column].dtype == 'object':
        # For object (categorical) columns, impute with mode
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)
    else:
        # For numeric columns, impute with median
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

prefix_mapping = {
    'E': 1000,  # Replace 'E' with 1000
    'V': 2000,  # Replace 'V' with 2000
}

# Function to replace letter prefixes with numeric values
def replace_prefix(code):
    for prefix, numeric_value in prefix_mapping.items():
        if str(code).startswith(prefix):
            return str(code).replace(prefix, str(numeric_value))
    return code  # If no prefix is found, return the original code as is

# Apply the mapping function to 'diag_1', 'diag_2', and 'diag_3' columns
df['diag_1'] = df['diag_1'].apply(replace_prefix)
df['diag_2'] = df['diag_2'].apply(replace_prefix)
df['diag_3'] = df['diag_3'].apply(replace_prefix)

# Verify that missing values have been imputed
missing_values = df.isnull().sum()
print(missing_values)
# Display the updated DataFrame with imputed values
print(df.head())

# Save the imputed DataFrame to a new CSV file
df.to_csv('cleaned_diabetic_data.csv', index=False)
