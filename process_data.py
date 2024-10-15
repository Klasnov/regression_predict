import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load the dataset from the relative path
data = pd.read_csv("data/train.csv")

# Display the first few rows to understand the data structure
print("Initial Data Preview:")
print(data.head())

# Check for missing values in each column
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Handle missing values for numerical columns by filling them with the median
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    median_value = data[col].median()
    data[col].fillna(median_value, inplace=True)
    print(f"Filled missing values in '{col}' with median value {median_value}")

# Handle missing values for categorical columns by filling them with the mode
categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
for col in categorical_cols:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)
    print(f"Filled missing values in '{col}' with mode value '{mode_value}'")

# Convert date columns to datetime objects
date_cols = ["original_reg_date", "reg_date", "lifespan"]
for col in date_cols:
    data[col] = pd.to_datetime(data[col], errors="coerce")
    print(f"Converted '{col}' to datetime")

# Handle any remaining missing values in date columns by filling with a default date
default_date = pd.Timestamp("1900-01-01")
data[date_cols] = data[date_cols].fillna(default_date)
print(f"Filled missing dates with default date {default_date}")

# Encode categorical variables using Label Encoding
label_encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])
    print(f"Encoded categorical column '{col}'")

# Preview the processed data
print("\nProcessed Data Preview:")
print(data.head())

# Save the processed data to a new CSV file
data.to_csv("data/processed_train.csv", index=False)
print("\nProcessed data saved to 'data/processed_train.csv'")
