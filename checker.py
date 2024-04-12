import pandas as pd

# Load data
transaction_data = pd.read_csv('data/transaction_data.csv')

# Print column names before one-hot encoding
print("Column names before one-hot encoding:", transaction_data.columns)

# One-hot encoding for categorical variables: payment_method and merchant_category
transaction_data = pd.get_dummies(transaction_data, columns=['payment_method', 'merchant_category'])

# Print column names after one-hot encoding
print("Column names after one-hot encoding:", transaction_data.columns)
