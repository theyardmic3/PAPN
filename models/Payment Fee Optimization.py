import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
from sklearn.metrics import r2_score, root_mean_squared_error

import numpy as np

# Load data
transaction_data = pd.read_csv('data/transaction_data.csv')
fee_data = pd.read_csv('data/fee_structures.csv')

# Ensure 'amount' is numeric
transaction_data['amount'] = pd.to_numeric(transaction_data['amount'], errors='coerce')

# One-hot encode 'payment_method' and 'merchant_category' columns
encoder = OneHotEncoder(drop='first')
encoded_columns = encoder.fit_transform(transaction_data[['payment_method', 'merchant_category']])
encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=encoder.get_feature_names_out(['payment_method', 'merchant_category']))

# Concatenate encoded columns with original dataframe
transaction_data = pd.concat([transaction_data, encoded_df], axis=1)

# Drop the original 'payment_method' and 'merchant_category' columns
transaction_data.drop(['payment_method', 'merchant_category'], axis=1, inplace=True)

# Preprocess the 'amount' column
imputer = SimpleImputer(strategy='mean')
transaction_data['amount'] = imputer.fit_transform(transaction_data[['amount']].values)
scaler = MinMaxScaler()
transaction_data['amount_scaled'] = scaler.fit_transform(transaction_data[['amount']])

# Define the function to estimate fees
def estimate_fee(row):
    # Check if the fee_data DataFrame is not empty and has the necessary columns
    if fee_data.empty or not {'payment_method', 'fee_percentage'}.issubset(fee_data.columns):
        raise ValueError("fee_data must contain 'payment_method' and 'fee_percentage' columns with data.")
    
    payment_type_cols = ['payment_method_Credit Card', 'payment_method_Debit Card', 'payment_method_PayPal']
    for col in payment_type_cols:
        if col in row.index and row[col] == 1:
            payment_type = col.split('_')[-1]
            fee_percentage = fee_data.loc[fee_data['payment_method'] == payment_type, 'fee_percentage'].values[0]
            return fee_percentage * row['amount_scaled']
    return 0  # Return 0 or an appropriate value instead of None

# Add features based on payment type, merchant category, and estimated fees
transaction_data['estimated_fee'] = transaction_data.apply(estimate_fee, axis=1)

# Handle NaN values in 'estimated_fee'
transaction_data['estimated_fee'].fillna(0, inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(transaction_data.drop(['estimated_fee'], axis=1), transaction_data['estimated_fee'], test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predicted_fees = model.predict(X_test)

# Calculate R-squared
r_squared = r2_score(y_test, predicted_fees)
print(f"R-squared: {r_squared:.2f}")

## Calculate MdAPE
mdape = np.median(np.abs((y_test - predicted_fees) / y_test)) * 100
print(f"Median Absolute Percentage Error: {mdape:.2f}%")

# Save model
joblib.dump(model, 'payment_fee_optimization_model.pkl')
