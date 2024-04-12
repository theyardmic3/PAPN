import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load data
transaction_data = pd.read_csv('transaction_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='mean')
transaction_data['amount'] = imputer.fit_transform(transaction_data[['amount']])
scaler = MinMaxScaler()
transaction_data['amount_scaled'] = scaler.fit_transform(transaction_data[['amount']])

# Feature engineering
# Add features based on payment type, merchant category, and estimated fees
transaction_data['estimated_fee'] = transaction_data.apply(lambda row: estimate_fee(row['payment_type'], row['amount_scaled']), axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(transaction_data.drop('estimated_fee', axis=1), transaction_data['estimated_fee'], test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predicted_fees = model.predict(X_test)
mae = mean_absolute_error(y_test, predicted_fees)
mse = mean_squared_error(y_test, predicted_fees)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Save model
joblib.dump(model, 'payment_fee_optimization_model.pkl')