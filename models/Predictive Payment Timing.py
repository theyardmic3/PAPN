import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load data
payment_data = pd.read_csv('payment_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
payment_data['due_date'] = imputer.fit_transform(payment_data[['due_date']])
payment_data['due_date'] = pd.to_datetime(payment_data['due_date'])

# Feature engineering
payment_data['payment_processing_time'] = (payment_data['due_date'] - payment_data['payment_date']).dt.days
payment_data['is_weekend'] = payment_data['due_date'].dt.day_name().isin(['Saturday', 'Sunday'])
payment_data['is_holiday'] = payment_data['due_date'].dt.date.isin(get_bank_holidays())  # Implement get_bank_holidays() function

# Split data
X_train, X_test, y_train, y_test = train_test_split(payment_data.drop('payment_processing_time', axis=1), payment_data['payment_processing_time'], test_size=0.2, shuffle=False)

# Train model
model = ARIMA(endog=y_train, order=(5, 1, 0))
fitted_model = model.fit()

# Evaluate model
forecast = fitted_model.forecast(steps=len(y_test))
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Save model
joblib.dump(fitted_model, 'payment_timing_prediction_model.pkl')