import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import joblib

# Function to return a list of bank holidays
def get_bank_holidays():
    # Common known bank holidays
    return [
        '2023-01-01',  # New Year's Day
        '2023-04-07',  # Good Friday
        '2023-04-10',  # Easter Monday
        '2023-05-01',  # Early May bank holiday
        '2023-05-29',  # Spring bank holiday
        '2023-08-28',  # Summer bank holiday
        '2023-12-25',  # Christmas Day
        '2023-12-26',  # Boxing Day
        '2023-01-16',  # Martin Luther King Jr. Day
        '2023-02-20',  # Washington's Birthday
        '2023-05-29',  # Memorial Day
        '2023-07-04',  # Independence Day
        '2023-09-04',  # Labor Day
        '2023-11-23',  # Thanksgiving Day
    ]

# Load data
payment_data = pd.read_csv('data/payment_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
payment_data['due_date'] = imputer.fit_transform(payment_data[['due_date']])
payment_data['due_date'] = pd.to_datetime(payment_data['due_date'])

# Feature engineering
payment_data['day'] = payment_data['due_date'].dt.day
payment_data['month'] = payment_data['due_date'].dt.month
payment_data['year'] = payment_data['due_date'].dt.year
payment_data['is_weekend'] = payment_data['due_date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
payment_data['is_holiday'] = payment_data['due_date'].dt.date.isin(get_bank_holidays()).astype(int)

# One-hot encode 'payment_method' column
encoder = OneHotEncoder()
encoded_payment_method = encoder.fit_transform(payment_data[['payment_method']]).toarray()
encoded_payment_method_df = pd.DataFrame(encoded_payment_method, columns=encoder.get_feature_names_out(['payment_method']))

# Concatenate encoded columns with original dataframe
payment_data = pd.concat([payment_data, encoded_payment_method_df], axis=1)
# Drop the original 'due_date' and 'payment_method' columns as they're no longer needed in this format
payment_data.drop(['due_date', 'payment_method'], axis=1, inplace=True)

# Split data
X = payment_data.drop(['payment_processing_time'], axis=1)
y = payment_data['payment_processing_time']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Save model
joblib.dump(model, 'payment_timing_prediction_model.pkl')
