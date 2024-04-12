from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load data
transaction_data = pd.read_csv('transaction_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='mean')
transaction_data['amount'] = imputer.fit_transform(transaction_data[['amount']])
scaler = MinMaxScaler()
transaction_data['amount_scaled'] = scaler.fit_transform(transaction_data[['amount']])


# Assuming data is loaded and preprocessed
model = LinearRegression()
model.fit(X_train, y_train)
predicted_fees = model.predict(X_test)
