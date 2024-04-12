from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.impute import SimpleImputer

# Load data
transaction_data = pd.read_csv('transaction_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='constant', fill_value='not_fraud')
transaction_data['flagged_transaction'].fillna('not_fraud', inplace=True)
transaction_data['fraud_label'] = (transaction_data['flagged_transaction'] == 'fraud').astype(int)

# Assuming data is loaded and preprocessed
clf = IsolationForest()
clf.fit(X_train)
predictions = clf.predict(X_test)
