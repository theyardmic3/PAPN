from sklearn.ensemble import IsolationForest
import pandas as pd

transaction_data = pd.read_csv('transaction_data.csv')

# Preprocessing
transaction_data.dropna(inplace=True)
transaction_data['fraud_label'] = (transaction_data['flagged_transaction'] == 'fraud').astype(int)

# Assuming data is loaded and preprocessed
clf = IsolationForest()
clf.fit(X_train)
predictions = clf.predict(X_test)
