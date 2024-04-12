import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load data
transaction_data = pd.read_csv('transaction_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='constant', fill_value='not_fraud')
transaction_data['flagged_transaction'].fillna('not_fraud', inplace=True)
transaction_data['fraud_label'] = (transaction_data['flagged_transaction'] == 'fraud').astype(int)

# Feature engineering
# Add more features based on transaction details, customer information, and other relevant data
transaction_data['transaction_amount'] = transaction_data['Total_Trans_Amt']
transaction_data['transaction_count'] = transaction_data['Total_Trans_Ct']
transaction_data['avg_monthly_transactions'] = transaction_data['Total_Trans_Ct'] / transaction_data['Months_on_book']

# Split data
X_train, X_test, y_train, y_test = train_test_split(transaction_data.drop('fraud_label', axis=1), transaction_data['fraud_label'], test_size=0.2)

# Train model
clf = IsolationForest()
clf.fit(X_train)

# Evaluate model
y_pred = clf.predict(X_test)
y_pred[y_pred == -1] = 1  # Treat anomalies as fraud
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Save model
joblib.dump(clf, 'fraud_detection_model.pkl')