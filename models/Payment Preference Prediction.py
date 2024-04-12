import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load data
transaction_data = pd.read_csv('data/transactions.csv', on_bad_lines='skip')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
transaction_data['product_class'] = imputer.fit_transform(transaction_data[['product_class']])
transaction_data['product_size'] = imputer.fit_transform(transaction_data[['product_size']])
transaction_data = pd.get_dummies(transaction_data, columns=['brand', 'product_line', 'product_class', 'product_size'])

# Feature engineering
transaction_data['order_value'] = transaction_data['list_price'] * transaction_data['order_status'].map({'Approved': 1, 'Declined': 0})
transaction_data['online_order_ratio'] = transaction_data.groupby('customer_id')['online_order'].transform('mean')

# Split data
features = transaction_data.drop(['transaction_id', 'product_id', 'customer_id', 'transaction_date', 'order_status', 'list_price', 'standard_cost', 'product_first_sold_date'], axis=1)
label = transaction_data['order_status']
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
joblib.dump(clf, 'order_status_prediction_model.pkl')