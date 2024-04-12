import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Load data
customer_data = pd.read_csv('customer_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
customer_data['gender'] = imputer.fit_transform(customer_data[['gender']])
customer_data = pd.get_dummies(customer_data, columns=['gender', 'payment_method'])

# Feature engineering
# Add more features based on the transaction history, demographics, and behavior
customer_data['avg_order_value'] = customer_data['total_sales'] / customer_data['num_orders']
customer_data['frequency_of_orders'] = customer_data['num_orders'] / customer_data['customer_lifetime_days']

# Split data
features = customer_data.drop('payment_method', axis=1)
label = customer_data['payment_method']
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
joblib.dump(clf, 'customer_payment_preference_model.pkl')