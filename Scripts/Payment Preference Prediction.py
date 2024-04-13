import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import classification_report

# Load data
transaction_data = pd.read_csv('data/customer_payment_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
transaction_data['payment_method'] = imputer.fit_transform(transaction_data[['payment_method']])

# One-hot encode 'payment_method' column
encoder = OneHotEncoder(drop='first')  # Sparse=False is default, so no need to specify
payment_method_encoded = encoder.fit_transform(transaction_data[['payment_method']])
payment_method_encoded_df = pd.DataFrame(payment_method_encoded.toarray(), columns=encoder.get_feature_names_out(['payment_method']))
transaction_data = pd.concat([transaction_data, payment_method_encoded_df], axis=1)

# Encode 'gender' column
gender_mapping = {'Male': 0, 'Female': 1}
transaction_data['gender'] = transaction_data['gender'].map(gender_mapping)

# Feature engineering
transaction_data['age_income_ratio'] = transaction_data['age'] / transaction_data['income']
transaction_data['amount_mean'] = transaction_data.groupby('customer_id')['amount'].transform('mean')

# Split data
features = transaction_data.drop(['transaction_id', 'customer_id', 'payment_method'], axis=1)  # Remove non-feature columns
label = transaction_data['payment_method']  # Target label for classification
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Additional evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model into the /Models directory
joblib.dump(clf, 'Models/Payment Preference Prediction.pkl')
