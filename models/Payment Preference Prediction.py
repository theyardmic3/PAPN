from cProfile import label
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load data
customer_data = pd.read_csv('customer_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
customer_data['gender'] = imputer.fit_transform(customer_data[['gender']])
customer_data = pd.get_dummies(customer_data, columns=['gender', 'payment_method'])


# train
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
