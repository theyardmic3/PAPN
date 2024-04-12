from cProfile import label
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

customer_data = pd.read_csv('customer_data.csv')

# Preprocessing
customer_data.dropna(inplace=True)
customer_data = pd.get_dummies(customer_data, columns=['gender', 'payment_method'])

# train
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
