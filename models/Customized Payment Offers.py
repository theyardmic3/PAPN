import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data
purchase_data = pd.read_csv('purchase_data.csv')

# Preprocessing
imputer = SimpleImputer(strategy='most_frequent')
purchase_data['loyalty_status'] = imputer.fit_transform(purchase_data[['loyalty_status']])
purchase_data = pd.get_dummies(purchase_data, columns=['loyalty_status'])

# Feature engineering
# Add more features based on customer demographics, past purchase behavior, and response to previous offers
purchase_data['total_purchases'] = purchase_data.groupby('CustomerID')['Quantity'].transform('sum')
purchase_data['avg_order_value'] = purchase_data['UnitPrice'] * purchase_data['Quantity']

# Compute similarity matrix
customer_features = purchase_data[['total_purchases', 'avg_order_value', 'loyalty_status_Bronze', 'loyalty_status_Gold', 'loyalty_status_Silver']]
similarity_matrix = cosine_similarity(customer_features)

# Save similarity matrix
joblib.dump(similarity_matrix, 'customer_similarity_matrix.pkl')