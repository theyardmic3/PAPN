import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
transaction_data = pd.read_csv('data/fraud_data.csv')

# Convert 'flagged_transaction' to binary labels
transaction_data['fraud_label'] = (transaction_data['flagged_transaction'] == 'fraud').astype(int)

# Select features and target variable
features = ['amount', 'time_of_day', 'point_of_transaction', 'country']
X = transaction_data[features]
y = transaction_data['fraud_label']

# Preprocessing: Standardize numerical features and one-hot encode categorical features
numerical_features = ['amount', 'time_of_day']
categorical_features = ['point_of_transaction', 'country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create a pipeline that first preprocesses the data then fits the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier within the pipeline
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the classifier
print(classification_report(y_test, y_pred))

# Save the model into the /Models directory
joblib.dump(pipeline, 'Models/fraud_detection_model.pkl')

