import pandas as pd
import numpy as np

# Generate synthetic data for customer payment preference prediction
np.random.seed(42)

# Generate customer profiles
customer_profiles = pd.DataFrame({
    'customer_id': np.arange(1, 1001),
    'age': np.random.randint(18, 70, size=1000),
    'gender': np.random.choice(['Male', 'Female'], size=1000),
    'income': np.random.randint(20000, 150000, size=1000),
    # Add more demographic features as needed
})

# Generate transaction history
transaction_history = pd.DataFrame({
    'customer_id': np.random.randint(1, 1001, size=5000),
    'transaction_id': np.arange(1, 5001),
    'amount': np.random.randint(10, 1000, size=5000),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal'], size=5000),
    # Add more transaction-related features as needed
})

# Combine customer profiles and transaction history
customer_payment_data = pd.merge(transaction_history, customer_profiles, on='customer_id')

# Save dataset to CSV
customer_payment_data.to_csv('customer_payment_data.csv', index=False)
