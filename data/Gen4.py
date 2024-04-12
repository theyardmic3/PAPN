import pandas as pd
import numpy as np

# Generate synthetic data for fraud detection in payment processing
np.random.seed(42)

# Generate transaction data
transaction_data = pd.DataFrame({
    'transaction_id': np.arange(1, 5001),
    'customer_id': np.random.randint(1, 1001, size=5000),
    'amount': np.random.randint(10, 1000, size=5000),
    'flagged_transaction': np.random.choice(['fraud', 'not_fraud'], size=5000),
    # Add more transaction-related features as needed
})

# Save dataset to CSV
transaction_data.to_csv('fraud_data.csv', index=False)
