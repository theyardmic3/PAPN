import pandas as pd
import numpy as np

# Generate synthetic data for dynamic payment fee optimization
np.random.seed(42)

# Generate transaction data
transaction_data = pd.DataFrame({
    'transaction_id': np.arange(1, 5001),
    'amount': np.random.randint(10, 1000, size=5000),
    'merchant_category': np.random.choice(['Retail', 'Food', 'Travel'], size=5000),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'E-Wallet', 'Bank Transfer', 'Cash' , 'Paypal'], size=3000),
    # Add more transaction-related features as needed
})

# Generate fee structures
fee_structures = pd.DataFrame({
    'payment_method': ['Credit Card', 'Debit Card', 'PayPal'],
    'fee_percentage': [0.02, 0.01, 0.005]
    
})

# Save datasets to CSV
transaction_data.to_csv('transaction_data.csv', index=False)
fee_structures.to_csv('fee_structures.csv', index=False)
