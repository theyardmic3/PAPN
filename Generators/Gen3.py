import pandas as pd
import numpy as np
import datetime

# Generate synthetic data for predictive analysis for payment timing
np.random.seed(42)

# Generate payment data
payment_data = pd.DataFrame({
    'payment_id': np.arange(1, 5001),
    'due_date': pd.date_range(start='2023-01-01', periods=5000, freq='D'),
    'payment_processing_time': np.random.randint(1, 10, size=5000),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'E-Wallet', 'Bank Transfer', 'Cash' , 'Paypal'], size=3000),
    
})

# Save dataset to CSV
payment_data.to_csv('payment_data.csv', index=False)
