import pandas as pd
import numpy as np

# Generate synthetic data for customized payment offers
np.random.seed(42)

# Generate purchase data with promotional information
purchase_data = pd.DataFrame({
    'customer_id': np.random.randint(1, 1001, size=5000),
    'purchase_id': np.arange(1, 5001),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'], size=5000),
    'promotion_applied': np.random.choice([True, False], size=5000, p=[0.3, 0.7]),
    'promotion_discount': np.random.uniform(0.05, 0.30, size=5000),
    # Add more purchase-related features as needed
})

# Save dataset to CSV
purchase_data.to_csv('promo_data.csv', index=False)
