import pandas as pd
import numpy as np

# Generate synthetic data for customized payment offers
np.random.seed(42)

# Define additional product categories
additional_categories = ['Home & Garden', 'Beauty & Health', 'Toys & Games', 'Sports & Outdoors']

# Define a list of countries
countries = [
    'USA', 'Canada', 'Australia', 'Japan', 'Germany', 'India',
    'Brazil', 'South Africa', 'Netherlands', 'Singapore', 'Mexico', 'Italy',
    'Spain', 'Russia', 'China', 'United Kingdom', 'France', 'New Zealand',
    'Sweden', 'Switzerland'
]

# Define possible payment methods
payment_methods = ['Credit Card', 'Debit Card', 'E-Wallet', 'Bank Transfer', 'Cash' , 'Paypal']

# Generate purchase data with promotional information
purchase_data = pd.DataFrame({
    'customer_id': np.random.randint(1, 1001, size=5000),
    'purchase_id': np.arange(1, 5001),
    'product_category': np.random.choice(['Electronics', 'Clothing', 'Books'] + additional_categories, size=5000),
    'promotion_applied': np.random.choice([True, False], size=5000, p=[0.3, 0.7]),
    'promotion_discount': np.round(np.random.uniform(3, 20, size=5000), 0) / 100,  # Discounts between 3% and 20%
    'country': np.random.choice(countries, size=5000),  # Add country location
    'payment_method': np.random.choice(payment_methods, size=5000),  # Add payment methods
    'payment_fee_percentage': np.round(np.random.uniform(0.5, 5, size=5000), 2),  # Payment fee between 0.5% and 5%
    # Add more purchase-related features as needed
})

# Save dataset to CSV
purchase_data.to_csv('promo_data.csv', index=False)
