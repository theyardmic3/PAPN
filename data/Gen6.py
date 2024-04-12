import pandas as pd
import numpy as np

# Generate synthetic data for payment system sentiment analysis
np.random.seed(42)

# Generate customer review data tied to payment method
customer_reviews = pd.DataFrame({
    'review_id': np.arange(1, 5001),
    'customer_id': np.random.randint(1, 1001, size=5000),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal'], size=5000),
    'rating': np.random.randint(1, 6, size=5000),
    'review_text': np.random.choice(['Excellent service!', 'Terrible experience...', 'Average quality.'], size=5000),
    # Add more review-related features as needed
})

# Save dataset to CSV
customer_reviews.to_csv('customer_reviews.csv', index=False)
