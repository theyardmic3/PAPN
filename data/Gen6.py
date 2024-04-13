import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define possible points of transaction
points_of_transaction = [
    'Online', 'Retail Store', 'ATM', 'Gas Station', 'Restaurant', 'Supermarket',
    'Hotel', 'Bar', 'Cinema', 'Amusement Park', 'Museum', 'Fitness Center'
]

# Define a list of countries
countries = [
    'USA', 'Canada', 'Australia', 'Japan', 'Germany', 'India',
    'Brazil', 'South Africa', 'Netherlands', 'Singapore', 'Mexico', 'Italy',
    'Spain', 'Russia', 'China', 'United Kingdom', 'France', 'New Zealand',
    'Sweden', 'Switzerland'
]

# Generate synthetic data for payment system sentiment analysis
def generate_sentiment_data(num_samples=3000):
    # Generate review IDs and customer IDs
    review_ids = np.arange(1, num_samples + 1)
    customer_ids = np.random.randint(1, 1001, size=num_samples)
    
    # Generate random payment methods
    payment_methods = np.random.choice(['Credit Card', 'Debit Card', 'PayPal' , 'Bank Transfer'], size=num_samples)
    
    # Generate random ratings
    ratings = np.random.randint(1, 6, size=num_samples)
    
    # Generate random review texts
    review_texts = np.random.choice(['Excellent service!', 'Terrible experience...', 'Average quality.'], size=num_samples)
    
    # Generate random points of transaction
    transaction_points = np.random.choice(points_of_transaction, size=num_samples)
    
    # Generate random countries
    transaction_countries = np.random.choice(countries, size=num_samples)
    
    # Create DataFrame with new 'point_of_transaction' and 'country' columns
    sentiment_data = pd.DataFrame({
        'review_id': review_ids,
        'customer_id': customer_ids,
        'payment_method': payment_methods,
        'rating': ratings,
        'review_text': review_texts,
        'point_of_transaction': transaction_points,
        'country': transaction_countries
    })
    
    return sentiment_data

# Generate the data
sentiment_data = generate_sentiment_data()

# Save dataset to CSV
sentiment_data.to_csv('sentiment_data_with_transaction_and_country.csv', index=False)
