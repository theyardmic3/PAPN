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

# Function to assign sentiment label based on rating
def assign_sentiment_label(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

# Generate synthetic data for payment system sentiment analysis
def generate_sentiment_data(num_samples=10000):
    # Generate review IDs and customer IDs
    review_ids = np.arange(1, num_samples + 1)
    customer_ids = np.random.randint(1, 1001, size=num_samples)
    
    # Generate random payment methods
    payment_methods = np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'], size=num_samples)
    
    # Generate ratings with a more balanced distribution
    ratings = []
    for _ in range(num_samples):
        rating_choice = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.2])
        ratings.append(rating_choice)
    ratings = np.array(ratings)
    
    # Generate random review texts
    review_texts = np.random.choice(['Excellent service!', 'Terrible experience...', 'Average quality.'], size=num_samples)
    
    # Generate random points of transaction
    transaction_points = np.random.choice(points_of_transaction, size=num_samples)
    
    # Generate random countries
    transaction_countries = np.random.choice(countries, size=num_samples)
    
    # Assign sentiment labels
    sentiment_labels = [assign_sentiment_label(rating) for rating in ratings]
    
    # Create DataFrame with new 'point_of_transaction', 'country', and 'sentiment_label' columns
    sentiment_data = pd.DataFrame({
        'review_id': review_ids,
        'customer_id': customer_ids,
        'payment_method': payment_methods,
        'rating': ratings,
        'review_text': review_texts,
        'point_of_transaction': transaction_points,
        'country': transaction_countries,
        'sentiment_label': sentiment_labels
    })
    
    return sentiment_data

# Generate the data
sentiment_data = generate_sentiment_data()

# Save dataset to CSV
sentiment_data.to_csv('sentiment_data.csv', index=False)
