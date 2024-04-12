import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Expanded list of locations
locations = [
    'Online', 'Retail Store', 'ATM', 'Gas Station', 'Restaurant', 'Supermarket',
    'Hotel', 'Bar', 'Cinema', 'Amusement Park', 'Museum', 'Fitness Center'
]

# Expanded and varied list of countries
countries = [
    'USA', 'Canada', 'Australia', 'Japan', 'Germany', 'India',
    'Brazil', 'South Africa', 'Netherlands', 'Singapore', 'Mexico', 'Italy',
    'Spain', 'Russia', 'China', 'United Kingdom', 'France', 'New Zealand',
    'Sweden', 'Switzerland'
]

# Generate synthetic data for fraud detection in payment processing
def generate_synthetic_data(num_samples=5000):
    # Generate transaction IDs and customer IDs
    transaction_ids = np.arange(1, num_samples + 1)
    customer_ids = np.random.randint(1, 1001, size=num_samples)
    
    # Generate transaction amounts with higher values for fraud cases
    amounts = np.random.exponential(scale=50, size=num_samples).astype(int) + 10
    amounts[np.random.choice(num_samples, size=int(num_samples * 0.05), replace=False)] += 1000
    
    # Simulate time of day (0-23 hours), with fraud more likely at night
    time_of_day = np.random.randint(0, 24, size=num_samples)
    fraud_hours = np.random.choice(range(0, 8), size=int(num_samples * 0.1), replace=True)
    
    # Define fraud locations
    fraud_locations = ['Online', 'ATM']  # Add this line to define fraud locations
    
    # Assign locations randomly, with corrected probabilities
    transaction_locations = np.random.choice(locations, size=num_samples, p=[1/12] * 12)
    
    # Assign countries randomly
    transaction_countries = np.random.choice(countries, size=num_samples)
    
    # Flag transactions as fraud based on amount, time of day, and location
    flagged_transactions = ['not_fraud'] * num_samples
    for i in range(num_samples):
        if (amounts[i] > 950 and time_of_day[i] in fraud_hours) or (transaction_locations[i] in fraud_locations):
            flagged_transactions[i] = 'fraud'
    
    # Introduce some noise
    noise_indices = np.random.choice(num_samples, size=int(num_samples * 0.01), replace=False)
    for i in noise_indices:
        flagged_transactions[i] = 'fraud' if flagged_transactions[i] == 'not_fraud' else 'not_fraud'
    
    # Create DataFrame with modified 'point_of_transaction' and new 'country' column
    transaction_data = pd.DataFrame({
        'transaction_id': transaction_ids,
        'customer_id': customer_ids,
        'amount': amounts,
        'time_of_day': time_of_day,
        'point_of_transaction': transaction_locations,
        'country': transaction_countries,
        'flagged_transaction': flagged_transactions
    })
    
    return transaction_data

# Generate the data
synthetic_data = generate_synthetic_data()

# Save dataset to CSV
synthetic_data.to_csv('enhanced_fraud_data_with_varied_locations_and_countries.csv', index=False)
