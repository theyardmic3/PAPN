import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load the dataset
purchase_data = pd.read_csv('data/promo_data.csv')

# Create a new column that combines all the features you want to use for recommendations
# Now including 'payment_method' and 'payment_fee_percentage'
purchase_data['features'] = purchase_data.apply(lambda x: f"{x['product_category']} {'Promo' if x['promotion_applied'] else 'NoPromo'} {x['country']} {x['payment_method']} {'HighFee' if x['payment_fee_percentage'] > 3 else 'LowFee'}", axis=1)

# Create a TF-IDF Vectorizer object
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the 'features' column to create the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(purchase_data['features'])

# Compute the cosine similarity matrix from the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations for a given customer ID
def get_recommendations(customer_id, purchase_data, cosine_sim):
    # Get the index of all purchases made by the customer
    indices = purchase_data.index[purchase_data['customer_id'] == customer_id].tolist()
    
    # Get the similarity scores for all other purchases
    sim_scores = []
    for idx in indices:
        sim_scores.extend(list(enumerate(cosine_sim[idx])))
    
    # Sort the purchases based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top 10 most similar purchases
    sim_scores = sim_scores[1:11]
    
    # Get the purchase indices
    purchase_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar purchases
    return purchase_data.iloc[purchase_indices]

# Example usage: Get recommendations for customer with ID 1
recommended_payments = get_recommendations(1, purchase_data, cosine_sim)
print(recommended_payments)

# Save the model to a file
joblib.dump(cosine_sim, 'Payment_Offers_model.pkl')