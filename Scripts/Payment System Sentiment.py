import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from joblib import dump, load
import joblib

# Load the dataset
data = pd.read_csv('data/sentiment_data.csv')

# Preprocess the text data
stop_words = set(stopwords.words('english'))
data['processed_text'] = data['review_text'].apply(lambda x: ' '.join(
    [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words]
))

# Vectorize the text data using TF-IDF with bigrams and trigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 5))
X = vectorizer.fit_transform(data['processed_text'])

# Use 'sentiment_label' as the target variable
y = data['sentiment_label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Make predictions and evaluate the model
predictions = model.predict(X_test)


# Analyze sentiment for each payment method
payment_methods = data['payment_method'].unique()
for method in payment_methods:
    method_data = data[data['payment_method'] == method]
    method_vectorized = vectorizer.transform(method_data['processed_text'])
    method_predictions = model.predict(method_vectorized)
    positive_sentiment = sum(method_predictions == 'positive')
    negative_sentiment = sum(method_predictions == 'negative')
    neutral_sentiment = sum(method_predictions == 'neutral')
    print(f"Sentiment analysis for {method}:")
    print(f"Positive: {positive_sentiment}, Neutral: {neutral_sentiment}, Negative: {negative_sentiment}")

# Save the model to a file in the /Models directory
joblib.dump(model, 'Models/Payment_Sentiment_model.joblib')
