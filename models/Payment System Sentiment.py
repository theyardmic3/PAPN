from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Load data
customer_reviews = pd.read_csv('/data/customer_reviews.csv')

# Preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

customer_reviews['review_cleaned'] = customer_reviews['review'].apply(preprocess_text)


# Assuming data is loaded and preprocessed
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(customer_reviews)
clf = LogisticRegression()
clf.fit(X, sentiment_labels)
