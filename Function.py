import json
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords and VADER lexicon (if not downloaded)
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load JSON file
@st.cache(allow_output_mutation=True)
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)
    text = text.lower()
    text = text.translate(translator)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Topic Modeling function
def get_topics(conversations, num_clusters=5):
    texts = [conv['value'] for conv in conversations if conv['from'] == 'gpt']
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_texts)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    
    topic_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    topic_dict = {}
    for i, label in enumerate(topic_labels):
        if label not in topic_dict:
            topic_dict[label] = []
        topic_dict[label].append(conversations[i * 2]['value'])
    
    return topic_dict

# Sentiment Analysis function
def analyze_sentiment(conversations):
    sid = SentimentIntensityAnalyzer()
    sentiments = []
    for conv in conversations:
        text = conv['value']
        sentiment_score = sid.polarity_scores(text)
        if sentiment_score['compound'] >= 0.05:
            sentiment = 'positive'
        elif sentiment_score['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        sentiments.append(sentiment)
    return sentiments

# Load data
data = load_data('conversations.json')

# Topic modeling and sentiment analysis
topics = get_topics(data)
sentiments = analyze_sentiment(data)

# Create DataFrames for Screen 1 and Screen 2
topic_counts = pd.DataFrame.from_dict({f'Topic {k+1}': [len(v)] for k, v in topics.items()}, orient='index', columns=['Count'])
sentiment_counts = pd.DataFrame.from_dict({'Sentiment': sentiments}).value_counts().reset_index(name='Count')

# Streamlit app
st.title('Conversation Analysis App')

# Screen 1: Counts
st.header('Counts')
st.subheader('Topic Counts')
st.table(topic_counts)

st.subheader('Sentiment Counts')
st.table(sentiment_counts)

# Screen 2: Sessions
st.header('Sessions')
st.subheader('Assigned Topic and Sentiment')
for i, conv in enumerate(data):
    if conv['from'] == 'gpt':
        st.write(f"Conversation No: {i//2 + 1}")
        st.write(f"Topic: {list(topics.keys())[i//2]}")  # Assigning topic based on index
        st.write(f"Sentiment: {sentiments[i]}")
        st.write(f"Message: {conv['value']}")
        st.write("---")

