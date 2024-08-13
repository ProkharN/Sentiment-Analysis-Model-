"""
Created on  09.09.2023

@author: Prokhar Navitski
"""

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Assuming normalized_corpus is a list of text reviews

for review_text in normalized_corpus:
    sentiment_scores = sentiment_analyzer.polarity_scores(review_text)

    # Determine the sentiment based on the compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0:
        sentiment = 'positive'
    else:
        sentiment = 'negative'

    # Print or store the result
    print(f"Review Text: {review_text}")
    print(f"Sentiment: {sentiment}")
