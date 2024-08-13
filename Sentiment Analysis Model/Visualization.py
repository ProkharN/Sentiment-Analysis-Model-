"""
Created on  09.09.2023

@author: Prokhar Navitski
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import numpy as np

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Assuming normalized_corpus is a list of text reviews
compound_scores = []

# Calculate sentiment scores
for review_text in normalized_corpus:
    sentiment_scores = sentiment_analyzer.polarity_scores(review_text)
    compound_score = sentiment_scores['compound']
    compound_scores.append(compound_score)

# Create a DataFrame with compound scores
import pandas as pd
df = pd.DataFrame({'Review Index': range(len(normalized_corpus)), 'Compound Score': compound_scores})

# Define color map for positive (green) and negative (red) sentiments
colors = ['green' if score >= 0 else 'red' for score in compound_scores]

# Define transparency levels based on data density
density = np.histogram(compound_scores, bins=20, range=(-1, 1))[0]
density = (density - min(density)) / (max(density) - min(density))
alpha = 0.3 + 0.7 * density  # Adjust transparency between 0.3 and 1.0 based on density

# Create an interactive scatter plot with Plotly
fig = px.scatter(
    df, x='Review Index', y='Compound Score',
    color=colors, opacity=alpha,
    color_discrete_map={'green': 'green', 'red': 'red'},
    labels={'Compound Score': 'Sentiment Score'},
    title='Interactive Sentiment Analysis of Reviews with Color Density'
)

# Customize hover text
fig.update_traces(
    customdata=df['Review Index'],
    hovertemplate='Review Index: %{customdata}<br>Compound Score: %{y:.2f}',
    selector=dict(type='scatter', mode='markers')
)

# Update layout for better interactivity
fig.update_layout(
    xaxis_title='Review Index',
    yaxis_title='Compound Sentiment Score',
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=True),
    yaxis=dict(showgrid=True),
)

# Show the interactive plot
fig.show()
