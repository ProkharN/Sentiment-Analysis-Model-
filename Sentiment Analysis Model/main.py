
# # Import necessary dependencies
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
from contractions import contractions_dict
import unicodedata

nlp = spacy.load('en_core_web_sm')
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


# # Cleaning Text - strip HTML
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


# # Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# # Expanding Contractions
def expand_contractions(text, contraction_mapping=None):
    if contraction_mapping is None:
        contraction_mapping = contractions_dict  # Use the default mapping

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())

        if expanded_contraction is not None:
            first_char = match[0]
            expanded_contraction = first_char + expanded_contraction[1:]

        return expanded_contraction or match  # Return the original if not expanded

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# # Removing Special Characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9!\s]', '', text)
    return text


# # Lemmatizing text
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# # Removing Stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# # Normalize text corpus - tying it all together
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True):
    normalized_corpus = []
    i = 0

    for doc in corpus:
        num_doc_total = len(corpus)
        print('PRE processing. Only__ '+str(num_doc_total - i)+' __docs left to process')

        if html_stripping:
            doc = strip_html_tags(doc)

        if accented_char_removal:
            doc = remove_accented_chars(doc)

        if contraction_expansion:
            doc = expand_contractions(doc)

        if text_lower_case:
            doc = doc.lower()


        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)

        if text_lemmatization:
            doc = lemmatize_text(doc)

        if special_char_removal:
            doc = remove_special_characters(doc)

            # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)

        i += 1

    return normalized_corpus

# assign directory for the corpus we are working with
import os

# List of directory paths
directory_paths = ['/Users/prokharnavitski/Desktop/HAUSARB/Corpus/Sentiment Analysis Model/data/aclImdb/train/unsup']

text_corpus = []

for directory_path in directory_paths:
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                text_corpus.append(text)


normalized_corpus = normalize_corpus(text_corpus)

first_two_elements = normalized_corpus[:2]
print(first_two_elements)

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
