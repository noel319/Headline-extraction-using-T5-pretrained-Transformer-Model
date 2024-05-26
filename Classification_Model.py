import nltk
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resuordes
nltk.download('punkt')
nltk.download('stopwords')
#read the dataset

train = pd.read_csv('Classification/train.csv')
validation = pd.read_csv('Classification/validation.csv')
test = pd.read_csv('Classification/test.csv')

# Load the spacy Engilsh model
# Since we are not using NER, we can disable it to speedup
nlp = spacy.load("en_core_web_sm", disable='ner')

def preprocess_text(texts):
    # lemmatize the tokens and store them in a list
    processed_texts = []
    for doc in nlp.pipe(texts, n_process=-1):
        lemmatized_tokens = [ token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_ not in nlp.Defaults.stop_words]
        # Join the lemmatized tokens into a string
        processed_text = " ".join(lemmatized_tokens)
        processed_texts.append(preprocess_text)
    return processed_texts

# apply preprocess_text function to user_review column

train['user_review'] = preprocess_text(train['user_review']) 
validation['user_review'] = preprocess_text(validation['user_review']) 
test['user_review'] = preprocess_text(test['user_review']) 

# Vectorization

count_vectorizer_one = CountVectorizer(min_df=0.001, binary=True)