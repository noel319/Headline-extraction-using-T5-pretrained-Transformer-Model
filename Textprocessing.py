import nltk
import pandas as pd
#download the necessary models for each task
nltk.download('punkt') # Download the tokenizer models
nltk.download('wordnet') #download wordNet, required for semantic analysis for lemmatization
nltk.download('stopwords') 
#nltk.download('average_perceptron_tragger') #Download POS tagger
nltk.download('omw-1.4') #Download the wordnet OMW corpus
