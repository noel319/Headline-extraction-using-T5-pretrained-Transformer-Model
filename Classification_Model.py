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
        processed_texts.append(processed_text)
    return processed_texts

# apply preprocess_text function to user_review column


#model
 # naive Bayes classifier
naive_bayes_classifier_bernoulli = BernoulliNB()
#naive Bayes classifier
naive_bayes_classifier_multinomial = MultinomialNB()
# Vectorization
def build_model_ohe():
    count_vectorizer_ohe = CountVectorizer(min_df=0.001, binary=True)

    # fit_transform user_review
    count_vectorizer_ohe_train = count_vectorizer_ohe.fit_transform(train['user_review'])

    # Building a naive Bayes Model

   
    # Create the naive bayes model for the train data
    naive_bayes_classifier_bernoulli.fit(count_vectorizer_ohe_train, train['user_suggestion'])
    num = naive_bayes_classifier_bernoulli.score(count_vectorizer_ohe_train, train['user_review'])
    print("naive_bayes_classifier_bernoulli train:\n",num)
    ##create the naive bayes model for the validation data
    count_vectorizer_ohe_val = count_vectorizer_ohe.transform(validation['user_review'])
    num = naive_bayes_classifier_bernoulli.score(count_vectorizer_ohe_val, validation['user_suggestion'])
    print("naive_bayes_classifier_bernoulli validation:\n", num)
# Count Vectorizer

def build_model_countvector():
    # initialize count_vectorizer and name it count_vectorizer
    count_vectorizer = CountVectorizer(min_df=0.001)

    #fit_transform user_review
    count_vectorizer_train = count_vectorizer.fit_transform(train['user_review'])

    #Buliding a naive Bayes Model using count vectorization
    
    #create the nvvie bayes model forthe train data
    naive_bayes_classifier_multinomial.fit(count_vectorizer_train, train['user_suggestion'])
    num = naive_bayes_classifier_multinomial.score(count_vectorizer_train, train['user_suggestion'])
    print("assifier_multinomial train:\n",num)
    ##create the naive bayes model for the validation data
    count_vectorizer_val = count_vectorizer.transform(validation['user_review'])
    num = naive_bayes_classifier_multinomial.score(count_vectorizer_val, validation['user_suggestion'])
    print("naive_bayes_classifier_multinomial validation:\n",num)

def fun_tfidf():
    #TF-IDF
    # import TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    # initialize tfifd vectorizer
    tfidf_vectorizer = TfidfVectorizer(min_df = 0.001)
    #create the naive nabyes model for the train data using tfidf
    tfidf_vectorizer_train = tfidf_vectorizer.fit_transform(train['user_review'])
    naive_bayes_classifier_multinomial.fit(tfidf_vectorizer_train, train['user_suggestion'])
    num = naive_bayes_classifier_multinomial.score(tfidf_vectorizer_train, train['user_suggestion'])
    print("naive_bayes_classifier_multinomial train:\m", num)
    #create the naive bayes model for the validation data using tfidf

    tfidf_vectorizer_val = tfidf_vectorizer.transform(validation['user_review'])
    num = naive_bayes_classifier_multinomial.score(tfidf_vectorizer_val, validation['user_suggestion'])
    print("naive_bayes_classifier_multinomial validation:\n", num)
    # Using n-grams with TfIdf
def fun_build_ngram():

    tfidf_ngram_vectorizer = TfidfVectorizer(min_df=0.001, ngram_range=(1,3))

    #create the naive bayes model for the train data using tfidf and ngram
    tfidf_ngram_vectorizer_train = tfidf_ngram_vectorizer.fit_transform(train['user_review'])
    naive_bayes_classifier_multinomial.fit(tfidf_ngram_vectorizer_train, train['user_suggestion'])
    num = naive_bayes_classifier_multinomial.score(tfidf_ngram_vectorizer_train, train['user_suggestion'])
    print("naive_bayes_classifier_multinomial train:\n", num)

    ss = tfidf_ngram_vectorizer.get_feature_names_out()[160:163]
    print(ss)
    #create the naive bayes model for the validation data using tfidf and ngram
    tfidf_ngram_vectorizer_val = tfidf_ngram_vectorizer.transform(validation['user_review'])
    naive_bayes_classifier_multinomial.score(tfidf_ngram_vectorizer_val, validation['user_suggestion'])

    count_ngram_vectorizer = CountVectorizer(min_df=0.001, ngram_range=(1,3))

    #create the naive bayes model for the train data using count vectorizer and ngram
    count_ngram_vectorizer_train = count_ngram_vectorizer.fit_transform(train['user_review'])
    naive_bayes_classifier_multinomial.fit(count_ngram_vectorizer_train, train['user_suggestion'])
    naive_bayes_classifier_multinomial.score(count_ngram_vectorizer_train, train['user_suggestion'])

    #create the naive bayes model for the validation data using count vectorizer and ngram

    count_ngram_vectorizer_val = count_ngram_vectorizer.transform(validation['user_review'])
    naive_bayes_classifier_multinomial.score(count_ngram_vectorizer_val, validation['user_suggestion'])



# POS Tagging and NER
# Load spaCy model
nlp = spacy.load("en_core_web_sm")
def preprocess_text_spacy(processed_texts):
    # Tokenization and POS tagging
    pos_texts = []
    for doc in nlp.pipe(processed_texts):
        pos_tags = [token.pos_ for token in doc]
        pos_text = " ".join(pos_tags)
        pos_texts.append(pos_text)

    # Named Entity Recognition (NER)
    ner_texts = []
    for doc in nlp.pipe(processed_texts):     
        ner_tags = [token.ent_type_ if token.ent_type_ else "O" for token in doc]
        ner_text = " ".join(ner_tags)
        ner_texts.append(ner_text)
    
    return [pos_texts, ner_texts]

# applying preprocess_text_spacy function touser_review column for train data
pos_texts, ner_texts = preprocess_text_spacy(train['user_review'])
# adding the lists as column to the dataset
train['pos_tags'] = pos_texts
train['ner_tags'] = ner_texts

train.head()

from spacy import displacy

text= train['user_review'][3]
doc = nlp(text)
displacy.render(doc, style="ent")

text1= "India is a country with leading IT companies such as Infosys, TCS, Wipro etc. Most of them make millions of dollars in revenues and are based out of Hyderabad"
text1
doc = nlp(text1)
displacy.render(doc, style="ent")
del train['pos_tags']
del train['ner_tags']
def remove_noun(df):

  nlp = spacy.load("en_core_web_sm")  

  # Process user_review column
  filtered_reviews = []
  for review in df['user_review']:
    filtered_review = " ".join([token.text for token in nlp(review) if token.pos_ not in ['NOUN', 'PROPN']])
    filtered_reviews.append(filtered_review)
  
  return filtered_reviews



# Main
if __name__ == '__main__':
    train['user_review'] = preprocess_text(train['user_review']) 
    validation['user_review'] = preprocess_text(validation['user_review']) 
    test['user_review'] = preprocess_text(test['user_review'])     
    fun_build_ngram()