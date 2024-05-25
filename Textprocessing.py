import nltk
import pandas as pd
#download the necessary models for each task
nltk.download('punkt') # Download the tokenizer models
nltk.download('wordnet') #download wordNet, required for semantic analysis for lemmatization
nltk.download('stopwords') 
#nltk.download('average_perceptron_tragger') #Download POS tagger
nltk.download('omw-1.4') #Download the wordnet OMW corpus
doc1 = pd.read_csv("TextProcessing/doc1.txt")
doc2 = pd.read_csv("TextProcessing/doc2.txt")
doc3 = pd.read_csv("TextProcessing/doc3.txt")
corpus = [doc1, doc2, doc3]

# Tokenization

from nltk.tokenize import word_tokenize
tokenized_docs=[]
for doc in corpus:
    tokens = word_tokenize(str(doc))
    tokenized_docs.append(tokens)

#printing the all the tokens for all documents and the token lenghts for each document
# print('\nTokenized Documents:')
# for i, doc in enumerate(tokenized_docs):
#     print(f"Docunent {i+1} Tokens:", doc[:], "...(Total:", len(doc), ")")
#     print() # add an empty print statement for a new line

# Lowercasing

lowercased_docs =[]
for tokens in tokenized_docs:
    lowercased_tokens = [token.lower() for token in tokens]
    lowercased_docs.append(lowercased_tokens)

# Print the tokenized and lowercased documents

# print("\nTokenized and Lowercased Documents:")
# for i, doc in enumerate(lowercased_docs):
#     print(f"Document {i+1} Tokens:", doc, "...(Total:", len(doc), ")\n")

# Removeing Punctuations

no_punctuation_docs = []
for tokens in lowercased_docs:
    filtered_tokens = [token for token in tokens if token.isalpha()] # Remove punctuation
    no_punctuation_docs.append(filtered_tokens)

# Print the documents without punctuation, along with their token counts

# print("\nDocuments with Tokens Lowercased and Punctuation Removed:")
# for i , doc in enumerate(no_punctuation_docs):
#     print(f"Document {i+1} Tokens:", doc, "... (Total:", len(doc), ")\n")

# Stopwords Removal

#stop words Removal

from nltk.corpus import stopwords
#Get the Engilsh stopwords
stop_words = set(stopwords.words('english'))
# Remove stopwords
no_stopwords_docs=[]
for tokens in no_punctuation_docs:
    filtered_tokens = [token for token in tokens if token not in stop_words]
    no_stopwords_docs.append(filtered_tokens)
# Add 'also' to the set of stopwords
stop_words.add('also')
# Remove stopwords again, now including 'also'
no_stopwords_including_also_docs=[]
for tokens in no_stopwords_docs:#we'll start from the already stopword-removed docs
    filtered_tokens = [token for token in tokens if token not in stop_words]
    no_stopwords_including_also_docs.append(filtered_tokens)

# Print the documents without stopwords, along with their token counts
# print("\nDocuments with Tokens Lowercased, Punctuation Removed, and Stopwords Removed:")
# for i, doc in enumerate(no_stopwords_including_also_docs):
#     print(f"Document {i+1} Tokens:", doc, "...(Total:", len(doc),")\n")
    
# Stemming

from nltk.stem import PorterStemmer
#Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Apply stemming to the documents
stemmed_docs=[]
for tokens in no_stopwords_including_also_docs:
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_docs.append(stemmed_tokens)
# Print the stemmed documents, along with their token counts
# print("\nDocuments with Tokens Lowercased, Punctuation Removed, Stopwords Including 'also' Removed, and Stemmed:")
# for i, doc in enumerate(stemmed_docs):
#     print(f"Document {i+1} Tokens:", doc, "...(Total:", len(doc), ")\n")

# Lemmatization   
from nltk.stem import WordNetLemmatizer
# Initialize the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
# Apply lemmatization to the documents
lemmatized_docs = []
for tokens in stemmed_docs:
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_docs.append(lemmatized_tokens)

#pring the lemmatized documents, along with their token counts
# print("\nDocuments with Tokens Lowercased, Punctuation Removed, Stopwords Including 'also' Removed, and Lemmatized:")
# for i, doc in enumerate(lemmatized_docs):
#     print(f"Document {i+1} Tokens:", doc, "...(Total:", len(doc), ")\n")

# Install spacy download en_core_web_sm
import spacy
# Loading the English language model with lemmatization capabilities
nlp = spacy.load("en_core_web_sm")
#Apply lemmatization using spaCy

spacy_lemmatized_docs =[]
for doc in no_stopwords_including_also_docs:
    #Process the document using spaCy
    spacy_doc = nlp(' '.join(doc))
    # Extract lemmatized tokens
    spacy_lemmatized_tokens=[token.lemma_ for token in spacy_doc]
    spacy_lemmatized_docs.append(spacy_lemmatized_tokens)

# Print the lemmatized documents using spaCy, along with their token counts
# print("\nDocuments Lemmatized using spaCy:")
# for i, doc in enumerate(spacy_lemmatized_docs):
#     print(f"Document {i+1} Tokens:", doc, "... (Total:", len(doc), ")\n")

#Vocabulary Building

#Initialize an empty set to store the vocabulary
vocabulary = set()

#Iterate over each document and add each unique word to the vocabulary set
for doc in spacy_lemmatized_docs:
    for token in doc:
        vocabulary.add(token)
# Convert the set to a sorted list if you want the vocabulary to be orderd
vocabulary = sorted(list(vocabulary))
#Print the vocabulary
print("Vocabulary:")
print(vocabulary)
print(f"Vocabulary Size: {len(vocabulary)}")

