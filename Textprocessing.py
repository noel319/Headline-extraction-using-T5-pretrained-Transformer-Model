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

print("\nTokenized and Lowercased Documents:")
for i, doc in enumerate(lowercased_docs):
    print(f"Document {i+1} Tokens:", doc, "...(Total:", len(doc), ")\n")
    
