# Building an RNN MOdel with Word Indexing

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    #read the dataset
    train = pd.read_csv('Classification/train.csv')
    validation = pd.read_csv('Classification/validation.csv')
    test = pd.read_csv('Classification/test.csv')

    import re

    def preprocess_text(text):
        #check if the text is a string
        if not isinstance(text, str):
            return []
        #Keep only letters and whitespaces
        pattern = f"[a-zA-Z\s]"
        text = ''.join(re.findall(pattern, text))

        # convert to lowercase
        text = text.lower()

        #Tokenize the text
        tokens = nltk.word_tokenize(text)
        #tokens = ' '.join(tokens)
        return tokens

    #apply the preprocess text to
    train['user_review'] = train['user_review'].apply(preprocess_text)
    validation['user_review'] = validation['user_review'].apply(preprocess_text)
    test['user_review'] = test['user_review'].apply(preprocess_text)

    def build_vocabulary(reviews):
        vocab = {}
        index = 1 # start indexing from 1; reserve 0 for padding
        for review in reviews:
            for word in review:
                if word not in vocab:
                    vocab[word] = index
                    index +=1
        return vocab
    # Concatenate all reviews to build the vocabulary
    all_reviews = train['user_review'].tolist()+validation['user_review'].tolist() + test['user_review'].tolist()
    vocab = build_vocabulary(all_reviews)
    print("Vocabulary Length:", len(vocab))
    first_50 = list(vocab.items())[:50]
    for key, value in first_50:
        print(f'{key} :{value}')