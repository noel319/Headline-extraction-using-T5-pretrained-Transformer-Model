import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read the dataset
train = pd.read_csv('Classification/train.csv')
validation = pd.read_csv('Classification/validation.csv')
test = pd.read_csv('Classification/test.csv')

import re

def preprocess_text(text):
    # Check if the text is a string
    if not isinstance(text, str):
        return[]
    # Keep only letters and whitespaces
    pattern = f"[a-zA-Z\s]"
    text = ''.join(re.findall(pattern, text))
    # Convert to lowercasee
    text = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    return tokens
# apply the preprodcess text to
train['user_review'] = train.user_review.apply(preprocess_text)
validation['user_review'] = validation.user_review.apply(preprocess_text)
test['user_review'] = test.user_review.apply(preprocess_text)

# Fetch embedding 
word2vec_model = Word2Vec(sentences=train.user_review.values.tolist(), vector_size=100, min_count=1, workers=4)

# Get vocabulary size
vocab_size = len(word2vec_model.wv)
print(vocab_size)

# Convert text to Word2Vec embeddings
def text_to_embeddings(text, word2vec_model, seq_length):
    embeddings = []
    
    for i, word in enumerate(text):
        if word in word2vec_model.wv:
            if i == seq_length:
                break
            embeddings.append(word2vec_model.wv[word])
        else:
            continue
        
    # Padding the sequences
    if len(embeddings) < seq_length:
        zero_padding = [np.zeros(word2vec_model.vector_size) \
                        for _ in range(seq_length - len(embeddings))]

        embeddings = embeddings + zero_padding

    return embeddings[:seq_length]

# Prepare data
def prepare_data(reviews, labels, word2vec_model):
    X = [text_to_embeddings(review, word2vec_model, 100) for review in reviews]
    X = [torch.tensor(embeddings, dtype=torch.float32) for embeddings in X]
    y = torch.tensor(labels, dtype=torch.long)
    return X
# Prepare data
X_train, y_train = prepare_data(train.user_review, train.user_suggestion,
                    word2vec_model)

X_val, y_val = prepare_data(validation.user_review, validation.user_suggestion,
                    word2vec_model)

# Define hyperparameters
input_size = word2vec_model.vector_size
hidden_size = 128
output_size = 1
num_layers = 1
learning_rate = 0.001
num_epochs = 30
batch_size = 64
dropout_rate = 0.5
leaky_relu_slope = 0.1
# Create DataLoader
train_data = TensorDataset(torch.stack(X_train), y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# Create DataLoader
val_data = TensorDataset(torch.stack(X_val), y_val)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(SentimentRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Basic RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.leaky_relu = nn.LeakyReLU()  # Leaky ReLU activation layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation layer

    def forward(self, x):
        # Initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # RNN output
        out, _ = self.rnn(x, h0)
        # Apply Leaky ReLU to the outputs of the RNN layer
        out = self.leaky_relu(out)
        out = self.dropout(out)
        # Get the last sequence output for classification
        out = out[:, -1, :]
        # Apply the linear layer for the final output
        out = self.fc(out)
        # Apply the sigmoid activation
        out = self.sigmoid(out)
        return out
