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
    pattern = f"[a-zA-z\s]"
    text = ''.join(re.findall(pattern, text))
    # Convert to lowercasee
    text = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    return tokens