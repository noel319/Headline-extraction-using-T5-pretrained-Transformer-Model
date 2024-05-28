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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#read the dataset
train = pd.read_csv('Classification/train.csv')
validation = pd.read_csv('Classification/validation.csv')
test = pd.read_csv('Classification/test.csv')