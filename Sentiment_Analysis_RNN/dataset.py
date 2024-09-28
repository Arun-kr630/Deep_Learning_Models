import pandas as pd 
from torch.utils.data import Dataset,DataLoader
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

nltk.download('punkt')
from nltk.tokenize import word_tokenize
class Sentiment_data(Dataset):
    def __init__(self):
        super().__init__()
        dataset=pd.read_csv("sentiment_dataset/train.csv", encoding='ISO-8859-1')
        X=list(dataset.iloc[:,1].values)
        Y=list(dataset.iloc[:,3].values)
        X_tokenized = [word_tokenize(sentence.lower()) for sentence in X]
        word2idx = defaultdict(lambda: len(word2idx))
        word2idx['<PAD>'] = 0  
        self.X_indices = [[word2idx[word] for word in sentence] for sentence in X_tokenized]
        all_words = [word for sentence in X_tokenized for word in sentence]
        vocab = set(all_words)
        self.vocab_size = len(vocab)
        label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        Y_numeric = [label_mapping[label] for label in Y]
        self.Y=torch.tensor(Y_numeric)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self,index):
        return torch.tensor(self.X_indices[index]),self.Y[index]
    def vocab_length(self):
        return self.vocab_size



