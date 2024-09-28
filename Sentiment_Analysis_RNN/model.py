import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Sentiment_data
from tqdm import tqdm
num_epochs=50
lr=3e-4
embed_size=768
hidden_size=256

class SentimentModel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.RNN(input_size=embed_size,hidden_size=hidden_size,batch_first=True)
        self.linear1=nn.Linear(hidden_size,64)
        self.relu1=nn.ReLU()
        self.relu2=nn.ReLU()
        self.relu3=nn.ReLU()
        self.linear2=nn.Linear(64,32)
        self.linear3=nn.Linear(32,16)
        self.linear4=nn.Linear(16,3)

    def forward(self,x):
        input_embedding=self.embedding(x)
        output,_=self.rnn(input_embedding)
        last_output=output[:,-1,:]
        last_output=self.linear1(last_output)
        last_output=self.relu1(last_output)
        last_output=self.linear2(last_output)
        last_output=self.relu2(last_output)
        last_output=self.linear3(last_output)
    
        last_output=self.relu3(last_output)
        return self.linear4(last_output)

data_set=Sentiment_data()
vocab_size=data_set.vocab_length()
datalaoder=DataLoader(data_set,batch_size=1,shuffle=True)
model=SentimentModel(vocab_size,embed_size,hidden_size)
criterian=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(num_epochs):
    loop=tqdm(datalaoder,leave=False,total=len(datalaoder),desc=f"Processing Epoch {epoch:02d}")
    for sentiment,label in loop:
        pred=model(sentiment)
        loss=criterian(pred,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
