import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_dataset
from tqdm import tqdm

num_epochs=100
n_classes=90
lr=3e-4

class CNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=128,kernel_size=8,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=5,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=7),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=4,kernel_size=5),
            nn.ReLU(),
        )
        self.flatten=nn.Flatten(1)
        self.linear1=nn.Linear(4*16*16,4096)
        self.linear2=nn.Linear(4096,2048)
        self.linear3=nn.Linear(2048,512)
        self.linear4=nn.Linear(512,90)

    def forward(self,x):
        x=self.net(x)
        x=self.flatten(x)
        x=x.view(-1,4*16*16)
        x=self.linear1(x)
        x=F.relu(x)
        x=self.linear2(x)
        x=F.relu(x)
        x=self.linear3(x)
        x=F.relu(x)
        x=self.linear4(x)
        return x
    


model=CNET()
criterian=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

data_loader,class_name=get_dataset()
for epoch in range(num_epochs):
    loop=tqdm(data_loader,leave=False,total=len(data_loader),desc=f"Processing Epoch {epoch:02d}")
    for img,label in loop:
        pred=model(img)
        loos=criterian(pred,label)
        loos.backward()
        optimizer.step()
        optimizer.zero_grad()
