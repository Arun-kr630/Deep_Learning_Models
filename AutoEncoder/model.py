import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


transform=transforms.ToTensor()
mnist=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
data_loader=torch.utils.data.DataLoader(mnist,batch_size=8,shuffle=True)


image_shape=(1,28,28)
num_epoch=10
lr=3e-4


class Linear_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3)
        )

        self.decoder=nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x


class Image_AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=8,kernel_size=7),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,out_channels=4,kernel_size=9),

        )
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(in_channels=4,out_channels=8,kernel_size=9),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8,out_channels=16,kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=3),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x




model=Image_AutoEncoder()
#model=Linear_Autoencoder()

criterian=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=lr)

for epoch in range(num_epoch):
    loop=tqdm(data_loader,leave=False,total=len(data_loader))
    for img,label in loop:
        x_hat=model(img)
        loss=criterian(img,x_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
              
