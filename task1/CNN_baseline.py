import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms as T
from torchvision import datasets
import numpy as np

# input: first 500 samples in MNIST
# output: test acc

# CNN model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,16,3,2,1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.fc = torch.nn.Linear(2*2*64,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x.view(x.size(0),-1))
        return x

# training parameters
parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=12)
parser.add_argument('--test_batch_size', type=int, default=200)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
config = parser.parse_args()
print(config)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batch_size=config.train_batch_size
test_batch_size=config.test_batch_size
epochs=config.epochs
lr=config.lr

# load dataset
train_sample_idx = np.arange(0,500)
np.random.shuffle(train_sample_idx)
sampler_train = torch.utils.data.sampler.SubsetRandomSampler(train_sample_idx)

trans = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=0,std=1)
    ])

train_dataset = torchvision.datasets.MNIST(root='./dataset',
                                     train=True,
                                     transform=trans,
                                     download=True)

train_loader = data.DataLoader(train_dataset, batch_size=train_batch_size,
                                              shuffle=False,
                                              sampler = sampler_train)

test_dataset = torchvision.datasets.MNIST(root='./dataset',
                                     train=False,
                                     transform=trans,
                                     download=True)
test_loader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# model
CNN = CNN()
CNN = CNN.to(device)

# loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNN.parameters(),lr=lr)

# training process
for epoch in range(epochs):
    trainning_loss = 0
    for i, (img, labels) in enumerate(train_loader):
        img = img.to(device)
        labels = labels.to(device)

        output = CNN(img)

        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # test
        if i % 10 == 9:
            acc=0
            for j,(img, labels) in enumerate(test_loader):
                img = img.to(device)
                labels = labels.to(device)

                output = CNN(img)
                pred = torch.argmax(output,dim=1)
                acc=(float((pred==labels).sum())/float(labels.size(0))*1+acc*j)/(j+1)
            print("epoch:", epoch+1, "iteration:", i+1+epoch*50, "acc:%.4f" %acc)
