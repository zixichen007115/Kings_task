import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms as T
from torchvision import datasets
from torch.autograd import Variable
import numpy as np

# input:  MNIST
# output: latent representation(dim*n)256*60000
# train an AE and generate latent representation for every sample

# AE model
class AE(torch.nn.Module):
    def __init__(self):
        super(AE,self).__init__()
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
            torch.nn.Conv2d(64,128,3,2,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128,256,3,2,1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )
        self.deconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256,128,4,2,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128,64,4,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32,16,4,2,1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.deconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16,1,4,2,1),
            torch.nn.Tanh()
        )
    def forward(self, x):
        ve_in = x.view(x.size(0),-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        re = x
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        ve_out = x.view(x.size(0),-1)
        return ve_in, ve_out, re

# training parameters
parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=12)
parser.add_argument('--lr', type=float, default=0.001)
config = parser.parse_args()
print(config)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_batch_size=config.train_batch_size
lr=config.lr

# load dataset
trans = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=0,std=1)
    ])

train_dataset = torchvision.datasets.MNIST(root='./dataset',
                                     train=True,
                                     transform=trans,
                                     download=True)

train_loader = data.DataLoader(train_dataset, batch_size=train_batch_size,
                                              shuffle=False)

# model
AE = AE()
AE = AE.to(device)

# loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(AE.parameters(),lr=lr)

# training process
for i, (img, labels) in enumerate(train_loader):
    img = Variable(img).to(device)
    labels = Variable(labels).to(device)
    ve_in, ve_out , _ = AE(img)
    loss = loss_function(ve_in, ve_out)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 99:
        print("iteration:{}, loss:{}.".format(i+1,loss))

# latent representation generation
rep_list=np.empty((0,256))
for i, (img, labels) in enumerate(train_loader):
    img = Variable(img).to(device)
    labels = Variable(labels).to(device)
    inp, _ , rep = AE(img)
    rep = rep.cpu().detach().numpy()
    rep = rep.reshape(train_batch_size,256)
    rep_list = np.concatenate((rep_list,rep))
    if i % 100 == 99:
        print("iteration:{}.".format(i+1))

# save latent representation
rep_list = np.transpose(rep_list)
np.save('latent_rep.npy',rep_list)

