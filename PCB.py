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
# output: pc(dim*n)200*60000
# extract principal content


# parameters
batch_size=100
image_size=28*28
dim=200

# load dataset
trans = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=0,std=1)
        ])

train_dataset = torchvision.datasets.MNIST(root='./dataset',
                                           train=True,
                                           transform=trans,
                                           download=True)
pca_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# read and reshape data
ve = torch.tensor([])
for image, _ in pca_loader:
    image = image
    image=image.reshape(batch_size,image_size)
    ve=torch.cat([ve,image],dim=0)

# calculate eigenvectors and reduce dim
ve = ve.t()
u,s,v = torch.svd(ve)
ve_pca = torch.mm(v[:dim,:],ve)

print("principal content shape:{}".format(ve_pca.shape))