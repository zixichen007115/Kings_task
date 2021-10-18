import sys
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

# x = torch.rand((20,60)).numpy()

# np.save('number',x)

# x=np.load('number.npy')
# print(np.size(x))
# a = np.load('ae.npy')
# print(a.shape)
a='pc'
b='.npy'
c=a+b
print(c)