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

# input:  feature (dim*n)_*60000
# output: typical data (600)

class KMEANS:
    def __init__(self,device):

        self.n_cluster = 200
        self.sample_in_cl = 3
        self.max_iter = 20
        self.per_change = 1
        self.count = 0
        self.centers = None
        self.label = None
        self.device = device
        self.pre_dis = 1
        self.n_train = 60000

    # cluster into n_cluster classes
    def cluster(self, x):
        init_row = torch.randint(0, x.shape[0], [self.n_cluster]).to(self.device)
        init_points = x[init_row]
        self.centers = init_points

        while True:
            self.nearest_center(x)
            self.calculate_center(x)
            print(self.per_change)
            if self.per_change < 1e-2 or self.max_iter < self.count:
                break
            self.count += 1

    # classify every pc into the nearest center
    def nearest_center(self, x):
        label = torch.empty(x.shape[0]).int().to(self.device)
        sum_dis = torch.zeros(1).to(self.device)
        for i, vec in enumerate(x):
            vec=vec.expand(self.centers.shape)
            dis = torch.mul(vec - self.centers, vec - self.centers)
            dis = torch.sum(dis,dim=1)
            label[i] = torch.argmin(dis)
            min_dis=torch.min(dis)
            sum_dis = torch.add(min_dis,sum_dis)
        self.label = label
        if self.count >0:
            self.per_change = 1-sum_dis/self.pre_dis
        self.pre_dis = sum_dis

    # calculate every center
    def calculate_center(self, x):
        centers = torch.empty([0, x.shape[1]]).to(self.device)
        for i in range(self.n_cluster):
            idx = self.label==i
            vec = x[idx]
            m=torch.mean(vec,dim=0,keepdim=True)
            centers = torch.cat([centers,m],dim=0)
        self.centers = centers

    # choose sample_in_cl samples in every class
    def choose_sample(self, x):
        sample_list = np.zeros([self.n_cluster,self.sample_in_cl])
        dis_list = np.ones([self.n_cluster,self.sample_in_cl])*np.inf
        max_dis_list = np.ones(self.n_cluster)*np.inf
        max_sample_list = np.ones(self.n_cluster).astype(int) 
        label = self.label
        for i in range(self.n_train):
            j = label[i].cpu().numpy()
            dis = torch.sum(torch.mul(x[i]-self.centers[j],x[i]-self.centers[j]))
            if dis < max_dis_list[j]:
                dis_list[j,max_sample_list[j]] = dis
                sample_list[j,max_sample_list[j]] = i
                max_dis_list[j]=dis_list[j,:].max()
                max_sample_list[j]=dis_list[j,:].argmax()
        return sample_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['ae', 'pca'])
    config = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    method = config.method
    if method == 'pca':
        in_dataset = 'pc.npy'
        out_dataset = 'pca_samples.npy'
    elif method == 'ae':
        in_dataset = 'latent_rep.npy'
        out_dataset = 'ae_samples.npy'
    ve=np.load(in_dataset)
    print(ve.shape)

    ve=torch.from_numpy(ve).t().to(device)
    k=KMEANS(device)
    k.cluster(ve)
    sample_list = k.choose_sample(ve)
    np.save(out_dataset,sample_list)




