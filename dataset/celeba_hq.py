import torch.utils.data as data
import torch
import os
from torchvision import datasets
from utils.dataset_transforms import read_transforms
import wandb

f = 'female'
m = 'male'
t = 'train'
v = 'val'

class CelebaHq(data.Dataset):
    
    def __init__(self, root, transform=None, mode=None, train=True):
        self.root = root
        self.transform = read_transforms(transform)
        self.mode = mode
        self.train = train
        self.load_to_mem(root)
        
    
    def load_to_mem(self,root):
        if self.train == True:
            train_dataset = datasets.ImageFolder(os.path.join(root,t),transform=self.transform)
            # ftrain_dataset = datasets.ImageFolder(os.path.join(root,t,f),transform=self.transform)
            # mtrain_dataset = datasets.ImageFolder(os.path.join(root,t,m),transform=self.transform)
            self.data = train_dataset
        if self.train == False:
            val_dataset = datasets.ImageFolder(os.path.join(root,v),transform=self.transform)
            # mval_dataset = datasets.ImageFolder(os.path.join(root,v,m),transform=self.transform)
            # fval_dataset = datasets.ImageFolder(os.path.join(root,v,f),transform=self.transform)
            self.data = val_dataset

    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        return self.data.__getitem__(idx)[0]