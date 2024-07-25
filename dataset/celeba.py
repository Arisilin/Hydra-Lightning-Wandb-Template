import torch.utils.data as data
import torch
import os
from torchvision import datasets
from utils.dataset_transforms import read_transforms
import wandb
from PIL import Image

class Celeba(data.Dataset):
    
    def __init__(self, root, transform=None, mode=None, train=True):
        self.root = root
        self.transform = read_transforms(transform)
        self.mode = mode
        self.train = train
        self.train_size = 0.8
        self.images = os.listdir(root)
        self.images = [img for img in self.images if img.endswith('.jpg')]
        if self.train == False:
            self.images = self.images[::10]
        # self.load_to_mem(root)
        # print(f"Image pixels: {Image.open(os.path.join(root,self.images[0])).size}")
        
    
    # def load_to_mem(self,root):
    #     if self.train == True:
    #         train_dataset = datasets.ImageFolder(root,transform=self.transform)
    #         # ftrain_dataset = datasets.ImageFolder(os.path.join(root,t,f),transform=self.transform)
    #         # mtrain_dataset = datasets.ImageFolder(os.path.join(root,t,m),transform=self.transform)
    #         self.data = train_dataset
    #     if self.train == False:
    #         val_dataset = datasets.ImageFolder(root,transform=self.transform)
    #         # mval_dataset = datasets.ImageFolder(os.path.join(root,v,m),transform=self.transform)
    #         # fval_dataset = datasets.ImageFolder(os.path.join(root,v,f),transform=self.transform)
    #         self.data = val_dataset

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image