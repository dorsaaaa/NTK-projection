import sys
import os 
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random

class ProductsDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path, delimiter=' ')
        
        self.X = torch.tensor(data[:, :-1], dtype=torch.float32)  
        self.Y = torch.tensor(data[:, -1], dtype=torch.long) 
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        
        inputs = self.X[idx]
        labels = self.Y[idx]
        return inputs, labels



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_products_dataset(num_train_tasks=20, num_test_tasks=10, num_samples=600, split_size = 0.8, seed = 42):
    set_random_seed(seed)
    folder_path = "/nfs/scistore23/chlgrp/dghobadi/tight-pac-bayes/data/MDPR/products"


    dataloaders = []
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        dataset = ProductsDataset(file_path) 
        train_size = int(split_size * len(dataset))
        if train_size >= num_samples:
            train_size = num_samples
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
        
        if train_size == num_samples:
            dataloaders.append({'train': train_loader, 'test':test_loader})

    assert len(dataloaders) >= num_train_tasks+num_test_tasks

    train_loaders, test_loaders = dataloaders[:num_train_tasks], dataloaders[num_train_tasks:num_train_tasks+num_test_tasks]
    return train_loaders, test_loaders
    

    




