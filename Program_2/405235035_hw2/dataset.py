from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import scipy.io as sio 
import os
import cv2
import matplotlib.pyplot as plt
import pprint

class TRAIN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.transform = transform
        
        annos = sio.loadmat('./cars_train_annos.mat')
		
        _, self.total_size = annos["annotations"].shape
        self.labels = np.zeros((self.total_size, 5))
		
        for i in range(self.total_size):
            path = annos["annotations"][:,i][0][5][0].split(".")
            id = int(path[0]) - 1
            for j in range(5):
                self.labels[id, j] = int(annos["annotations"][:,i][0][j][0])
            self.y.append(int(self.labels[id,4]))
		 			
        for file in self.root_dir.glob('*'):
            self.x.append(file)			

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        #image = im.crop( (int(self.labels[0]), int(self.labels[1]), int(self.labels[2]), int(self.labels[3])) )
        
        if self.transform:
            image = self.transform(image)

        return image, self.y[index]


class TEST_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []
        self.y = []
        self.transform = transform
        
        annos = sio.loadmat('./cars_test_annos_withlabels.mat')
		
        _, self.total_size = annos["annotations"].shape
        self.labels = np.zeros((self.total_size, 5))
		
        for i in range(self.total_size):
            path = annos["annotations"][:,i][0][5][0].split(".")
            id = int(path[0]) - 1
            for j in range(5):
                self.labels[id, j] = int(annos["annotations"][:,i][0][j][0])
            self.y.append(int(self.labels[id,4]))
		 			
        for file in self.root_dir.glob('*'):
            self.x.append(file)			

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        #image = im.crop( (int(self.labels[0]), int(self.labels[1]), int(self.labels[2]), int(self.labels[3])) )
        
        if self.transform:
            image = self.transform(image)

        return image, self.y[index]
