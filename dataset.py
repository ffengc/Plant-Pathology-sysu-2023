import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os
from data_enhancement import *

class LeafDataset(Dataset): 
    def __init__(self, csv_file, imgs_path, transform=None):
        self.df = pd.read_csv(csv_file) 
        self.imgs_path = imgs_path 
        self.transform = transform 
        self.len = self.df.shape[0] 
    def __len__(self):
        return self.len
    def __getitem__(self, index): 
        row = self.df.iloc[index]
        image_path = self.imgs_path + row[0]
        image = torchvision.io.read_image(image_path).float()
        target = torch.tensor(row[-6:], dtype=torch.float)
        if self.transform:
            return self.transform(image), target
        image = enhance(image) # 数据增强
        return image, target



class TestDataSet(Dataset):

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = os.listdir("test_images/")

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        name = self.total_imgs[idx]
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = torchvision.io.read_image(img_loc).float()
        if self.transform:
            return self.transform(image), name
        image = enhance(image) # 数据增强
        return image, name

'''
这个例子中的RandAugment(N=2, M=9)表示我们将随机选择2种增强操作（例如旋转、剪切、色彩抖动等）并按顺序应用，
每种操作的强度（例如旋转的角度、剪切的比例等）由参数M决定，M的值越大，操作的强度越大。
在实际使用时，你可能需要根据你的任务和数据集来调整N和M的值。
此外，你也可以尝试其他的数据增强方法，例如Mixup、CutMix等，这些方法都已经被证明在许多任务中都能有效地提升模型的性能。
'''