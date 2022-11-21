import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

# Augmentations.
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
])


def get_clothing1M(root, batch_size, num_batches, train=True):
    if train:
        train_dataset = Clothing1M_Dataset(root, transform=transform_train, mode='train', num_samples=batch_size*num_batches)
        valid_dataset = Clothing1M_Dataset(root, transform=transform_val, mode='valid')
        print(f"Train: {len(train_dataset)} Val: {len(valid_dataset)}")
        return train_dataset, valid_dataset
    else:
        test_dataset = Clothing1M_Dataset(root, transform=transform_val, mode='test')
        print(f"Test: {len(test_dataset)}")
        return test_dataset
       
class Clothing1M_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode, num_samples=0, num_class=14):
        
        self.root = root
        self.transform = transform
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        
        with open('%s/annotations/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])                         
        with open('%s/annotations/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])
                
        if mode=='train':
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)
        elif mode=='valid':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.val_imgs.append(img_path)
        elif mode=='test':
            self.test_imgs = []
            with open('%s/annotations/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.root+l[7:]
                    self.test_imgs.append(img_path)
                    
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
        elif self.mode=='valid':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            
        image = Image.open(img_path).convert('RGB')
        
        if self.mode=='train':
            img0 = self.transform(image)
            return img0, target, index, target
        else:
            img0 = self.transform(image)
            return img0, target, index
        
    def __len__(self):
        if self.mode=='train':
            return len(self.train_imgs)
        elif self.mode=='valid':
            return len(self.val_imgs)
        elif self.mode=='test':
            return len(self.test_imgs)
        
def test():
    x, y = get_clothing1M('/home/work/KAIST-OSI-JONGWOO/dataset/clothing1m/', batch_size=64, num_batches=1000, train=True)
    return x, y

def test2():
    x = get_clothing1M('/home/work/KAIST-OSI-JONGWOO/dataset/clothing1m/', batch_size=64, num_batches=1000, train=False)
    return x