import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
import torch

# Augmentation.
transform_train = transforms.Compose([
    transforms.Resize(256), # transforms.Resize(320),
    transforms.RandomCrop(224), # transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])

transform_val = transforms.Compose([
    transforms.Resize(256), # transforms.Resize(320),
    transforms.RandomCrop(224), # transforms.RandomResizedCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])

transform_imagenet = transforms.Compose([
    transforms.Resize(256), # transforms.Resize(320),
    transforms.RandomCrop(224), # transforms.RandomResizedCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])

def get_webvision(root, train=True, num_class=50):
    if train:
        train_dataset = Webvision_Dataset(root=root, transform=transform_train, mode='train', num_class=num_class)
        valid_dataset = Webvision_Dataset(root=root, transform=transform_val, mode='valid', num_class=num_class)
        print(f"Train: {len(train_dataset)} WebVision Val: {len(valid_dataset)}")
        return train_dataset, valid_dataset
    else:
        test_dataset = ImagenetVal(root, transform=transform_imagenet, num_class=num_class)
        print(f"Imagnet Val: {len(test_dataset)}")
        return test_dataset

class ImagenetVal(torch.utils.data.Dataset):
    def __init__(self, root, transform, num_class):
        self.root = root+'imagenet/'
        self.transform = transform
        
        with open(self.root+'imagenet_val.txt') as f:
            lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target
                    
    def __getitem__(self, index):
        img_path = self.val_imgs[index]
        target = self.val_labels[img_path]     
        image = Image.open(self.root+img_path).convert('RGB')   
        img = self.transform(image)

        return img, target, index
    
    def __len__(self):
        return len(self.val_imgs)
    
class Webvision_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, mode, num_class):
        self.root = root
        self.transform = transform
        self.mode = mode
        
        if self.mode=='valid':
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img]=target
        else:
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()    
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    train_imgs.append(img)
                    self.train_labels[img]=target
            self.train_imgs = np.array(train_imgs)
            
    def __getitem__(self, index):
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]   
            image = Image.open(self.root+img_path)
            img0 = image.convert('RGB')
            img0 = self.transform(img0)
            return img0, target, index, target
        elif self.mode=='valid':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]     
            image = Image.open(self.root+'val_images_256/'+img_path).convert('RGB')   
            img = self.transform(image)
            return img, target, index
        
    def __len__(self):
        if self.mode=='train':
            return len(self.train_imgs)
        elif self.mode=='valid':
            return len(self.val_imgs)
        
def test():
    x, y = get_webvision(root='/home/osilab/dataset/webvision/', train=True)
    return x, y
    
def test2():
    x = get_webvision(root='/home/osilab/dataset/webvision/', train=False)
    return x