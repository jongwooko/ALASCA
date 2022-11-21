import numpy as np
from PIL import Image
import copy

import torchvision
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault


# Parameters for data
cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

# Augmentations.
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

transform_strong = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
])

class TransformTwice:
    def __init__(self, transform, transform2, multiplicity=None):
        self.transform = transform
        self.transform2 = transform2
    
    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

def get_cifar10(root, percent, noise_type, transform_train=transform_train, transform_val=transform_val, download=True):
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    
    train_idxs, val_idxs = train_val_split(base_dataset.targets)
    train_dataset = CIFAR10_train(root, train_idxs, percent, train=True, transform=TransformTwice(transform_train, transform_strong))
    test_dataset = CIFAR10_val(root, transform=transform_val)
    
    if noise_type == 'sym':
        train_dataset.symmetric_noise()
    elif noise_type == 'asym':
        train_dataset.asymmetric_noise()
    else:
        train_dataset.instance_noise()
    
    print (f"#Train: {len(train_dataset)} #Test: {len(test_dataset)}")
    print (f"Observed Labels: {train_dataset.targets[:10]} True Labels: {train_dataset.targets_gt[:10]}")
    print (f"Noise Ratio: {(train_dataset.targets != train_dataset.targets_gt).sum() / len(train_dataset)} Noise Type: {noise_type.upper()}")
    
    return train_dataset, test_dataset

def train_val_split(base_dataset: torchvision.datasets.CIFAR10):
    num_classes = 10
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

def noisify_instance(train_data, train_labels, noise_rate):
    if max(train_labels) > 10:
        num_class = 100
    else:
        num_class = 10
        
    np.random.seed(0)
    q_ = np.random.normal(loc=noise_rate, scale=0.1, size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q) == 45000:
            break
            
    w = np.random.normal(loc=0, scale=1, size=(32*32*3, num_class))
    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = np.array(sample).flatten()
        p_all = np.matmul(sample, w)
        p_all[train_labels[i]] = -1000000
        p_all = q[i]* F.softmax(torch.tensor(p_all),dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class),p=p_all/sum(p_all)))
    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/50000
    return noisy_labels, over_all_noise_rate

class CIFAR10_train(torchvision.datasets.CIFAR10):
    def __init__(self, root, indexs=None, percent=0.0,
                 train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                    transform=transform, target_transform=target_transform,
                    download=download)
        
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        else:
            self.targets = np.array(self.targets)
        self.data = [Image.fromarray(img) for img in self.data]
        self.percent = percent
        self.noise_indx = []
        self.indexs = indexs
        self.num_classes = 10
        
    def symmetric_noise(self):
        self.targets_gt = copy.deepcopy(self.targets)
        indices = np.random.permutation(len(self.data))
        for i, idx in enumerate(indices):
            if i < self.percent * len(self.data):
                self.noise_indx.append(idx)
                noise_target = np.random.randint(self.num_classes, dtype=np.int32)
                self.targets[idx] = noise_target
                
    def asymmetric_noise(self):
        self.targets_gt = copy.deepcopy(self.targets)
        for i in range(self.num_classes):
            indices = np.where(self.targets==i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.percent * len(indices):
                    self.noise_indx.append(idx)
                    # truck -> automobile
                    if i == 9:
                        self.targets[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.targets[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.targets[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.targets[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.targets[idx] = 7
                        
    def instance_noise(self):
        self.targets_gt = copy.deepcopy(self.targets)
        self.targets, _ = noisify_instance(self.data, self.targets, self.percent)
                
    def __getitem__(self, index):
        img, target, target_gt = self.data[index], self.targets[index], self.targets_gt[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, index, target_gt
    
class CIFAR10_val(CIFAR10_train):
    def __init__(self, root, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_val, self).__init__(root, indexs=None, percent=None,
                 train=False, transform=transform, target_transform=target_transform,
                 download=download)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, index