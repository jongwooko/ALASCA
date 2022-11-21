import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault
import copy

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

def get_cifar100(root, percent, noise_type, transform_train=transform_train, transform_val=transform_val, download=True):
    base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
    
    train_idxs, val_idxs = train_val_split(base_dataset.targets)
    train_dataset = CIFAR100_train(root, train_idxs, percent, train=True, transform=TransformTwice(transform_train, transform_strong))
    test_dataset = CIFAR100_val(root, transform=transform_val)
    
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
    num_classes = 100
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

class CIFAR100_train(torchvision.datasets.CIFAR100):
    def __init__(self, root, indexs=None, percent=0.0,
                 train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_train, self).__init__(root, train=train,
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
        self.num_classes = 100
        
    def symmetric_noise(self):
        self.targets_gt = copy.deepcopy(self.targets)
        indices = np.random.permutation(len(self.data))
        for i, idx in enumerate(indices):
            if i < self.percent * len(self.data):
                self.noise_indx.append(idx)
                self.targets[idx] = np.random.randint(self.num_classes, dtype=np.int32)
            
    def multiclass_noisify(self, y, P, random_state=0):
        """
        Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """
        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]
        
        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y
    
    
    def build_for_cifar100(self, size, noise):
        """
        The noise matrix flips to the "next" class with probability 'noise'.
        """
        assert (noise >= 0.) and (noise <= 1.)
        P = (1. - noise) * np.eye(size)
        for i in np.arange(size - 1):
            P[i, i+1] = noise
            
        # adjust last row
        P[size-1, 0] = noise
        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P
        
    def asymmetric_noise(self):
        self.targets_gt = copy.deepcopy(self.targets)
        P = np.eye(self.num_classes)
        nb_superclasses = 20
        nb_subclasses = 5
        n = self.percent # We conduct only asymmetric noise as 0.4
        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)
                
            y_train_noisy = self.multiclass_noisify(self.targets, P=P,
                                               random_state=0)
            actual_noise = (y_train_noisy != self.targets).mean()
            assert actual_noise > 0.0
            self.targets = y_train_noisy
            
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
        
class CIFAR100_val(CIFAR100_train):
    def __init__(self, root, transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_val, self).__init__(root, indexs=None, percent=None,
                 train=False, transform=transform, target_transform=target_transform,
                 download=download)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, index