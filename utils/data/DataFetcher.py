'''
===================
=   DataFetcher   =
===================
'''
import sys
sys.path.append('../..')
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
import random
import datasets as DS
from utils import operator as opt


class DataFetcher(object):
    def __init__(self, dataset:str, root:str, batch_size:int, shuffle_train:bool=True, shuffle_test:bool=False, drop_last_train:bool=True, drop_last_test:bool=False, download:bool=False, resize:int=0, gray_to_rgb:bool=False, is_augmented:bool=False, **kwargs) -> None:
        '''
        @Desc:
            DataFetcher is an object that preprocesses and loads the datasets for training a deep learning model.
            With DataFetcher, one is able to utilize any existing dataset to train a neural network.
        @Params:
            dataset: the name of the dataset, e.g., CIFAR-10;
            root: the root directory of the datasets, e.g., dataset/cifar-10;
            batch_size:
            shuffle_train: Whether to shuffle the training samples;
            shuffle_test: Whether to shuffle the test samples;
            drop_last_train: Whether to drop out the last training batch;
            drop_last_test: Whether to drop out the last test batch;
            download: whether to download the dataset;
            resize:
            gray_to_rgb:
            is_augmented:
        '''
        # Check dataset
        self.valid_datasets = ['mnist', 'MNIST',
            'EMNIST-DIGITS', 'emnist-digits', 'EMNISTDIGITS', 'emnistdigits', 
            'EMNIST-LETTERS', 'emnist-letters', 'EMNISTLETTERS', 'emnistletters',
            'cifar-10', 'cifar10', 'CIFAR-10', 'CIFAR10',
            'cifar-100', 'cifar100', 'CIFAR-100', 'CIFAR100',
            'cifar10ext', 'cifar-10-ext', 'CIFAR10EXT', 'CIFAR-10-EXT'
            'cifar100ext', 'cifar-100-ext', 'CIFAR100EXT', 'CIFAR-100-EXT'
            'cinic10', 'CINIC10', 'cinic-10', 'CINIC-10',
            'fashionmnist', 'FASHIONMNIST'
            'gtsrb', 'GTSRB'
            'imagenette', 'IMAGENETTE',
            'caltech256', 'CALTECH256'
            ]
        assert dataset in self.valid_datasets, f'Choose a valid dataset from the followings:\n {self.valid_datasets}'
        self.train_ds = None
        self.test_ds = None

        # MNIST
        if dataset in ['mnist', 'MNIST']:
            # Set transform
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
            if gray_to_rgb:
                transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)

            # Load dataset
            self.train_ds = datasets.MNIST(root, True, transform, download=download)
            self.test_ds = datasets.MNIST(root, False, transform, download=download)
        
        # EMNIST-Digits
        if dataset in ['emnistdigits', 'EMNISTDIGITS', 'emnist-digits', 'EMNIST-DIGITS']:
            transform = [
                lambda img: transforms.functional.rotate(img, -90),
                lambda img: transforms.functional.hflip(img),
                transforms.ToTensor(),
                transforms.Normalize((0.1733,), (0.3317,)),
            ]
            if gray_to_rgb:
                transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.train_ds = datasets.EMNIST(root, split='digits', train=True, transform=transform, download=download)
            self.test_ds = datasets.EMNIST(root, split='digits', train=False, transform=transform, download=download)
        
        # EMNIST-Letters
        if dataset in ['emnistletters', 'EMNISTLETTERS', 'emnist-letters', 'EMNIST-LETTERS']:
            transform = [
                lambda img: transforms.functional.rotate(img, -90),
                lambda img: transforms.functional.hflip(img),
                transforms.ToTensor(),
                transforms.Normalize((0.1733,), (0.3317,)),
            ]
            if gray_to_rgb:
                transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.train_ds = datasets.EMNIST(root, split='letters', train=True, transform=transform, download=download)
            self.test_ds = datasets.EMNIST(root, split='letters', train=False, transform=transform, download=download)
        
        # FashionMNIST
        if dataset in ['fashionmnist', 'FASHIONMNIST']:
            # Load train set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
            if gray_to_rgb:
                transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.train_ds = datasets.FashionMNIST(root, True, transform, download=download)
            # Load test set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
            if gray_to_rgb:
                transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.test_ds = datasets.FashionMNIST(root, False, transform, download=download)

        # CIFAR-10
        if dataset in ['cifar10', 'cifar-10', 'CIFAR-10', 'CIFAR10']:
            # Load train set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            if is_augmented:
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform = transforms.Compose(transform)
            self.train_ds = datasets.CIFAR10(root, True, transform, download=download)
            
            # Load test set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.test_ds = datasets.CIFAR10(root, False, transform, download=download)
        

        # CINIC-10
        if dataset in ['cinic10', 'cinic-10', 'CINIC10', 'CINIC-10']:
            # Load train set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            if is_augmented:
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform=transforms.Compose(transform)
            self.train_ds = ImageFolder(root=opt.os.join(root, 'train'), transform=transform)
            
            # Load test set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.test_ds = ImageFolder(root=opt.os.join(root, 'test'), transform=transform)
        
        # CIFAR-100
        if dataset in ['cifar100', 'cifar-100', 'CIFAR100', 'CIFAR-100']:
            # Load train set
            transform = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
                ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            if is_augmented:
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform = transforms.Compose(transform)
            self.train_ds = datasets.CIFAR100(root, True, transform, download=download)

            # Load test set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.test_ds = datasets.CIFAR100(root, False, transform, download=download)
        
        # CIFAR-10E
        if dataset in ['cifar10ext', 'cifar-10-ext', 'CIFAR10EXT', 'CIFAR-10-EXT']:
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4715, 0.4701, 0.4249), (0.2409, 0.2352, 0.2593)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.train_ds = datasets.ImageFolder(root, transform)
        
        # CIFAR-100E
        if dataset in ['cifar100ext', 'cifar-100-ext', 'CIFAR100EXT', 'CIFAR-100-EXT']:
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4672, 0.4581, 0.4093), (0.2580, 0.2470, 0.2671)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.train_ds = datasets.ImageFolder(root, transform)

        # GTSRB
        if dataset in ['gtsrb', 'GTSRB']:
            # Load train set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.3805, 0.3484, 0.3574), (0.3031, 0.2950, 0.3007)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            if is_augmented:
                transform.append(transforms.RandomCrop(32, padding=4))
                transform.append(transforms.RandomHorizontalFlip())
            transform = transforms.Compose(transform)
            self.train_ds = datasets.GTSRB(root, 'train', transform, download=download)
            
            # Load test set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.3805, 0.3484, 0.3574), (0.3031, 0.2950, 0.3007)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.test_ds = datasets.GTSRB(root, 'test', transform, download=download)
        
        # ImageNette
        if dataset in ['imagenette', 'IMAGENETTE']:
            # Load train set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4672, 0.4581, 0.4093), (0.2580, 0.2470, 0.2671)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            if is_augmented:
                transform.append(transforms.RandomCrop(128, padding=16))
                transform.append(transforms.RandomHorizontalFlip())
            transform = transforms.Compose(transform)
            self.train_ds = datasets.ImageFolder(root+'/train', transform)
            # Load test set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4672, 0.4581, 0.4093), (0.2580, 0.2470, 0.2671)),
            ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))
            transform = transforms.Compose(transform)
            self.test_ds = datasets.ImageFolder(root+'/val', transform)
        
        # Caltech256
        if dataset == 'Caltech256':
            # Load train set
            transform = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ]
            if resize:
                transform.append(transforms.Resize((resize, resize)))  # 224
            if is_augmented:
                transform.append(transforms.RandomCrop(224, padding=16))
                transform.append(transforms.RandomHorizontalFlip())
            transform = transforms.Compose(transform),
            self.train_ds = DS.Caltech256(root, True, transform, target_transform=None, download=True, random_seed=27)
            # Load test set
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])
            D = DS.Caltech256(root, False, transform, target_transform=None, download=True, random_seed=27)

        # Build loader
        if self.train_ds is not None:
            self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=shuffle_train, pin_memory=True, drop_last=drop_last_train)
        if self.test_ds is not None:
            self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True, drop_last=drop_last_test)

        return
    

    @staticmethod
    def plot_samples(ds, n_samples:int=10) -> None:
        '''Plot a fraction of samples in the dataset'''
        idx = np.random.randint(len(ds), size=(n_samples))
        img = None
        for i in idx:
            x = ds[i][0].unsqueeze(0)
            img = opt.tensor.cat_tensors(img, x)
        opt.tensor.plot_image_grid(img, 10)
        return
    

    @staticmethod
    def load_dataset(dataset:str, root:str, train:bool, download:bool, resize:int, return_loader:bool=False, transform=None, **kwargs):
        return 

    @staticmethod
    def get_subsets(D:Dataset, n_subsets:int, n_samples_per_subset:int):
        n_samples_total = n_subsets * n_samples_per_subset
        n_samples = len(D)
        assert n_samples_total < n_samples, 'Subset size must be smaller than the dataset size'
        idx = random.sample(list(range(n_samples)), n_samples_total)
        subsets = ()
        for i in range(n_subsets):
            sp = i*n_samples_per_subset
            ep = (i+1)*n_samples_per_subset
            cur_idx = idx[sp:ep]
            subsets = subsets + (Subset(D, cur_idx), )
        return subsets

    @staticmethod
    def load_tensor_dataset(pth_tensor_ds:str) -> Dataset:
        D = torch.load(pth_tensor_ds)
        D = TensorDataset(D)
        return D
    
    @staticmethod
    def tensors_to_dataset(*tensors:torch.Tensor) -> Dataset:
        return TensorDataset(*tensors)
    
    @staticmethod
    def data_loader(dataset:Dataset, bs:int, shuffle:bool, num_worker:int=2, pin_memory:bool=True) -> DataLoader:
        return DataLoader(dataset, bs, shuffle, num_workers=num_worker, pin_memory=pin_memory)


def debug() -> None:
    return


if __name__ == '__main__':
    debug()