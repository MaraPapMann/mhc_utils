# DataFetcher

DataFetcher is a universal dataset object that preprocesses and loads the dataset.

Currently, the following datasets are contained within the DataFetcher:

- MNIST (Digits in gray scale)
- EMNIST (Extended version of MNIST)
- Fashion-MNIST (Clothes in gray scale)
- CIFAR-10 (10-class objects in RGB)
- CINIC-10 (Extended version of CIFAR-10)
- CIFAR-100 (100-class objects in RGB)
- CIFAR-100-EXT (Extended version of CIFAR-100)
- Caltech256 (256-class objects of high resolution in RGB)
- Cubs200 (200-class objects of high resolution in RGB)
- GTSRB (German traffic signs)
- DTD
- ImageNet1k (1000-class version of ImageNet)
- ImageNette
- TinyImagenet200 (200-class version of ImageNet)
- Inaturalist
- Indoor67
- Lisa
- OxfordPet
- STL10

## How to use
An example of loading the CIFAR-10 dataset:
```py
from utils import DataFetcher

Cifar10 = DataFetcher(Params)

print(Cifar10.train_ds)  # The training dataset
print(Cifar10.test_ds)  # The test dataset
print(Cifar10.train_loader)  # The training DataLoader
print(Cifar10.test_loader)  # The test DataLoader
```