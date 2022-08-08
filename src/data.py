import os
import torch

from torchvision.datasets import CIFAR10

import torchvision.transforms as T

import pytorch_lightning as pl


DATA_DIR = 'data/cifar10'
CROP_SIZE = 32
PADDING = 4

BATCH_SIZE = 128
NUM_WORKERS = 4

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, dataset_conf, augmentation_conf):
        super().__init__()
        self.dataset_conf = dataset_conf
        self.train_transform = T.Compose([
            T.RandomCrop(augmentation_conf.crop_size, augmentation_conf.padding),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
        ])
        
        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
        ])
    
    def prepare_data(self):
        if not os.path.exists(os.path.join(self.dataset_conf.path, 'cifar-10-batches-py')):
            CIFAR10(root=self.dataset_conf.path, train=True, download=True)
            CIFAR10(root=self.dataset_conf.path, train=False, download=True)
    
    def setup(self, stage=None):
        if stage in ['fit', None]:
            _dataset = CIFAR10(root=self.dataset_conf.path, train=True, transform=self.train_transform)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                _dataset, [45000, 5000], 
                generator=torch.Generator().manual_seed(self.dataset_conf.seed)
            )
            
        if stage in ['test', None]:
            self.test_dataset = CIFAR10(root=self.dataset_conf.path, train=False, transform=self.val_transform)
    
    def get_dataloader(self, dataset, shuffle):
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            shuffle=shuffle, 
            batch_size=self.dataset_conf.batch_size, 
            num_workers=self.dataset_conf.num_workers, 
            pin_memory=True
        )
        
        return dataloader
    
    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, shuffle=True)
    
    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, shuffle=False)
    
    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, shuffle=False)
        
if __name__ == '__main__':
    cifar10_dm = CIFAR10DataModule()
    cifar10_dm.setup()
    print(cifar10_dm.train_dataset[0][1])
    print(cifar10_dm.train_dataloader())
    print(cifar10_dm.val_dataloader())
    
    for inputs, labels in cifar10_dm.train_dataloader():
        print(inputs)
        break