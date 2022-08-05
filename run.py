import torch
import pytorch_lightning as pl

from module import ResNetModule
from data import CIFAR10DataModule

if __name__ == '__main__':
    
    RESNET_VERSION = 18
    NUM_CLASSES = 10
    PRETRAIN=True
    TUNE_ONLY_FC=False
    
    module = ResNetModule(RESNET_VERSION, NUM_CLASSES, PRETRAIN, TUNE_ONLY_FC)
    cifar10_dm = CIFAR10DataModule()
    cifar10_dm.prepare_data()
    cifar10_dm.setup()
    
    trainer = pl.Trainer(
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=50,
        default_root_dir=f"lightning_logs/'test'",
        fast_dev_run = 1
    )
    
    trainer.fit(module, cifar10_dm.train_dataloader(), cifar10_dm.val_dataloader())
    trainer.test(module, cifar10_dm.test_dataloader())