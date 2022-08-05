import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import SGD, Adam

from models import ResNetModel
from data import CIFAR10DataModule

RESNET_VERSION = 18
NUM_CLASSES = 10
PRETRAIN=True
TUNE_ONLY_FC=False

OPTIMIZER_NAME = 'Adam'
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

class ResNetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNetModel(RESNET_VERSION, NUM_CLASSES, pre_train=PRETRAIN, tune_only_fc=TUNE_ONLY_FC)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = {'adam': Adam, 'sgd': SGD}
        self.prepare_data_per_node = True
    
    def configure_optimizers(self, optimizer_name=OPTIMIZER_NAME, lr=LR):
        optimizer = self.get_optimizer(optimizer_name, self.parameters(), lr)
        return [optimizer]
    
    def forward(self, x):
        output = self.model(x)
        return output
    
    def compute_loss_and_acc(self, batch):
        inputs, labels = batch
        preds = self.forward(inputs)
        loss = self.criterion(preds, labels)
        acc =  (labels == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        
        return loss, acc
    
    def training_step(self, batch, batch_idx) :
        loss, acc = self.compute_loss_and_acc(batch)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss

    def validation_step(self, batch, batch_idx) :
        loss, acc = self.compute_loss_and_acc(batch)
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
    def test_step(self, batch, batch_idx) :
        loss, acc = self.compute_loss_and_acc(batch)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
    
    def get_optimizer(self, optimizer_name, parameters, lr):
        if hasattr(torch.optim, optimizer_name):
            optimizer_class = getattr(torch.optim, optimizer_name)
        else:
            raise ValueError(f'Optimizer torch.optim.{optimizer_name} does not exist. Change the configuration optimization.optimizer_name.')
        
        optimizer = optimizer_class(parameters, lr=lr)
        
        return optimizer
    
if __name__ == '__main__':
    
    module = ResNetModule()
    cifar10_dm = CIFAR10DataModule()
    cifar10_dm.prepare_data()
    cifar10_dm.setup()
    
    trainer = pl.Trainer(
        devices=None,
        max_epochs=1,
        default_root_dir=f"lightning_logs/'test'",
        fast_dev_run = 1
    )
    
    trainer.fit(module, cifar10_dm.train_dataloader(), cifar10_dm.val_dataloader())
    trainer.test(module, cifar10_dm.test_dataloader())