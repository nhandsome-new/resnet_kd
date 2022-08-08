import os
import torch
from torch import nn
import pytorch_lightning as pl

from src.models import ResNetModel
from src.data import CIFAR10DataModule

class ResNetModule(pl.LightningModule):
    def __init__(self, model_conf, optimizer_conf, criterion_conf):
        super().__init__()
        self.model_conf = model_conf
        self.optimizer_conf = optimizer_conf
        self.criterion_conf = criterion_conf
        
        self.model = ResNetModel(
            model_conf.resnet_version, 
            model_conf.num_classes, 
            model_conf.pre_train, 
            model_conf.tune_only_fc
        )
        
        self.criterion = self.get_criterion(criterion_conf)
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = self.get_optimizer(self.optimizer_conf, self.parameters())
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
    
    def get_optimizer(self, optimizer_conf, parameters):
        optimizer_name = optimizer_conf.name
        if hasattr(torch.optim, optimizer_name):
            optimizer_class = getattr(torch.optim, optimizer_name)
        else:
            raise ValueError(f'Optimizer torch.optim.{optimizer_name} does not exist. Change the configuration optimization.optimizer_name.')
        
        optimizer = optimizer_class(
            parameters, 
            lr=optimizer_conf.lr, 
            # momentum=optimizer_conf.momentum,
            # weight_decay=optimizer_conf.weight_decay
        )
        
        return optimizer
    
    def get_criterion(self):
        criterion_name = self.criterion_conf.name
        if criterion_name == 'CE':
            return nn.CrossEntropyLoss()
    
class KDResNetModule(pl.LightningModule):
    def __init__(self, model_conf, optimizer_conf, criterion_conf):
        super().__init__()
        self.model_conf = model_conf
        self.optimizer_conf = optimizer_conf
        self.criterion_conf = criterion_conf
        
        self.model = ResNetModel(
            model_conf.resnet_version, 
            model_conf.num_classes, 
            model_conf.pre_train, 
            model_conf.tune_only_fc
        )
        self.teacher_model = ResNetModel(model_conf.teacher_resnet_version, model_conf.num_classes)
        # self.teacher_model.load_state_dict()
        self.criterion = self.get_criterion(criterion_conf)
        self.save_hyperparameters()
    
    def configure_optimizers(self):
        optimizer = self.get_optimizer(self.optimizer_conf, self.parameters())
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
    
    def get_optimizer(self, optimizer_conf, parameters):
        optimizer_name = optimizer_conf.name
        if hasattr(torch.optim, optimizer_name):
            optimizer_class = getattr(torch.optim, optimizer_name)
        else:
            raise ValueError(f'Optimizer torch.optim.{optimizer_name} does not exist. Change the configuration optimization.optimizer_name.')
        
        optimizer = optimizer_class(
            parameters, 
            lr=optimizer_conf.lr, 
            # momentum=optimizer_conf.momentum,
            # weight_decay=optimizer_conf.weight_decay
        )
        
        return optimizer
    
    def get_criterion(self, criterion_conf):
        criterion_name = self.criterion_conf.name
        if criterion_name == 'CE':
            return nn.CrossEntropyLoss()