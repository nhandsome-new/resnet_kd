import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models import ResNetModel
from src.cifar10_models import resnet18

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
        
        self.criterion = self.get_criterion()
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        optimizer = self.get_optimizer()
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
    
    def get_optimizer(self):
        optimizer_name = self.optimizer_conf.name
        if hasattr(torch.optim, optimizer_name):
            optimizer_class = getattr(torch.optim, optimizer_name)
        else:
            raise ValueError(f'Optimizer torch.optim.{optimizer_name} does not exist. Change the configuration optimization.optimizer_name.')
        
        if optimizer_name == 'SGD':
            optimizer = optimizer_class(
                self.parameters(), 
                lr = self.optimizer_conf.lr, 
                momentum = self.optimizer_conf.momentum,
                weight_decay = self.optimizer_conf.weight_decay
            )
        else:
            optimizer = optimizer_class(
                self.parameters(), 
                lr = self.optimizer_conf.lr, 
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
        
        if 'teacher_model_name' in model_conf:
            self.T = model_conf.temperature
            self.teacher_lambda = model_conf.teacher_lambda
            self.teacher_model = self.load_model()
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.teacher_model = None    
        
        self.criterion = self.get_criterion()
        self.save_hyperparameters()
    
    def load_model(self):
        model = resnet18()
        model = resnet18(pretrained=True)
        # model = torch.load(self.model_conf.teacher_model_path)
        return model
        
    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        return [optimizer]
    
    def forward(self, x):
        output = self.model(x)
        return output
    
    def compute_loss_and_acc(self, batch, stage):
        inputs, labels = batch
        preds = self.forward(inputs)
        loss_SL = self.criterion(preds, labels)
        
        if stage in ['train', 'val']:
            teacher_outputs = self.teacher_model(inputs)
            loss_KD = nn.KLDivLoss()(F.log_softmax(preds / self.T, dim=1),
                                                F.softmax(teacher_outputs / self.T, dim=1))
            loss = (1 - self.teacher_lambda) * loss_SL + self.teacher_lambda * self.T * self.T * loss_KD
            
            self.log(f'{stage}_loss_SL', loss_SL)
            self.log(f'{stage}_loss_KD', loss_KD)
            acc_teacher =  (labels == torch.argmax(teacher_outputs, 1)).type(torch.FloatTensor).mean()
            self.log(f'{stage}_acc_teacher', acc_teacher)
        
        acc =  (labels == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        
        self.log(f'{stage}_loss', loss)
        self.log(f'{stage}_acc', acc)
        
        return loss
    
    def training_step(self, batch, batch_idx) :
        loss = self.compute_loss_and_acc(batch, 'train')
        
        return loss

    def validation_step(self, batch, batch_idx) :
        _ = self.compute_loss_and_acc(batch, 'val')
        
    def test_step(self, batch, batch_idx) :
        _ = self.compute_loss_and_acc(batch, 'test')
    
    def get_optimizer(self):
        optimizer_name = self.optimizer_conf.name
        if hasattr(torch.optim, optimizer_name):
            optimizer_class = getattr(torch.optim, optimizer_name)
        else:
            raise ValueError(f'Optimizer torch.optim.{optimizer_name} does not exist. Change the configuration optimization.optimizer_name.')
        
        if optimizer_name == 'SGD':
            optimizer = optimizer_class(
                self.parameters(), 
                lr = self.optimizer_conf.lr, 
                momentum = self.optimizer_conf.momentum,
                weight_decay = self.optimizer_conf.weight_decay
            )
        else:
            optimizer = optimizer_class(
                self.parameters(), 
                lr = self.optimizer_conf.lr, 
            )
        return optimizer
    
    def get_criterion(self):
        criterion_name = self.criterion_conf.name
        if criterion_name == 'CE':
            return nn.CrossEntropyLoss()