import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models import ResNetModel
from src.cifar_resnet import multi_resnet18
from src.teacher_models.densenet import densenet
from src.utils.cosine_annealing_warmup import CosineAnnealingWarmUpRestarts
from src.utils.support_utils import mixup_data, mixup_criterion, kd_loss_function, feature_loss_function
import src.teacher_models.repvgg as repvgg

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
        if self.optimizer_conf.use_cosine_anneal :
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=self.optimizer_conf.anneal_T0, T_mult=1, eta_max=self.optimizer_conf.anneal_max, T_up=1, gamma=self.optimizer_conf.anneal_gamma)
            return [optimizer], [scheduler]
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
            print(self.model_conf.teacher_model_name)
            self.teacher_model = self.load_model()
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.teacher_model = None    
        self.criterion = self.get_criterion()
        self.save_hyperparameters()
    
    def load_model(self):
        if self.model_conf.teacher_model_name == 'densenet':
            checkpoint = torch.load(self.model_conf.teacher_model_path)
            model = densenet(depth=190, growthRate=40, num_classes=self.model_conf.num_classes)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint['state_dict'])
        
        elif self.model_conf.teacher_model_name == 'repvgg':
            
            
            checkpoint = torch.load(self.model_conf.teacher_model_path)
            num_classes = self.model_conf.num_classes
            model = getattr(repvgg, f'cifar{num_classes}_repvgg_a0')()
            model.load_state_dict(checkpoint)

        assert model, 'ERROR : model conf teacher model name'
        return model
        
    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        if self.optimizer_conf.use_cosine_anneal :
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=self.optimizer_conf.anneal_T0, T_mult=1, eta_max=self.optimizer_conf.anneal_max, T_up=1, gamma=self.optimizer_conf.anneal_gamma)
            return [optimizer], [scheduler]
        return [optimizer]
    
    def forward(self, x):
        output = self.model(x)
        return output
    
    def compute_loss_and_acc(self, batch, stage):
        inputs, labels = batch
        
        if (self.criterion_conf.use_mixup) and (stage == 'train'):
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, self.criterion_conf.mixup_alpha)
            preds = self.forward(inputs)
            loss_SL = mixup_criterion(self.criterion, preds, targets_a, targets_b, lam)
        else:
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
        
class SelfKDResNetModule(pl.LightningModule):
    def __init__(self, model_conf, optimizer_conf, criterion_conf):
        super().__init__()
        self.model_conf = model_conf
        self.optimizer_conf = optimizer_conf
        self.criterion_conf = criterion_conf
        
        self.model = multi_resnet18()

        self.criterion = self.get_criterion()
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        if self.optimizer_conf.use_cosine_anneal :
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=self.optimizer_conf.anneal_T0, T_mult=1, eta_max=self.optimizer_conf.anneal_max, T_up=1, gamma=self.optimizer_conf.anneal_gamma)
            return [optimizer], [scheduler]
        return [optimizer]
    
    def forward(self, x):
        output = self.model(x)
        return output
    
    def compute_loss_and_acc(self, batch, stage):
        inputs, labels = batch
        
        if (self.criterion_conf.use_mixup) and (stage == 'train'):
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, self.criterion_conf.mixup_alpha)
            preds, middle_output_1, middle_output_2, middle_output_3, \
            final_feature, middle_feature_1, middle_feature_2, middle_feature_3 = self.forward(inputs)
            loss = mixup_criterion(self.criterion, preds, targets_a, targets_b, lam)
        else:
            preds, middle_output_1, middle_output_2, middle_output_3, \
            final_feature, middle_feature_1, middle_feature_2, middle_feature_3 = self.forward(inputs)
            loss = self.criterion(preds, labels)
        
        if stage in ['train', 'val']:
            middle1_loss = self.criterion(middle_output_1, labels)
            middle2_loss = self.criterion(middle_output_2, labels)
            middle3_loss = self.criterion(middle_output_3, labels)
            
            temp4 = preds / self.model_conf.temperature
            temp4 = torch.softmax(temp4, dim=1)
            
            loss1by4 = kd_loss_function(middle_output_1, temp4.detach(), self.model_conf.temperature) * (self.model_conf.temperature**2)
            loss2by4 = kd_loss_function(middle_output_2, temp4.detach(), self.model_conf.temperature) * (self.model_conf.temperature**2)
            loss3by4 = kd_loss_function(middle_output_3, temp4.detach(), self.model_conf.temperature) * (self.model_conf.temperature**2)
            
            feature_loss_1 = feature_loss_function(middle_feature_1, final_feature.detach()) 
            feature_loss_2 = feature_loss_function(middle_feature_2, final_feature.detach()) 
            feature_loss_3 = feature_loss_function(middle_feature_3, final_feature.detach()) 

            total_loss = (1 - self.model_conf.self_alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                        self.model_conf.self_alpha * (loss1by4 + loss2by4 + loss3by4) + \
                        self.model_conf.self_beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)
                        
            self.log(f'{stage}_total_loss', total_loss)
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_middle1_loss', middle1_loss)
            self.log(f'{stage}_middle2_loss', middle2_loss)
            self.log(f'{stage}_middle3_loss', middle3_loss)
            self.log(f'{stage}_feature_loss_1', feature_loss_1)
            self.log(f'{stage}_feature_loss_2', feature_loss_2)
            self.log(f'{stage}_feature_loss_3', feature_loss_3)
        
        acc =  (labels == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        self.log(f'{stage}_acc', acc)
    
        return total_loss
    
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
        
        
class TeacherKDResNetModule(pl.LightningModule):
    def __init__(self, model_conf, optimizer_conf, criterion_conf):
        super().__init__()
        self.model_conf = model_conf
        self.optimizer_conf = optimizer_conf
        self.criterion_conf = criterion_conf
        
        self.model = multi_resnet18()

        if 'teacher_model_name' in model_conf:
            self.T = model_conf.temperature
            print(self.model_conf.teacher_model_name)
            self.teacher_model = self.load_model()
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.teacher_model = None    
        self.criterion = self.get_criterion()
        self.save_hyperparameters()
    
    def load_model(self):
        if self.model_conf.teacher_model_name == 'densenet':
            checkpoint = torch.load(self.model_conf.teacher_model_path)
            model = densenet(depth=190, growthRate=40, num_classes=self.model_conf.num_classes)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint['state_dict'])
        
        elif self.model_conf.teacher_model_name == 'repvgg':
            checkpoint = torch.load(self.model_conf.teacher_model_path)
            num_classes = self.model_conf.num_classes
            model = getattr(repvgg, f'cifar{num_classes}_repvgg_a0')()
            model.load_state_dict(checkpoint)

        assert model, 'ERROR : model conf teacher model name'
        return model
        
    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        if self.optimizer_conf.use_cosine_anneal :
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=self.optimizer_conf.anneal_T0, T_mult=1, eta_max=self.optimizer_conf.anneal_max, T_up=1, gamma=self.optimizer_conf.anneal_gamma)
            return [optimizer], [scheduler]
        return [optimizer]
    
    def forward(self, x):
        output = self.model(x)
        return output
    
    def compute_loss_and_acc(self, batch, stage):
        inputs, labels = batch
        
        if (self.criterion_conf.use_mixup) and (stage == 'train'):
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, self.criterion_conf.mixup_alpha)
            preds, middle_output_1, middle_output_2, middle_output_3, \
            final_feature, middle_feature_1, middle_feature_2, middle_feature_3 = self.forward(inputs)
            loss = mixup_criterion(self.criterion, preds, targets_a, targets_b, lam)
        else:
            preds, middle_output_1, middle_output_2, middle_output_3, \
            final_feature, middle_feature_1, middle_feature_2, middle_feature_3 = self.forward(inputs)
            loss = self.criterion(preds, labels)
        
        if stage in ['train', 'val']:
            teacher_outputs = self.teacher_model(inputs)
            middle1_loss = self.criterion(middle_output_1, labels)
            middle2_loss = self.criterion(middle_output_2, labels)
            middle3_loss = self.criterion(middle_output_3, labels)
            
            temp = teacher_outputs / self.model_conf.temperature
            temp = torch.softmax(temp, dim=1)
            
            temp4 = preds / self.model_conf.temperature
            temp4 = torch.softmax(temp4, dim=1)
            
            loss4byteacher = kd_loss_function(preds, temp.detach(), self.model_conf.temperature) * (self.model_conf.temperature**2)
            loss1by4 = kd_loss_function(middle_output_1, temp4.detach(), self.model_conf.temperature) * (self.model_conf.temperature**2)
            loss2by4 = kd_loss_function(middle_output_2, temp4.detach(), self.model_conf.temperature) * (self.model_conf.temperature**2)
            loss3by4 = kd_loss_function(middle_output_3, temp4.detach(), self.model_conf.temperature) * (self.model_conf.temperature**2)
            
            feature_loss_1 = feature_loss_function(middle_feature_1, final_feature.detach()) 
            feature_loss_2 = feature_loss_function(middle_feature_2, final_feature.detach()) 
            feature_loss_3 = feature_loss_function(middle_feature_3, final_feature.detach()) 

            total_loss = (1 - self.model_conf.self_alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                        self.model_conf.self_alpha * (loss4byteacher + loss1by4 + loss2by4 + loss3by4) + \
                        self.model_conf.self_beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)
                        
            self.log(f'{stage}_total_loss', total_loss)
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_loss4byteacher', loss4byteacher)
            self.log(f'{stage}_middle1_loss', middle1_loss)
            self.log(f'{stage}_middle2_loss', middle2_loss)
            self.log(f'{stage}_middle3_loss', middle3_loss)
            self.log(f'{stage}_feature_loss_1', feature_loss_1)
            self.log(f'{stage}_feature_loss_2', feature_loss_2)
            self.log(f'{stage}_feature_loss_3', feature_loss_3)
        
        acc =  (labels == torch.argmax(preds, 1)).type(torch.FloatTensor).mean()
        self.log(f'{stage}_acc', acc)
    
        return total_loss
    
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