import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl

class ResNetModel(nn.Module):
    def __init__(self, resnet_version, num_classes, pre_train=False, tune_only_fc=False):
        super().__init__()
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50
        }
        
        self.resnet = resnets[resnet_version](pretrained=pre_train)
        fc_input_size = list(self.resnet.children())[-1].in_features
        self.resnet.fc = nn.Linear(fc_input_size, num_classes)
        
        if tune_only_fc:
            for chlid in list(self.resnet.children())[:-1]:
                for param in chlid.parameters():
                    param.require_grad = False
    def forward(self, inputs):
        outputs = self.resnet(inputs)
        return outputs
                    

if __name__ == '__main__':
    resnet = ResNetModel(18, 10)
    print(resnet)