import os
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
    
    # def load_saved_model(self, model_path):
    #     if not os.path.exists(model_path):
    #         raise("File doesn't exist {}".format(model_path))
    #     if torch.cuda.is_available():
    #         checkpoint = torch.load(model_path)
    #     else:
    #         # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
    #         checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    #     get_empty_model()
        
    #     model.load_state_dict(checkpoint['state_dict'])

    #     if optimizer:
    #         optimizer.load_state_dict(checkpoint['optim_dict'])

    #     return checkpoint
                    

if __name__ == '__main__':
    resnet = ResNetModel(18, 10)
    print(resnet)