import hydra
import numpy as np
import torch

def extract_callbacks(callbacks_dict):
    '''
    Extract Pytorch lightning callbacks using callback yaml files
    Return 
        callback list
    '''
    callbacks = []
    for _, cb_conf in callbacks_dict.items():
        if "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

def mixup_data(x, y, alpha, device='cuda'):
    '''
    Create Mixup image using one in the same batch
    Params
        x: images in the batch
        y: labels of images
        alpha
    Return
        mixed_x: Mixed images
        y_a, y_b: targets of the mixed two images
        lambda: ratio
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def kd_loss_function(output, target_output, temperature):
    '''
    Compute kd loss
    Params
        output: 
        target_output: 
        temperature:
    Return
        loss_kd
    '''
    output = output / temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

def feature_loss_function(feature, target_feature):
    loss = (feature - target_feature)**2 * ((feature > 0) | (target_feature > 0)).float()
    return torch.abs(loss).sum()