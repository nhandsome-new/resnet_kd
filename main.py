import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl

from src.utils.support_utils import extract_callbacks

@hydra.main(config_path='configs/', config_name='train.yaml')
def main(config):
    print(OmegaConf.to_yaml(cfg=config))
    if "seed" in config:
        pl.seed_everything(config.seed)
    pl_module = hydra.utils.instantiate(config.modules)
    data_module = hydra.utils.instantiate(config.datamodules)
    callbacks = extract_callbacks(config.callbacks)
    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    trainer.fit(model=pl_module, datamodule=data_module)

@hydra.main(config_path='configs/', config_name='train.yaml')
def test(config):
    print(OmegaConf.to_yaml(cfg=config))
    pl_module = hydra.utils.instantiate(config.modules)
    data_module = hydra.utils.instantiate(config.datamodules)
    # trainer = hydra.utils.instantiate(config.trainer)

    # trainer.fit(model=pl_module, datamodule=data_module)
    data_module.setup()
    print(data_module.train_dataloader().dataset[0][0].shape)

if __name__ == '__main__':
    main()
    # test()
    # import torch
    # import src.teacher_models.repvgg as repvgg_feature
    # checkpoint = torch.load('/content/drive/MyDrive/fusic/resnet_kd/data/cifar100/cifar100_repvgg_a0-2df1edd0.pt')
    # num_classes = 100
    # model = getattr(repvgg_feature, f'cifar{num_classes}_repvgg_a0')()
    # model.load_state_dict(checkpoint)
    
    # temp_input = torch.rand(1,3,32,32)
    # print(model)