import hydra
from omegaconf import OmegaConf
from src.module import ResNetModule

@hydra.main(config_path='configs/', config_name='train.yaml')
def main(config):
    print(OmegaConf.to_yaml(cfg=config))
    pl_module = hydra.utils.instantiate(config.modules)
    data_module = hydra.utils.instantiate(config.datamodules)
    trainer = hydra.utils.instantiate(config.trainer)

    trainer.fit(model=pl_module, datamodule=data_module)

@hydra.main(config_path='configs/', config_name='train.yaml')
def test(config):
    print(OmegaConf.to_yaml(cfg=config))
    pl_module = hydra.utils.instantiate(config.modules)
    # data_module = hydra.utils.instantiate(config.datamodules)
    # trainer = hydra.utils.instantiate(config.trainer)

    # trainer.fit(model=pl_module, datamodule=data_module)
    
    print(pl_module.teacher_model)

if __name__ == '__main__':
    main()