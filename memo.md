# Knowledge Distillation

## Concept
- Base Line : ResNet18 
- Offline KD : Teacher(ResNet50)
    - [teacher model](https://github.com/huyvnphan/PyTorch_CIFAR10)
- Online KD : Teatcher(ResNet50)
    - 
- Self KD : ResBet18

- Dataset : CIFAR10

- With subset augmentation
- W/O subset augmentation


### 'unexpected key "module.xxx.weight" in state_dict'
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])

### middle_feature = middle_output


