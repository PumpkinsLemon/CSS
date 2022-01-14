## Channel Self-Supervision for Online Knowledge Distillation



This is the Pytorch implementation of CSS in the paper Channel Self-Supervision for Online Knowledge Distillation.

This implementation is based on these repositories:

- [Pytorch-classification](https://github.com/bearpaw/pytorch-classification/)
- [OKDDip-AAAI2020](https://github.com/DefangChen/OKDDip-AAAI2020)
- [ONE_NeurIPS2018](https://github.com/Lan1991Xu/ONE_NeurIPS2018)

### Main Requirements

- torch == 1.0.1
- torchvision == 0.2.1
- Python 3.5

### Training Examples

- Training between ResNet-110 and ResNet-56 on CIFAR-100

  `python train.py -d cifar100 --model_t resnet110 --model_s resnet56`

- Training between ResNet-110 and ResNet-110 on CIFAR-100 with diversity

  `python train.py -d cifar100 --model_t resnet110 --model_s resnet110 --get_dis`
