## Introduction
This repository contains the pytorch implementation (with multi-gpu support) of Noisy Natural Gradient as Variational Inference [Paper](https://arxiv.org/abs/1712.02390), [Video](https://www.youtube.com/watch?v=bWItvHYqKl8).

Noisy Natural Gradient: Variational Inference can be instantiated as natural gradient with adaptive weight noise. By further approximating full Fisher with [K-FAC](https://arxiv.org/abs/1503.05671), we get noisy K-FAC, a surprisingly simple variational training algorithm for Bayesian Neural Nets. Noisy K-FAC not only improves the classification accuracy, but also gives well-calibrated prediction. 
There is a concurrent work called [VOGN](https://proceedings.mlr.press/v80/khan18a/khan18a.pdf) by Emti Khan from Riken.

*Note: this repo was orginally built on top of [Pytorch-SSO](https://github.com/cybertronai/pytorch-sso).*

## Dependencies
This project uses Python 3.6.0. Before running the code, you have to install
* [PyTorch 1.8.0](http://pytorch.org/)

## Example

VGG16 (w/o batch norm) on CIFAR10. We used `kl-lam = 0.2` because of the use of data augmentation. You should expect to get 89.5% acc and roughly same confidence.
```
python examples/cifar.py --kl-lam 0.2 --lr 0.01 --precision 0.0
```

The MNIST example is just for debugging purpose.

## Working in Progress

- ImageNet ResNet50 example
- Support of Noisy Adam for ConvNets
- Support of Batch norm (for now, one can just froze the batch norm parameters)

## Citation
To cite this work, please use
```
@inproceedings{zhang2018noisy,
  title={Noisy natural gradient as variational inference},
  author={Zhang, Guodong and Sun, Shengyang and Duvenaud, David and Grosse, Roger},
  booktitle={International Conference on Machine Learning},
  pages={5852--5861},
  year={2018},
  organization={PMLR}
}
```
