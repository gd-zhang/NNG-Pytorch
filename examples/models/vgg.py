import torch
import torch.nn as nn


class VGG_Cifar(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG_Cifar, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_cifar(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_Cifar(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn_cifar(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG_Cifar(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13_cifar(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_Cifar(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn_cifar(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_Cifar(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16_cifar(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_Cifar(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn_cifar(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG_Cifar(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19_cifar(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_Cifar(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn_cifar(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG_Cifar(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
