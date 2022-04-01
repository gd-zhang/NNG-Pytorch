import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'tat_resnet50', 'resnet101']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1,
            dilation: int = 1, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def delta_init(tensor):
    if tensor.ndimension() != 2 and tensor.ndimension() != 4:
        raise ValueError("The tensor to initialize must be "
                         "two-dimensional and at most four-dimensional")

    q = tensor.new(tensor.size(0), tensor.size(1)).normal_(0, 1)
    gain = 1.0 / math.sqrt(tensor.size(1))
    q.mul_(gain)
    with torch.no_grad():
        tensor.zero_()
        if tensor.ndimension() == 2:
            tensor[:, :] = q
        elif tensor.ndimension() == 4:
            tensor[:, :, (tensor.size(2) - 1) // 2, (tensor.size(3) - 1) // 2] = q
    return tensor


class LeakyReLU(nn.Module):
    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 0.0, inplace: bool = False) -> None:
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope, self.inplace) * math.sqrt(2 / (1 + self.negative_slope ** 2))

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        bias: bool,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        shortcuts: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = LeakyReLU(inplace=True)
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.shortcuts = shortcuts
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.activation(out)
        if self.downsample is not None and self.shortcuts:
            identity = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        if self.shortcuts:
            return out + identity
        else:
            return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        bias: bool,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        shortcuts: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = LeakyReLU(inplace=True)
        self.shortcuts = shortcuts
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, bias=bias)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, bias=bias)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=bias)
        self.bn3 = norm_layer(width)
        self.downsample = downsample
        self.stride = stride
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.activation(out)
        if self.downsample is not None and self.shortcuts:
            identity = self.downsample(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)

        if self.shortcuts:
            return out + identity
        else:
            return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        shortcuts: bool = True,
        include_bias: bool = False,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        if activation is None:
            activation = LeakyReLU(inplace=True)
        self._activaton = activation

        self.inplanes = 64
        self.dilation = 1
        self.shortcuts = shortcuts
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=include_bias)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bias=include_bias)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], bias=include_bias)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], bias=include_bias)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], bias=include_bias)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_bn = norm_layer(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if shortcuts:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                else:
                    delta_init(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, bias: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        activation = self._activaton
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.shortcuts:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride, bias=bias),
                )

        layers = []
        layers.append(block(self.inplanes, planes, bias, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.shortcuts, norm_layer, activation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bias, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, shortcuts=self.shortcuts,
                                norm_layer=norm_layer, activation=activation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_bn(x)
        x = self._activaton(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(use_bn=True, neg_slope=0.0, **kwargs: Any) -> ResNet:
    norm_layer = None
    if not use_bn:
        norm_layer = nn.Identity
    activation = LeakyReLU(neg_slope, inplace=True)
    return _resnet(BasicBlock, [2, 2, 2, 2], include_bias=(not use_bn),
                   norm_layer=norm_layer, activation=activation, **kwargs)


def resnet34(use_bn=True, neg_slope=0.0, **kwargs: Any) -> ResNet:
    norm_layer = None
    if not use_bn:
        norm_layer = nn.Identity
    activation = LeakyReLU(neg_slope, inplace=True)
    return _resnet(BasicBlock, [3, 4, 6, 3], include_bias=(not use_bn),
                   norm_layer=norm_layer, activation=activation, **kwargs)


def resnet50(use_bn=True, neg_slope=0.0, **kwargs: Any) -> ResNet:
    norm_layer = None
    if not use_bn:
        norm_layer = nn.Identity
    activation = LeakyReLU(neg_slope, inplace=True)
    return _resnet(Bottleneck, [3, 4, 6, 3], include_bias=(not use_bn),
                   norm_layer=norm_layer, activation=activation, **kwargs)

def tat_resnet50(**kwargs: Any) -> ResNet:
    norm_layer = nn.Identity
    activation = LeakyReLU(0.43, inplace=True)
    return _resnet(Bottleneck, [3, 4, 6, 3], include_bias=True,
                   norm_layer=norm_layer, activation=activation, shortcuts=False, **kwargs)


def resnet101(use_bn=True, neg_slope=0.0, **kwargs: Any) -> ResNet:
    norm_layer = None
    if not use_bn:
        norm_layer = nn.Identity
    activation = LeakyReLU(neg_slope, inplace=True)
    return _resnet(Bottleneck, [3, 4, 23, 3], include_bias=(not use_bn),
                   norm_layer=norm_layer, activation=activation, **kwargs)
