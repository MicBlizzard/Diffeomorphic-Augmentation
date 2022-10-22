"""VGG11/13/16/19 in Pytorch."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

warnings.filterwarnings(
    "ignore", message="Setting attributes on ParameterList is not supported."
)

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

class VGG(nn.Module):
    def __init__(
        self,
        vgg_name,
        num_ch=3,
        num_classes=10,
        bias=True,
        batch_norm=True,
        pooling="max",
        pooling_size=2,
        param_list=False,
        width_factor=1,
        stride=2,
    ):
        super(VGG, self).__init__()
        if pooling == True:
            pooling = "max"
        self.features = self._make_layers(
            cfg[vgg_name],
            ch=num_ch,
            bias=bias,
            bn=batch_norm,
            pooling=pooling,
            ps=pooling_size,
            param_list=param_list,
            width_factor=width_factor,
            stride=stride
        )
        stride_factor = 729 if stride == 1 else 1
        self.classifier = nn.Linear(int(512 * width_factor) * stride_factor, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, ch, bias, bn, pooling, ps, param_list, width_factor, stride):
        layers = []
        in_channels = ch
        if ch == 1:
            layers.append(nn.ZeroPad2d(2))
        if param_list:
            convLayer = Conv2dList
        else:
            convLayer = nn.Conv2d
        for x in cfg:
            if x == "M":
                if pooling == "max":
                    layers += [
                        nn.MaxPool2d(
                            kernel_size=ps, stride=stride, padding=ps // 2 + ps % 2 - 1
                        )
                    ]
                elif pooling == "avg":
                    layers += [
                        nn.AvgPool2d(
                            kernel_size=ps, stride=stride, padding=ps // 2 + ps % 2 - 1
                        )
                    ]
                else:
                    layers += [SubSampling(kernel_size=ps, stride=stride)]
            else:
                x = int(x * width_factor)
                if bn:
                    layers += [
                        convLayer(in_channels, x, kernel_size=3, padding=1, bias=bias),
                        nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        convLayer(in_channels, x, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                    ]
                in_channels = x
        return nn.Sequential(*layers)


class Conv2dList(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        padding_mode="constant",
    ):
        super().__init__()

        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)

        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if bias is not None:
            bias = nn.Parameter(
                torch.empty(
                    out_channels,
                )
            )
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        n = max(1, 128 * 256 // (in_channels * kernel_size * kernel_size))
        weight = nn.ParameterList(
            [nn.Parameter(weight[j : j + n]) for j in range(0, len(weight), n)]
        )

        setattr(self, "weight", weight)
        setattr(self, "bias", bias)

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride

    def forward(self, x):

        weight = self.weight
        if isinstance(weight, nn.ParameterList):
            weight = torch.cat(list(self.weight))

        return F.conv2d(x, weight, self.bias, self.stride, self.padding)


class SubSampling(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(SubSampling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.unfold(2, self.kernel_size, self.stride).unfold(
            3, self.kernel_size, self.stride
        )[..., 0, 0]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_ch=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_ch, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_ch=3, num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_ch=num_ch)


def ResNet34(num_ch=3, num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, num_ch=num_ch)


def ResNet50(num_ch=3, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, num_ch=num_ch)


def ResNet101(num_ch=3, num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, num_ch=num_ch)


def ResNet152(num_ch=3, num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, num_ch=num_ch)
