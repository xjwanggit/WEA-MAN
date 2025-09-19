import torch
import torch.nn as nn
from collections import namedtuple
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
from collections import OrderedDict
import re



# __all__ = [
#     'VGG','VGG11','VGG13','VGG16','VGG19',
# ]


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d','DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'AlexNet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'VGG11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'VGG13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'VGG19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        for layer in self.features:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        # x = self.features(x)
        x = self.avgpool(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        x = x.reshape(x.size(0), 256 * 6 * 6)
        # layer_output.append(x)
        x = self.classifier(x)
        layer_output.append(x)
        return x, layer_output, layer_output_feature


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['AlexNet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        for layer in self.features:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        # x = self.features(x)
        x = self.avgpool(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        layer_output.append(x)
        return x, layer_output, layer_output_feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def VGG11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('VGG11', 'A', False, pretrained, progress, **kwargs)

def VGG13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('VGG13', 'B', False, pretrained, progress, **kwargs)

def VGG16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('VGG16', 'D', False, pretrained, progress, **kwargs)

def VGG19(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('VGG19', 'E', False, pretrained, progress, **kwargs)

_GoogLeNetOuputs = namedtuple('GoogLeNetOuputs', ['logits', 'aux_logits2', 'aux_logits1'])


def googlenet(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model

    return GoogLeNet(**kwargs)

class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input'] # 补充
    
    def __init__(self, num_classes=100, aux_logits=True, transform_input=False, init_weights=True):

        super(GoogLeNet, self).__init__()
       
        print("load googlenet from net_forward")
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)


        # N x 3 x 224 x 224
        # x = self.conv1(x)
#        for layer in self.conv1:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 64 x 112 x 112
#        x = self.maxpool1(x)
#        layer_output.append(x)
#        layer_output_feature.append(x)
#        # N x 64 x 56 x 56
#        # x = self.conv2(x)
#        for layer in self.conv2:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 64 x 56 x 56
#        # x = self.conv3(x)
#        for layer in self.conv3:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 192 x 56 x 56
#        x = self.maxpool2(x)
#        layer_output.append(x)
#        layer_output_feature.append(x)
#        # N x 192 x 28 x 28
#        # x = self.inception3a(x)
#        for layer in self.inception3a:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 256 x 28 x 28
#        # x = self.inception3b(x)
#        for layer in self.inception3b:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 480 x 28 x 28
#        x = self.maxpool3(x)
#        layer_output.append(x)
#        layer_output_feature.append(x)
#        # N x 480 x 14 x 14
#        # x = self.inception4a(x)
#        for layer in self.inception4a:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 512 x 14 x 14
#        if self.training and self.aux_logits:
#            aux1 = self.aux1(x)
#
#        # x = self.inception4b(x)
#        for layer in self.inception4b:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 512 x 14 x 14
#        # x = self.inception4c(x)
#        for layer in self.inception4c:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 512 x 14 x 14
#        # x = self.inception4d(x)
#        for layer in self.inception4d:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 528 x 14 x 14
#        if self.training and self.aux_logits:
#            aux2 = self.aux2(x)
#
#        x = self.inception4e(x)
#        for layer in self.inception4e:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 832 x 14 x 14
#        x = self.maxpool4(x)
#        layer_output.append(x)
#        layer_output_feature.append(x)
#        # N x 832 x 7 x 7
#        # x = self.inception5a(x)
#        for layer in self.inception5a:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 832 x 7 x 7
#        # x = self.inception5b(x)
#        for layer in self.inception5b:
#            x = layer(x)
#            layer_output.append(x)
#            layer_output_feature.append(x)
#        # N x 1024 x 7 x 7
#
#        x = self.avgpool(x)
#        layer_output.append(x)
#        layer_output_feature.append(x)
#        # N x 1024 x 1 x 1
#        x = x.reshape(x.size(0), -1)
#        # N x 1024
#        x = self.dropout(x)
#        x = self.fc(x)
#        layer_output.append(x)
#        # N x 1000 (num_classes)
#        if self.training and self.aux_logits:
#            return _GoogLeNetOuputs(x, aux2, aux1)
##            return x, layer_output, layer_output_feature
#        return x, layer_output, layer_output_feature
    # def forward(self, x):
    #     if self.transform_input:
    #         x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    #         x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    #         x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    #         x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    #
        # N x 3 x 224 x 224
        x = self.conv1(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        layer_output.append(x)
        layer_output_feature.append(x)        
        # N x 64 x 56 x 56
        x = self.conv3(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 480 x 14 x 14
        x = self.inception4a(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 512 x 14 x 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
    #
        x = self.inception4b(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 512 x 14 x 14
        x = self.inception4c(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 512 x 14 x 14
        x = self.inception4d(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
    #
        x = self.inception4e(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 832 x 14 x 14
        x = self.maxpool4(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 832 x 7 x 7
        x = self.inception5a(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 832 x 7 x 7
        x = self.inception5b(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 1024 x 7 x 7
    #
        x = self.avgpool(x)
        layer_output.append(x)
        layer_output_feature.append(x)
    #     # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
    #     # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        layer_output.append(x)
    #     # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _GoogLeNetOuputs(x, aux2, aux1)
        return x, layer_output, layer_output_feature



class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)



class InceptionAux(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionAux, self).__init__()

        self.auxiliary_avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.auxiliary_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.auxiliary_conv2 = nn.Conv2d(in_channels=128, out_channels=768, kernel_size=5,stride=1)
        self.auxiliary_dropout = nn.Dropout(p=0.7)
        self.auxiliary_linear1 = nn.Linear(in_features=768, out_features=out_channels)

    def forward(self, x):
        x = self.auxiliary_conv1(self.auxiliary_avgpool(x))
        x = self.auxiliary_conv2(x)
        x = x.view(x.size(0), -1)
        out = self.auxiliary_linear1(self.auxiliary_dropout(x))
        return out

def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

#class InceptionAux(nn.Module):
#
#    def __init__(self, in_channels, num_classes):
#        super(InceptionAux, self).__init__()
#        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
#
#        self.fc1 = nn.Linear(2048, 1024)
#        self.fc2 = nn.Linear(1024, num_classes)
#
#    def forward(self, x):
#        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
#        x = F.adaptive_avg_pool2d(x, (4, 4))
#        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
#        x = self.conv(x)
#        # N x 128 x 4 x 4
#        x = x.view(x.size(0), -1)
#        # N x 2048
#        x = F.relu(self.fc1(x), inplace=True)
#        # N x 2048
#        x = F.dropout(x, 0.7, training=self.training)
#        # N x 2048
#        x = self.fc2(x)
#        # N x 1024
#
#        return x



class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)



class ZFNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        for layer in self.features:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        # x = self.features(x)
        x = self.avgpool(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        x = x.reshape(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        for layer_fc in self.classifier:
            x = layer_fc(x)
            layer_output.append(x)
        return x, layer_output, layer_output_feature




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
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
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        x = self.conv1(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        x = self.bn1(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        for layer in self.layer1:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        for layer in self.layer2:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        for layer in self.layer3:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        for layer in self.layer4:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        layer_output.append(x)
        layer_output_feature.append(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        layer_output.append(x)
        return x, layer_output, layer_output_feature


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)





class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        for layer in self.features:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        # features = self.features(x)
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).reshape(out.size(0), -1)
        out = self.classifier(out)
        layer_output.append(out)
        return out, layer_output, layer_output_feature


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)


def densenet161(pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)


def densenet169(pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)


def densenet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)

class Mobile_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Mobile_Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                          stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out



class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(
                    Mobile_Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        x_tmp = self.conv1(x)
        layer_output.append(x_tmp)
        layer_output_feature.append(x_tmp)
        x_tmp = self.bn1(x_tmp)
        layer_output.append(x_tmp)
        layer_output_feature.append(x_tmp)
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
            layer_output.append(out)
            layer_output_feature.append(out)
        # out = self.layers(out)
        x_tmp = self.conv2(out)
        layer_output.append(x_tmp)
        layer_output_feature.append(x_tmp)
        x_tmp = self.bn2(x_tmp)
        layer_output.append(x_tmp)
        layer_output_feature.append(x_tmp)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        layer_output.append(out)
        layer_output_feature.append(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        layer_output.append(out)
        return out, layer_output, layer_output_feature

def mobilenet(pretrained=False, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['MobileNetV2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def split(x, groups):
    out = x.chunk(groups, dim=1)

    return out


def shuffle(x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    return out


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        mid_channels = out_channels // 2
        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = split(x, 2)
            out = torch.cat((self.branch1(x1), self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = shuffle(out, 2)
        return out

class BN_Conv2d(nn.Module):
    """
    BN_CONV, default activation is ReLU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False, activation=nn.ReLU(inplace=True)) -> object:
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias),
                  nn.BatchNorm2d(out_channels)]
        if activation is not None:
            layers.append(activation)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

# class ShuffleNetV2(nn.Module):
#     """ShuffleNet-v2"""
# 
#     _defaults = {
#         "sets": {0.5, 1, 1.5, 2},
#         "units": [3, 7, 3],
#         "chnl_sets": {0.5: [24, 48, 96, 192, 1024],
#                       1: [24, 116, 232, 464, 1024],
#                       1.5: [24, 176, 352, 704, 1024],
#                       2: [24, 244, 488, 976, 2048]}
#     }
# 
#     def __init__(self, scale, num_cls, is_se=False, is_res=False) -> object:
#         super(ShuffleNetV2, self).__init__()
#         self.__dict__.update(self._defaults)
#         assert (scale in self.sets)
#         self.is_se = is_se
#         self.is_res = is_res
#         self.chnls = self.chnl_sets[scale]
# 
#         # make layers
#         self.conv1 = BN_Conv2d(3, self.chnls[0], 3, 2, 1)
#         self.maxpool = nn.MaxPool2d(3, 2, 1)
#         self.stage2 = self.__make_stage(self.chnls[0], self.chnls[1], self.units[0])
#         self.stage3 = self.__make_stage(self.chnls[1], self.chnls[2], self.units[1])
#         self.stage4 = self.__make_stage(self.chnls[2], self.chnls[3], self.units[2])
#         self.conv5 = BN_Conv2d(self.chnls[3], self.chnls[4], 1, 1, 0)
#         self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.body = self.__make_body()
#         self.fc = nn.Linear(self.chnls[4], num_cls)
# 
#     def make_layers(self, in_channels, out_channels, layers_num, stride):
#         layers = []
#         layers.append(ShuffleUnit(in_channels, out_channels, stride))
#         in_channels = out_channels
# 
#         for i in range(layers_num - 1):
#             ShuffleUnit(in_channels, out_channels, 1)
# 
#         return nn.Sequential(*layers)
# 
#     def forward(self, x):
#         layer_output = []
#         layer_output_feature = []
#         for layer in self.conv1:
#             x = layer(x)
#             layer_output.append(x)
#             layer_output_feature.append(x)
#         # out = self.conv1(x)
#         # layer_output.append(x)
#         # layer_output_feature.append(x)
#         x = self.maxpool(x)
#         layer_output.append(x)
#         layer_output_feature.append(x)
#         for layer in self.stage2:
#             x = layer(x)
#             layer_output.append(x)
#             layer_output_feature.append(x)
#         for layer in self.stage3:
#             x = layer(x)
#             layer_output.append(x)
#             layer_output_feature.append(x)
#         for layer in self.stage4:
#             x = layer(x)
#             layer_output.append(x)
#             layer_output_feature.append(x)
#         # out = self.stage2(out)
#         # out = self.stage3(out)
#         # out = self.stage4(out)
#         for layer in self.conv5:
#             x = layer(x)
#             layer_output.append(x)
#             layer_output_feature.append(x)
#         # x = self.conv5(x)
#         x = self.avgpool(x)
#         layer_output.append(x)
#         x = x.flatten(1)
#         # print(x.shape)
#         x = self.fc(x)
#         layer_output.append(x)
# 
#         return x, layer_output, layer_output_feature
# 
# def shufflenet(num,pretrained=False, progress=True):
#     model = ShuffleNetV2(2, num)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls['ShuffleNetV2'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x

class DSampling(nn.Module):
    """Spatial down sampling of SuffleNet-v2"""

    def __init__(self, in_chnls, groups=2):
        super(DSampling, self).__init__()
        self.groups = groups
        self.dwconv_l1 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1,  # down-sampling, depth-wise conv.
                                   groups=in_chnls, activation=None)
        self.conv_l2 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.conv_r1 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)
        self.dwconv_r2 = BN_Conv2d(in_chnls, in_chnls, 3, 2, 1, groups=in_chnls, activation=None)
        self.conv_r3 = BN_Conv2d(in_chnls, in_chnls, 1, 1, 0)

    def forward(self, x):
        # left path
        out_l = self.dwconv_l1(x)
        out_l = self.conv_l2(out_l)

        # right path
        out_r = self.conv_r1(x)
        out_r = self.dwconv_r2(out_r)
        out_r = self.conv_r3(out_r)

        # concatenate
        out = torch.cat((out_l, out_r), 1)
        return shuffle_chnls(out, self.groups)

class BasicUnit(nn.Module):
    """Basic Unit of ShuffleNet-v2"""

    def __init__(self, in_chnls, out_chnls, is_se=False, is_residual=False, c_ratio=0.5, groups=2):
        super(BasicUnit, self).__init__()
        self.is_se, self.is_res = is_se, is_residual
        self.l_chnls = int(in_chnls * c_ratio)
        self.r_chnls = in_chnls - self.l_chnls
        self.ro_chnls = out_chnls - self.l_chnls
        self.groups = groups

        # layers
        self.conv1 = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0)
        self.dwconv2 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 3, 1, 1,  # same padding, depthwise conv
                                 groups=self.ro_chnls, activation=None)
        act = None if self.is_res else nn.ReLU(inplace=True)
        self.conv3 = BN_Conv2d(self.ro_chnls, self.ro_chnls, 1, 1, 0, activation=act)
        if self.is_se:
            self.se = SE(self.ro_chnls, 16)
        if self.is_res:
            self.shortcut = nn.Sequential()
            if self.r_chnls != self.ro_chnls:
                self.shortcut = BN_Conv2d(self.r_chnls, self.ro_chnls, 1, 1, 0, activation=None)

    def forward(self, x):
        x_l = x[:, :self.l_chnls, :, :]
        x_r = x[:, self.l_chnls:, :, :]

        # right path
        out_r = self.conv1(x_r)
        out_r = self.dwconv2(out_r)
        out_r = self.conv3(out_r)
        if self.is_se:
            coefficient = self.se(out_r)
            out_r *= coefficient
        if self.is_res:
            out_r += self.shortcut(x_r)

        # concatenate
        out = torch.cat((x_l, out_r), 1)
        return shuffle_chnls(out, self.groups)




class ShuffleNet_v2(nn.Module):
    """ShuffleNet-v2"""

    _defaults = {
        "sets": {0.5, 1, 1.5, 2},
        "units": [3, 7, 3],
        "chnl_sets": {0.5: [24, 48, 96, 192, 1024],
                      1: [24, 116, 232, 464, 1024],
                      1.5: [24, 176, 352, 704, 1024],
                      2: [24, 244, 488, 976, 2048]}
    }

    def __init__(self, scale, num_cls, is_se=False, is_res=False) -> object:
        super(ShuffleNet_v2, self).__init__()
        self.__dict__.update(self._defaults)
        assert (scale in self.sets)
        self.is_se = is_se
        self.is_res = is_res
        self.chnls = self.chnl_sets[scale]

        # make layers
        self.conv1 = BN_Conv2d(3, self.chnls[0], 3, 2, 1)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.stage2 = self.__make_stage(self.chnls[0], self.chnls[1], self.units[0])
        self.stage3 = self.__make_stage(self.chnls[1], self.chnls[2], self.units[1])
        self.stage4 = self.__make_stage(self.chnls[2], self.chnls[3], self.units[2])
        self.conv5 = BN_Conv2d(self.chnls[3], self.chnls[4], 1, 1, 0)
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.body = self.__make_body()
        self.fc = nn.Linear(self.chnls[4], num_cls)

    def __make_stage(self, in_chnls, out_chnls, units):
        layers = [DSampling(in_chnls),
                  BasicUnit(2 * in_chnls, out_chnls, self.is_se, self.is_res)]
        for _ in range(units-1):
            layers.append(BasicUnit(out_chnls, out_chnls, self.is_se, self.is_res))
        return nn.Sequential(*layers)

    def __make_body(self):
        return nn.Sequential(
            self.conv1, self.maxpool, self.stage2, self.stage3,
            self.stage4, self.conv5, self.globalpool
        )

    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        for layer in self.body:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        # out = self.body(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        layer_output.append(x)
        return F.softmax(x), layer_output, layer_output_feature

def shufflenet(num,pretrained=False, progress=True):
    model = ShuffleNet_v2(2, num)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['ShuffleNet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layer_output = []
        layer_output_feature = []
        for layer in self.features:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        # x = self.features(x)
        for layer in self.classifier:
            x = layer(x)
            layer_output.append(x)
            layer_output_feature.append(x)
        # x = self.classifier(x)
        return x.reshape(x.size(0), self.num_classes), layer_output, layer_output_feature

def squeezenet(pretrained=False, progress=True, **kwargs):
    model = SqueezeNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['SqueezeNet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


_InceptionOuputs = namedtuple('InceptionOuputs', ['logits', 'aux_logits'])

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.brach3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.brach3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.brach3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.brach3x3dbl_1(x)
        branch3x3dbl = self.brach3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.brach3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):

    def __init__(self, in_channels, channel_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channel_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x7_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0,3))
        self.branch7x7x7_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3,0))
        self.branch7x7x7_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kerenl_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kerenl_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch_3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

#class InceptionAux(nn.Module):
#
#    def __init__(self, in_channels, num_classes):
#        super(InceptionAux, self).__init__()
#        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
#        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
#        self.conv1.stddev = 0.01
#        self.fc = nn.Linear(768, num_classes)
#        self.fc.stddev = 0.001
#
#    def forward(self, x):
#        # N x 768 x 17 x 17
#        x = F.avg_pool2d(x, kernel_size=5, stride=3)
#        # N x 768 x 5 x 5
#        x = self.conv0(x)
#        # N x 128 x 5 x 5
#        x = self.conv1(x)
#        # N x 768 x 1 x 1
#        x = F.adaptive_avg_pool2d(x, (1, 1))
#        # N X 768 X 1 X 1
#        X = x.view(x.size(0), -1)
#        # N x 768
#        x = self.fc(x)
#        # N x 1000
#        return x



class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channel_7x7=128)
        self.Mixed_6c = InceptionC(768, channel_7x7=160)
        self.Mixed_6d = InceptionC(768, channel_7x7=160)
        self.Mixed_6e = InceptionC(768, channel_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x) # 32+64+64+96=256
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x) # 64+64+64+96=288
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x) # 64+64+64+96=288
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x) # 384+96+288=768
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x) # 192+192+192+192=768
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x) # 192+192+192+192=768
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x) # 192+192+192+192=768
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x) # 192+192+192+192=768
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x) #fc->768
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x) # 320+192+768=1280
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x) #320+384x2+384x2+192=2048
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x) #320+384x2+384x2+192=2048
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1,1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        # N x 1000
        if self.training and self.aux_logits:
            return _InceptionOuputs(x, aux)
        return x



def inception_v3(pretrained=False, progress=True, **kwargs):
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        model = Inception3(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model
    return Inception3(**kwargs)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(3,16,5)#卷积层的参数依次是（输入channels，输出channels，卷积核的大小）
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,32,5)
        self.pool2=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        layer_output = []
        layer_output_feature = []
        x=F.relu(self.conv1(x))   #input(3,32,32)   output(16,28,28)
        layer_output.append(x)
        layer_output_feature.append(x)
        x=self.pool1(x)           #output(16,14,14)
        layer_output.append(x)
        layer_output_feature.append(x)
        x=F.relu(self.conv2(x))   #output(32,10,10)
        layer_output.append(x)
        layer_output_feature.append(x)
        x=self.pool2(x)           #output(32,5,5)
        layer_output.append(x)
        layer_output_feature.append(x)
        x=x.contiguous().view(-1,32*5*5)       #output(32*5*5)
        x=F.relu(self.fc1(x))     #output(120)
        layer_output.append(x)
        x=F.relu(self.fc2(x))     #output(84)
        layer_output.append(x)
        x=self.fc3(x)             #output(10)
        layer_output.append(x)
        return F.softmax(x),layer_output,layer_output_feature
def lenet(pretrained=False, progress=True, **kwargs):
    model = LeNet(**kwargs)
    return model
