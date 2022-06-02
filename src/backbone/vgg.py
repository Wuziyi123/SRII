import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import init



__all__ = ['vgg', 'resnet18_cbam']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class vgg(nn.Module):

    def __init__(self, num_classes=100, dataset='cifar10', init_weights=True, cfg=None):
        super(vgg, self).__init__()
        print("正在初始化!")
        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        self.feature = self.make_layers(cfg, True)

        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'cifar10':
            num_classes = 10
        self.fc = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        # y = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(0.5)
                init.normal_(m.weight, mean=0, std=1)
                m.bias.data.zero_()

    def forward(self, x, fusion_material_from_downBranch=None, a=0.5, getValue=False, once=True, useMask=False):
        if once and not useMask:
            x = self.forward_once(x)
        elif once and useMask:
            x = self.forward_useMask(x)
        else:
            x = self.forward_fusion(x, fusion_material_from_downBranch, a)
        return x

    # for m in self.modules():
    #     if isinstance(m, nn.Conv2d):
    #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     elif isinstance(m, nn.BatchNorm2d):
    #         init.normal_(m.weight, mean=0, std=1)
    #         m.bias.data.zero_()

def VGG(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model = vgg()
    # if pretrained:
    #     pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
    #     now_state_dict        = model.state_dict()
    #     now_state_dict.update(pretrained_state_dict)
    #     model.load_state_dict(now_state_dict)
    return model
