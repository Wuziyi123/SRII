import numpy as np
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import init



__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# def data_normal(origin_data):
#     d_min = origin_data.min()
#     if d_min < 0:
#         origin_data += torch.abs(d_min)
#         d_min = origin_data.min()
#     d_max = origin_data.max()
#     dst = d_max - d_min
#     norm_data = (origin_data - d_min).true_divide(dst)
#     return norm_data

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print('输入的shape为:'+str(x.shape))
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #print('avg_out的shape为:' + str(avg_out.shape))
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #print('max_out的shape为:' + str(max_out.shape))
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, count_layer=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.conv1_1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.conv2_1 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.ca = ChannelAttention(planes)
        # self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride
        self.count_layer = count_layer

        # self.mask_print = False
        # self.yanma = []
        # self.channel_weights = []

    def mask(self, gate, out):
        # mask = 1 / (1 + torch.exp(-50 * (gate1 - 0.1)))
        # *************************************************************************
        if self.count_layer <= 9:
            prune_rate = self.count_layer / 9   # 12-30
            # prune_rate = 1.0
        else:
            prune_rate = (18 - self.count_layer) / 9  # 18-4
        # *************************************************************************
        # if self.count_layerd <= 5:
        #     prune_rate = (self.count_layer-1)*2.5+20
        # elif self.count_layer <= 9:
        #     prune_rate = (self.count_layer - 5)*2.5+30
        # elif self.count_layer <= 13:
        #     prune_rate = 40-(self.count_layer - 9)*2.5
        # else:
        #     prune_rate = 30-(self.count_layer - 13)*2.5

        # prune_rate = (18-self.count)/18
        # **************************************************************************

        # mask = torch.add(torch.mul(gate, -50), prune_rate)
        mask = torch.add(torch.mul(gate, -50),  10*prune_rate)
        mask = torch.ones_like(out) / torch.add(torch.exp(mask), 1)
        out = torch.mul(out, mask)

        # out = torch.where(mask1 > 0, out, mask1)
        # out = self.bn1(out)
        # out_x = self.relu(out)
        # gate1 = torch.where(gate1 > 0, gate1, torch.zeros_like(gate1))
        # out_x = out_x * gate1

        # if self.count_layer == 9:
        #     print((mask < 1e-3).nonzero())

        return out, mask

    # def forward(self, x):
    #     residual = x
    #
    #     # if x.shape[2] == 32:
    #     #     level = 2
    #     # elif x.shape[2] == 16:
    #     #     level = 1
    #     # else:
    #     #     level = 0
    #     # prune_rate = 50 * (0.2 + level * 0.05)
    #
    #     out = self.conv1(x)
    #     # a中大于0.5的用zero(0)替换,否则a替换,即不变
    #     # if self.training:
    #     gate = torch.sigmoid(self.conv1_1(x))
    #     out, mask = self.mask(gate, out)
    #
    #     out = self.bn1(out)
    #     out_x = self.relu(out)
    #     # out_x = torch.mul(out_x, mask)
    #
    #     out = self.conv2(out_x)
    #     # if self.training:
    #
    #     # if self.mask_print and self.count_layer == 8:
    #     #     # print("取出mask1！")
    #     #     # if self.count_layer == 10:
    #     #     #     self.yanma = []
    #     #     self.channel_weights.append(self.bn1.weight.data)
    #     #     self.yanma.append(mask)
    #
    #
    #     self.count_layer = self.count_layer + 1
    #     gate = torch.sigmoid(self.conv2_1(out_x))
    #     out, mask = self.mask(gate, out)
    #     out = self.bn2(out)
    #
    #     # if self.mask_print and self.count_layer == 9:
    #     #     print("取出mask2！")
    #     #     self.yanma.append(mask)
    #     #     self.channel_weights.append(self.bn2.weight.data)
    #
    #     if self.downsample is not None:
    #         residual = self.downsample(x)
    #
    #     out += residual
    #     out = self.relu(out)
    #     # out = torch.mul(out, mask)
    #
    #     self.count_layer = self.count_layer-1
    #
    #     torch.cuda.empty_cache()
    #
    #     return out

    def forward(self, x):
        residual = x

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.count_layer = 0
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], count=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, count=6)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, count=10)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, count=14)
        self.feature = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.conv_gff1 = conv3x3(in_planes=64, out_planes=256, stride=4)
        self.conv_gff2 = conv3x3(in_planes=128, out_planes=256, stride=2)
        # self.conv_gff2 = conv3x3(in_planes=128, out_planes=128, stride=1)
        self.conv_gff3 = conv3x3(in_planes=256, out_planes=256, stride=1)
        self.conv_gff1_1 = conv3x3(in_planes=256, out_planes=256, stride=1)
        self.conv_gff2_1 = conv3x3(in_planes=256, out_planes=256, stride=1)
        self.conv_gff3_1 = conv3x3(in_planes=256, out_planes=256, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, count=0):
        # self.count_layer = self.count_layer + 1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # block.count = self.count_layer
        layers.append(block(self.inplanes, planes, stride, downsample, count_layer=count))
        self.inplanes = planes * block.expansion
        # self.count_layer = self.count_layer + 2
        for i in range(1, blocks):
            # block.count = self.count_layer
            layers.append(block(self.inplanes, planes, count_layer=count+2))
            # self.count_layer = self.count_layer + 2

        return nn.Sequential(*layers)


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_fusion(self, x, fusion_from_down, a):
        x1n_down, g1_down, x2n_down, g2_down = fusion_from_down[0], fusion_from_down[1], fusion_from_down[2], \
                                               fusion_from_down[3]
        x3n_down, g3_down = fusion_from_down[4], fusion_from_down[5]

        # l1_down, l2_down, l3_down = fusion_from_down[0], fusion_from_down[1], fusion_from_down[2]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        layer1 = x

        # g1 = self.conv_gff1(layer1)
        # g1 = torch.sigmoid(g1)

        x = self.layer2(x)
        layer2 = x

        # l2 = layer2.clone().detach()
        # x2n = self.conv_gff2(l2)
        # g2 = self.conv_gff2_1(x2n)
        # g2 = torch.tanh(g2)

        # g2 = self.conv_gff2(layer2)
        # g2 = self.bn_g2_up(g2)
        # g2 = torch.sigmoid(g2)
        # g2_down = self.conv_gff2(l2_down)
        # g2_down = torch.sigmoid(g2_down)


        x = self.layer3(x)
        layer3 = x

        # x3n = layer3.clone().detach()
        # layer1 - x1n - g1
        # x1n = self.conv_gff1(layer1)
        # g1 = self.conv_gff1_1(x1n)
        # g1 = torch.tanh(g1)


        # x3n = self.conv_gff3(layer3)  # layer3-x3n-g3
        # l3 = layer3 + x2n + x1n
        g3 = self.conv_gff3(layer3)
        g3 = torch.sigmoid(g3)
        # g3_down = self.conv_gff3(l3_down)
        # g3_down = torch.sigmoid (g3_down)

        # x3n = self.conv_gff3(layer3)  # layer3-x3n-g3
        # g3 = self.conv_gff3_1(x3n)
        # g3 = torch.sigmoid(g3)

        # g3_down = self.conv_gff3_down(x3n_down)
        # g3_down = torch.tanh(g3_down)

        # 增量训练中添加的融合内容
        # if a > 0:
        #     after_fusion = (1 - a) * (1 + g3) * x3n + a * (1 - g3) * \
        #                    (g1 * x1n + g2 * x2n + g1_down * x1n_down + g2_down * x2n_down + g3_down * x3n_down)
        # else:
        #     after_fusion = (1 + g3) * x3n + (1 - g3) * (g1 * x1n + g2 * x2n)

        if a > 0:
            after_fusion = (1 - a) * (1 + g3) * layer3 + a * (1 - g3) * \
                           (g3_down * x3n_down) + 0.03 * (1 - g3) * g2 * x2n
        else:
            after_fusion = layer3

        x = self.layer4(after_fusion)

        x = self.feature(x)
        x = x.view(x.size(0), -1)

        return x

    # def forward_fusion(self, x, fusion_from_down, a):
    #     ly1_down, ly2_down, ly3_down = fusion_from_down[0], fusion_from_down[1], fusion_from_down[2]
    #
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.layer1(x)
    #
    #     # 增量训练中添加的融合内容
    #     if a > 0:
    #         x = (1 - a) * x + a * ly1_down
    #
    #     x1n = self.conv_gff2(x)
    #     x = self.layer2(x)
    #
    #     # 增量训练中添加的融合内容
    #     if a > 0:
    #         x = (1 - a) * x + a * ly2_down
    #
    #     # 增量训练中添加的融合内容
    #     after_fusion1 = torch.cat((x1n, x), dim=1)
    #     x = self.conv_fusion1(after_fusion1)
    #
    #     x2n = self.conv_gff3(x)
    #     x = self.layer3(x)
    #
    #     # 增量训练中添加的融合内容
    #     if a > 0:
    #         x = (1 - a) * x + a * ly3_down
    #
    #     # 增量训练中添加的融合内容
    #     after_fusion2 = torch.cat((x2n, x), dim=1)
    #     x = self.conv_fusion2(after_fusion2)
    #
    #     x = self.layer4(x)
    #     x = self.feature(x)
    #     x = x.view(x.size(0), -1)
    #
    #     return x


    def forward(self, x, fusion_material_from_downBranch=None, a=0.5, getValue=False, once=True):
        if once:
            x = self.forward_once(x)
        else:
            x = self.forward_fusion(x, fusion_material_from_downBranch, a)
        return x



def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model