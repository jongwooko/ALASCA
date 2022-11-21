import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out//4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

def branchMLP(channel_in, channel_out):
    middle_channel = channel_out//4
    return nn.Sequential(
            conv1x1(channel_in, channel_in, stride=8),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),
        )

def invertedBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out * 2
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

class BatchNorm2dMul(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2dMul, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=track_running_stats)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.affine = affine

    def forward(self, x):
        bn_out = self.bn(x)
        if self.affine:
            out = self.gamma[None, :, None, None] * bn_out + self.beta[None, :, None, None]
        return out, bn_out
    
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dMul(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dMul(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        bn_outputs = []
        
        residual = x
        output = self.conv1(x)
        output, bn_out = self.bn1(output)
        bn_outputs.append(bn_out)
        output = self.relu(output)

        output = self.conv2(output)
        output, bn_out = self.bn2(output)
        bn_outputs.append(bn_out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        output += residual
        output = self.relu(output)
        return output, bn_outputs

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output
    
class LayerBlock(nn.Module):
    def __init__(self, block, inplanes, planes, num_blocks, stride):
        super(LayerBlock, self).__init__()
        downsample = None
        if stride !=1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer = []
        layer.append(block(inplanes, planes, stride=stride, downsample=downsample))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layer.append(block(inplanes, planes))
        self.layers = nn.Sequential(*layer)
        
    def forward(self, x):
        bn_outputs = []
        for layer in self.layers:
            x, bn_output = layer(x)
            bn_outputs.extend(bn_output)
        return x, bn_outputs
    
class SDResNet(nn.Module):
    """
    Resnet model
    
    Args:
        block (class): block type, BasicBlock or BottlenetckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """
    
    def __init__(self, block, layers, num_classes=10, position_all=True):
        super(SDResNet, self).__init__()
        
        self.position_all = position_all
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = LayerBlock(block, 64, 64, layers[0], stride=1)
        self.layer2 = LayerBlock(block, 64, 128, layers[1], stride=2)
        self.layer3 = LayerBlock(block, 128, 256, layers[2], stride=2)
        self.layer4 = LayerBlock(block, 256, 512, layers[3], stride=2)

        self.downsample1_1 = nn.Sequential(
                            conv1x1(64 * block.expansion, 512 * block.expansion, stride=8),
                            nn.BatchNorm2d(512 * block.expansion),
                            nn.ReLU(),
        )
        self.bottleneck1_1 = branchBottleNeck(64 * block.expansion, 512 * block.expansion, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)


        self.downsample2_1 = nn.Sequential(
                            conv1x1(128 * block.expansion, 512 * block.expansion, stride=4),
                            nn.BatchNorm2d(512 * block.expansion),
            )
        self.bottleneck2_1 = branchBottleNeck(128 * block.expansion, 512 * block.expansion, kernel_size=4)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)


        self.downsample3_1 = nn.Sequential(
                            conv1x1(256 * block.expansion, 512 * block.expansion, stride=2),
                            nn.BatchNorm2d(512 * block.expansion),
        )
        self.bottleneck3_1 = branchBottleNeck(256 * block.expansion, 512 * block.expansion, kernel_size=2)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.apply(_weights_init)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers
        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer = []
        layer.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            layer.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layer)
    
    def forward(self, x):
        all_bn_outputs = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x, bn_outputs = self.layer1(x)
        all_bn_outputs.extend(bn_outputs)
        middle_output1 = self.bottleneck1_1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)

        x, bn_outputs = self.layer2(x)
        all_bn_outputs.extend(bn_outputs)
        middle_output2 = self.bottleneck2_1(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)

        x, bn_outputs = self.layer3(x)
        all_bn_outputs.extend(bn_outputs)
        middle_output3 = self.bottleneck3_1(x)
        middle_output3 = self.avgpool3(middle_output3)
        middle3_fea = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)

        x, bn_outputs = self.layer4(x)
        all_bn_outputs.extend(bn_outputs)
        x = self.avgpool(x)
        final_fea = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.position_all:
            return {'outputs': [x, middle_output1, middle_output2, middle_output3],
                'features': [final_fea, middle1_fea, middle2_fea, middle3_fea],
                'bn_outputs': all_bn_outputs}
        else:
            return {'outputs': [x, middle_output1],
                    'features': [final_fea, middle1_fea, middle2_fea, middle3_fea],
                    'bn_outputs': all_bn_outputs}
        
class SDResNet_mlp(nn.Module):
    """
    Resnet model
    
    Args:
        block (class): block type, BasicBlock or BottlenetckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """
    
    def __init__(self, block, layers, num_classes=10, position_all=True):
        super(SDResNet_mlp, self).__init__()
        
        self.position_all = position_all
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = LayerBlock(block, 64, 64, layers[0], stride=1)
        self.layer2 = LayerBlock(block, 64, 128, layers[1], stride=2)
        self.layer3 = LayerBlock(block, 128, 256, layers[2], stride=2)
        self.layer4 = LayerBlock(block, 256, 512, layers[3], stride=2)

        self.downsample1_1 = nn.Sequential(
            conv1x1(64 * block.expansion, 512 * block.expansion),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),
        )
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)


        self.downsample2_1 = nn.Sequential(
            conv1x1(128 * block.expansion, 512 * block.expansion),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU()
            )
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)


        self.downsample3_1 = nn.Sequential(
            conv1x1(256 * block.expansion, 512 * block.expansion),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU()
        )
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.apply(_weights_init)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers
        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer = []
        layer.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            layer.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layer)
    
    def forward(self, x):
        all_bn_outputs = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x, bn_outputs = self.layer1(x)
        all_bn_outputs.extend(bn_outputs)
#         middle_output1 = self.downsample1_1(x)
#         middle_output1 = self.avgpool1(middle_output1)
#         middle1_fea = middle_output1
#         middle_output1 = torch.flatten(middle_output1, 1)
#         middle_output1 = self.middle_fc1(middle_output1)

        x, bn_outputs = self.layer2(x)
        all_bn_outputs.extend(bn_outputs)
#         middle_output2 = self.downsample2_1(x)
#         middle_output2 = self.avgpool2(middle_output2)
#         middle2_fea = middle_output2
#         middle_output2 = torch.flatten(middle_output2, 1)
#         middle_output2 = self.middle_fc2(middle_output2)

        x, bn_outputs = self.layer3(x)
        all_bn_outputs.extend(bn_outputs)
#         middle_output3 = self.downsample3_1(x)
#         middle_output3 = self.avgpool3(middle_output3)
#         middle3_fea = middle_output3
#         middle_output3 = torch.flatten(middle_output3, 1)
#         middle_output3 = self.middle_fc3(middle_output3)

        x, bn_outputs = self.layer4(x)
        all_bn_outputs.extend(bn_outputs)
        x = self.avgpool(x)
        final_fea = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.position_all:
            return {'outputs': [x, middle_output1, middle_output2, middle_output3],
                'bn_outputs': all_bn_outputs}
        else:
            return {'outputs': [x, x],
                    'bn_outputs': all_bn_outputs}
        
class SDResNet_residual(nn.Module):
    """
    Resnet model
    
    Args:
        block (class): block type, BasicBlock or BottlenetckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """
    
    def __init__(self, block, layers, num_classes=10, position_all=True):
        super(SDResNet_residual, self).__init__()
        
        self.position_all = position_all
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = LayerBlock(block, 64, 64, layers[0], stride=1)
        self.layer2 = LayerBlock(block, 64, 128, layers[1], stride=2)
        self.layer3 = LayerBlock(block, 128, 256, layers[2], stride=2)
        self.layer4 = LayerBlock(block, 256, 512, layers[3], stride=2)
        
        self.bottleneck1_1 = LayerBlock(block, 64, 512, 1, stride=8)
#         branchBottleNeck(64 * block.expansion, 512 * block.expansion, kernel_size=8)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)

        self.bottleneck2_1 = LayerBlock(block, 128, 512, 1, stride=4)
#         branchBottleNeck(128 * block.expansion, 512 * block.expansion, kernel_size=4)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)


#         self.downsample3_1 = nn.Sequential(
#                             conv1x1(256 * block.expansion, 512 * block.expansion, stride=2),
#                             nn.BatchNorm2d(512 * block.expansion),
#         )
        self.bottleneck3_1 = LayerBlock(block, 256, 512, 1, stride=2)
        self.avgpool3 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.apply(_weights_init)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers
        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layer = []
        layer.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            layer.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layer)
    
    def forward(self, x):
        all_bn_outputs = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x, bn_outputs = self.layer1(x)
        all_bn_outputs.extend(bn_outputs)
        middle_output1, _ = self.bottleneck1_1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)

        x, bn_outputs = self.layer2(x)
        all_bn_outputs.extend(bn_outputs)
        middle_output2, _ = self.bottleneck2_1(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)

        x, bn_outputs = self.layer3(x)
        all_bn_outputs.extend(bn_outputs)
        middle_output3, _ = self.bottleneck3_1(x)
        middle_output3 = self.avgpool3(middle_output3)
        middle3_fea = middle_output3
        middle_output3 = torch.flatten(middle_output3, 1)
        middle_output3 = self.middle_fc3(middle_output3)

        x, bn_outputs = self.layer4(x)
        all_bn_outputs.extend(bn_outputs)
        x = self.avgpool(x)
        final_fea = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.position_all:
            return {'outputs': [x, middle_output1, middle_output2, middle_output3],
                'features': [final_fea, middle1_fea, middle2_fea, middle3_fea],
                'bn_outputs': all_bn_outputs}
        else:
            return {'outputs': [x, middle_output3],
                    'features': [final_fea, middle1_fea, middle2_fea, middle3_fea],
                    'bn_outputs': all_bn_outputs}
    
class SDResNet_s(nn.Module):
    """
    Resnet model small
    
    Args:
        block (class): block type, BasicBlock or BottlenetckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """
    
    def __init__(self, block, layers, num_classes=10):
        super(SDResNet_s, self).__init__()
        
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.downsample1_1 = nn.Sequential(
                            conv1x1(16 * block.expansion, 64 * block.expansion, stride=4),
                            nn.BatchNorm2d(64 * block.expansion),
        )
        self.bottleneck1_1 = branchBottleNeck(16 * block.expansion, 64 * block.expansion, kernel_size=4)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc1 = nn.Linear(64 * block.expansion, num_classes)


        self.downsample2_1 = nn.Sequential(
                            conv1x1(32 * block.expansion, 64 * block.expansion, stride=2),
                            nn.BatchNorm2d(64 * block.expansion),
            )
        self.bottleneck2_1 = branchBottleNeck(32 * block.expansion, 64 * block.expansion, kernel_size=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        self.middle_fc2 = nn.Linear(64 * block.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers
        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        strides = [stride] + [1]*(layers-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        middle_output1 = self.bottleneck1_1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle1_fea = middle_output1
        middle_output1 = torch.flatten(middle_output1, 1)
        middle_output1 = self.middle_fc1(middle_output1)

        x = self.layer2(x)
        middle_output2 = self.bottleneck2_1(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle2_fea = middle_output2
        middle_output2 = torch.flatten(middle_output2, 1)
        middle_output2 = self.middle_fc2(middle_output2)

        x = self.layer3(x)
        x = self.avgpool(x)
        final_fea = x
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return {'outputs': [x, middle_output1, middle_output2],
                'features': [final_fea, middle1_fea, middle2_fea]}
    
def sdresnet18(num_classes=10):
    return SDResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def sdresnet34(num_classes=10, position_all=True):
    return SDResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, position_all=position_all)

def sdresnet34_mlp(num_classes=10, position_all=True):
    return SDResNet_mlp(BasicBlock, [3,4,6,3], num_classes=num_classes, position_all=position_all)

def sdresnet34_residual(num_classes=10, position_all=True):
    return SDResNet_residual(BasicBlock, [3,4,6,3], num_classes=num_classes, position_all=position_all)

def sdresnet32(num_classes=10):
    return SDResNet_s(BasicBlock_s, [5,5,5], num_classes=num_classes)
