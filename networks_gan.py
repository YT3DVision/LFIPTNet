#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os

from torchvision.transforms import ToPILImage

class TransNet(nn.Module):
    def __init__(self, angRes, n_blocks, n_layers, channels):
        super(TransNet, self).__init__()
        # Feature Extraction
        self.AngFE = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False))
        self.SpaFE = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes), bias=False))
        # Spatial-Angular Interaction
        self.CascadeInterBlock = CascadeInterBlock(angRes, n_blocks, n_layers, channels)
        # Condition Feature Extraction
        self.ConditionBlock = ConditionBlock()
        # Fusion and Reconstruction
        self.BottleNeck = BottleNeck(angRes, n_blocks, channels)
        self.ReconBlock = ReconBlock(angRes, channels)

    def forward(self, x, depth, condition):
        print(x.shape)
        xa = self.AngFE(x)
        print(xa.shape)
        xs = self.SpaFE(x)
        print(xs.shape)
        
        condition, conditionimage = self.ConditionBlock(depth,condition)
        # print(conditionimage.shape)
        buffer_a, buffer_s = self.CascadeInterBlock(xa, xs)
        print(buffer_a.shape)
        print(buffer_s.shape)
        buffer_out,depth,centerReslut = self.BottleNeck(buffer_a, buffer_s, condition)
        out = self.ReconBlock(buffer_out)
        print(out.shape)
        # print('ss')
        return out, depth, centerReslut

class make_chains(nn.Module):
    def __init__(self, angRes, channels):
        super(make_chains, self).__init__()

        self.Spa2Ang = nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), padding=0, bias=False)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(channels, int(angRes*angRes*channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.AngConvSq = nn.Conv2d(2*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.SpaConvSq = nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                            padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, xa, xs):
        buffer_ang1 = xa
        buffer_ang2 = self.ReLU(self.Spa2Ang(xs))
        buffer_spa1 = xs
        buffer_spa2 = self.Ang2Spa(xa)
        buffer_a = torch.cat((buffer_ang1, buffer_ang2), 1)
        buffer_s = torch.cat((buffer_spa1, buffer_spa2), 1)
        out_a = self.ReLU(self.AngConvSq(buffer_a)) + xa
        out_s = self.ReLU(self.SpaConvSq(buffer_s)) + xs
        return out_a, out_s


class InterBlock(nn.Module):
    def __init__(self, angRes, n_layers, channels):
        super(InterBlock, self).__init__()
        modules = []
        self.n_layers = n_layers
        for i in range(n_layers):
            modules.append(make_chains(angRes, channels))
        self.chained_layers = nn.Sequential(*modules)

    def forward(self, xa, xs):
        buffer_a = xa
        buffer_s = xs
        for i in range(self.n_layers):
            buffer_a, buffer_s = self.chained_layers[i](buffer_a, buffer_s)
        out_a = buffer_a
        out_s = buffer_s
        return out_a, out_s


class ConditionBlock(nn.Module):
    def __init__(self):
        super(ConditionBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(7, 32, 3, 1, 1),#输入通道 输出通道 卷积核大�?步长 padding
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),#输入通道 输出通道 卷积核大�?步长 padding
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.imageConv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),#输入通道 输出通道 卷积核大�?步长 padding
        )
    def forward(self, center,condition):
        x = torch.cat((center, condition), 1)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        ximage = self.imageConv(x)
        return x, ximage


class CascadeInterBlock(nn.Module):
    def __init__(self, angRes, n_blocks, n_layers, channels):
        super(CascadeInterBlock, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(InterBlock(angRes, n_layers, channels))
        self.body = nn.Sequential(*body)
    def forward(self, buffer_a, buffer_s):
        out_a = []
        out_s = []
        for i in range(self.n_blocks):
            buffer_a, buffer_s = self.body[i](buffer_a, buffer_s)
            out_a.append(buffer_a)
            out_s.append(buffer_s)
        return torch.cat(out_a, 1), torch.cat(out_s, 1)


class BottleNeck(nn.Module):
    def __init__(self, angRes, n_blocks, channels):
        super(BottleNeck, self).__init__()

        self.AngBottle = nn.Conv2d(n_blocks*channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.depthConv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.depthBottle1 = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.depthBottle2 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.AngBottle2 = nn.Conv2d(channels + 32 + 1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.resblock = ResBlock(channels,channels)
        self.Ang2Spa = nn.Sequential(
            nn.Conv2d(3, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes),
        )
        self.SpaBottle = nn.Conv2d((n_blocks+1)*channels, channels, kernel_size=3, stride=1, dilation=int(angRes),
                                    padding=int(angRes), bias=False)
        self.ReLU = nn.ReLU(inplace=True)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, xa, xs, condition):
    
        
        xa = self.ReLU(self.AngBottle(xa))
      
        depth = self.ReLU(self.depthConv(xa))
        depth = self.ReLU(self.depthConv(depth))
        depth = self.ReLU(self.depthBottle1(depth))
        depth = self.Sigmoid(self.depthBottle2(depth))
      
        xa = torch.cat((condition, xa, depth), 1)
        centerReslut = self.ReLU(self.AngBottle2(xa))

        xs = torch.cat((xs, self.Ang2Spa(centerReslut)), 1)
        out = self.ReLU(self.SpaBottle(xs))
        return out,depth,centerReslut

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
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

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ReconBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(ReconBlock, self).__init__()
        self.PreConv = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                                 dilation=int(angRes), padding=int(angRes), bias=False)
        self.PreConv1 = nn.Conv2d(channels, 2*channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.resblock = ResBlock(channels,channels)
        self.CBAM = BasicBlock(channels,channels)
        self.FinalConv = nn.Conv2d(int(channels), 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.angRes = angRes

    def forward(self, x):
        buffer = self.PreConv(x)
        buffer = self.resblock(buffer)
        buffer = self.resblock(buffer)
        buffer = self.resblock(buffer)
        buffer = self.resblock(buffer)
        bufferSAI_LR = MacroPixel2SAI(buffer, self.angRes)
        out = self.FinalConv(bufferSAI_LR)
        return out


def MacroPixel2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out



class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        # self.bn2 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = F.relu(x + output)
        return output


class _netD(nn.Module):
    def __init__(self, use_GPU=True):
        super(_netD, self).__init__()
        self.use_GPU = use_GPU
        self.conv01 = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(6, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv02 = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv0 = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv1 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            nn.Conv2d(64, 128, 4, 2, 1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True))
        # self.conv4 = nn.Sequential(
        #     # state size. (ndf*8) x 8 x 8
        #     nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(0.2, inplace=True))
        self.conv5= nn.Sequential(
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(512, 1,4, 2, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv51= nn.Sequential(
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(1, 1,2, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True))
        self.conv6 = nn.Sequential(
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(1, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x0, condition):
        x = torch.cat((x0, condition), 1)
        # print(x.shape)
        x = self.conv01(x)
        x = self.conv02(x)
        # print(x.shape)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        x = self.conv51(x)
        # print(x.shape)
        output = self.conv6(x)

        return output.view(-1)
