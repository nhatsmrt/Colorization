import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
import torchvision
from torchvision.models import vgg19_bn
from torchvision.models.vgg import model_urls

import numpy as np
import matplotlib.pyplot as plt
import cv2
import random



class ResidualBlock(nn.Module):

    def __init__(self, inp_channel):
        super(ResidualBlock, self).__init__()
        self._conv_layer = ConvolutionalLayer(inp_channel, inp_channel, 3, padding = 1)
        self._relu = nn.LeakyReLU(inplace = True)
        self._batch_norm = nn.BatchNorm2d(num_features = inp_channel)


    def forward(self, x):
        residual = x

        conv_1 = self._conv_layer(x)
        conv_2 = self._conv_layer(conv_1)

        res = self._relu(residual + conv_2)
        res_h = self._batch_norm(res)

        return res_h

class ConvolutionalLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0, bias = True):
        super(ConvolutionalLayer, self).__init__()
        self._conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias
        )
        self._relu = nn.LeakyReLU(inplace = True)
        self._batch_norm = nn.BatchNorm2d(num_features = out_channels)


    def forward(self, x):

        conv = self._conv(x)
        a = self._relu(conv)
        h = self._batch_norm(a)

        return h

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class ResizeConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResizeConvolutionalLayer, self).__init__()
        self._conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            padding = 1
        )

    def forward(self, input, out_h, out_w):
        upsampled = F.interpolate(input, size = (out_h, out_w), mode = 'bilinear')
        conv = self._conv(upsampled)
        return conv


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()

        self._main = nn.Sequential(
            nn.BatchNorm2d(num_features = in_channels),
            nn.ReLU(inplace = True),
            ConvolutionalLayer(
                in_channels = in_channels,
                out_channels = growth_rate,
                kernel_size = 1,
                stride = 1,
                bias = False
            ),
            nn.Conv2d(
                in_channels = growth_rate,
                out_channels = growth_rate,
                kernel_size = 3,
                stride = 1,
                padding=1,
                bias = False
            )
        )

    def forward(self, input):
        return torch.cat((input, self._main(input)), dim = 1)

class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module(
                "DenseLayer_" + str(i),
                DenseLayer(in_channels = in_channels + growth_rate * i, growth_rate = growth_rate)
            )

class Reshape(nn.Module):
    def forward(self, input, new_shape):
        return input.view(new_shape)

# Input: N * C * H * W
def spatial_pyramid_pool(input, op_sizes, pool_layer = nn.MaxPool2d):
    ops = []
    batch_size = input.shape[0]
    inp_c = input.shape[1]
    inp_h = input.shape[2]
    inp_w = input.shape[3]

    for size in op_sizes:
        pool = pool_layer(
            kernel_size = (torch.ceil(inp_h / size), torch.ceil(inp_w / size)),
            stride = (torch.floor(inp_h / size), torch.floor(inp_w / size))
        )
        ops.append(pool(input).view(batch_size, inp_c, -1))

    return torch.cat(ops, dim = -1)



# based onhttps://github.com/chenyuntc/pytorch-book/blob/master/chapter8-%E9%A3%8E%E6%A0%BC%E8%BF%81%E7%A7%BB(Neural%20Style)/PackedVGG.py
class PretrainedModel(nn.Module):
    def __init__(self, model = vgg19_bn, last_layer = None, fine_tune = True):
        super(PretrainedModel, self).__init__()
        model = model(pretrained = True)
        if not fine_tune:
            for param in model.parameters():
                param.requires_grad = False

        self._features = list(model.features)
        if last_layer is not None:
            self._features = self._features[:last_layer]

    def forward(self, input, layers = None):
        op = []

        for ind in range(len(self._features)):
            input = self._features[ind](input)

            if layers is not None:
                if ind in layers:
                    op.append(input)

                if ind >= max(layers):
                    break
            else:
                if ind == len(self._features) - 1:
                    op.append(input)


        return op

class FusionLayer(nn.Module):
    def __init__(self):
        super(FusionLayer, self).__init__()
        self._conv = ConvolutionalLayer(
            in_channels = 256 + 16,
            out_channels = 256
        )

    def forward(self, encoded, features):

        if encoded.shape[2] % features.shape[2] != 0 or encoded.shape[3] % features.shape[3] != 0:
            h_pad = (features.shape[2] - encoded.shape[2] % features.shape[2]) % features.shape[2]
            w_pad = (features.shape[3] - encoded.shape[3] % features.shape[3]) % features.shape[3]
            pad_size = (h_pad // 2, h_pad - h_pad // 2, w_pad // 2, w_pad - w_pad // 2)
            encoded = F.pad(encoded, pad_size)


        features = features.repeat(
            1, 1,
            encoded.shape[2] // features.shape[2],
            encoded.shape[3] // features.shape[3]
        )
        concat = torch.cat((encoded, features), dim = 1)
        return self._conv(concat)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self._main = nn.Sequential(
            ConvolutionalLayer(
                in_channels = 1,
                stride = 2,
                out_channels = 8
            ),
            ResidualBlock(inp_channel = 8),
            ConvolutionalLayer(
                in_channels = 8,
                stride = 2,
                out_channels = 16
            ),
        )

    def forward(self, input):
        return self._main(input)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self._upsample1 = ResizeConvolutionalLayer(
            in_channels = 256,
            out_channels = 64
        )
        self._res1 = ResidualBlock(
            inp_channel = 64
        )
        self._upsample2 = ResizeConvolutionalLayer(
            in_channels = 64,
            out_channels = 2
        )



    def forward(self, input, img_h = None, img_w = None):

        upsample1 = F.relu(self._upsample1(input, img_h // 2, img_w // 2))
        res1 = self._res1(upsample1)
        return torch.tanh(self._upsample2(res1, img_h, img_w))




