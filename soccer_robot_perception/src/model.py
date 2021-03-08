from torchsummary import summary
from collections import namedtuple

import torchvision.models as models
import torch
import torch.nn as nn

'''
# location bias convolution
class LocationAwareConv2d(torch.nn.Conv2d):
    def __init__(self, gradient, w, h, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.locationBias = torch.nn.Parameter(torch.zeros(w, h, 3))
        self.locationEncode = torch.autograd.Variable(torch.ones(w, h, 3))
        if gradient:
            for i in range(w):
                self.locationEncode[i, :, 1] = self.locationEncode[:, i, 0] = (i / float(w - 1))

    def forward(self, inputs):
        if self.locationBias.device != inputs.device:
            self.locationBias = self.locationBias.to(inputs.get_device())
        if self.locationEncode.device != inputs.device:
            self.locationEncode = self.locationEncode.to(inputs.get_device())
        b = self.locationBias * self.locationEncode
        return super().forward(inputs) + b[:, :, 0] + b[:, :, 1] + b[:, :, 2]
'''


class NimbRoNet2(nn.Module):
    def __init__(self):
        super(NimbRoNet2, self).__init__()
        res18_model = models.resnet18(pretrained=True)

        self.block_1 = nn.Sequential(*list(res18_model.children())[0:2])  # :: shape [-1, 64, 112, 112]
        self.block_2 = nn.Sequential(
            *list(res18_model.children())[2:5])  # extract output from here : tap1  :: shape [-1, 64, 56, 56]
        self.block_3 = nn.Sequential(
            *list(res18_model.children())[5:6])  # extract output from here : tap2  :: shape [-1, 128, 28, 28]
        self.block_4 = nn.Sequential(
            *list(res18_model.children())[6:7])  # extract output from here : tap3  :: shape [-1, 256, 14, 14]
        self.block_5 = nn.Sequential(
            *list(res18_model.children())[7:8])  # extract output from here : tap4  :: shape [-1, 512, 7, 7]

        self.one_by_one_conv_block_2 = nn.Conv2d(64, 128, 1)  # applies 1x1 convolution to block_2 output
        self.one_by_one_conv_block_3 = nn.Conv2d(128, 256, 1)  # applies 1x1 convolution to block_3 output
        self.one_by_one_conv_block_4 = nn.Conv2d(256, 256, 1)  # applies 1x1 convolution to block_4 output

        self.relu = nn.ReLU()

        self.bn_4_5 = nn.BatchNorm2d(256)
        self.bn_3_4_5 = nn.BatchNorm2d(256)
        self.bn_2_3_4_5 = nn.BatchNorm2d(128)

        self.transpose_convolution_block_512_256 = nn.ConvTranspose2d(512, 256, 2,
                                                                      2)  # applies transpose convolution to block_5 output
        self.transpose_convolution_block_256_256 = nn.ConvTranspose2d(256, 256, 2,
                                                                      2)  # applies transpose convolution to block_4 and block_5 output
        self.transpose_convolution_block_256_128 = nn.ConvTranspose2d(256, 128, 2,
                                                                      2)  # applies transpose convolution to block_3, block_4 and block_5 output

        self.location_dependent_one_one_conv_detection_head = nn.Conv2d(128, 3, 1, 1)
        self.location_dependent_one_one_conv_segmentation_head = nn.Conv2d(128, 3, 1, 1)

    def forward(self, x):
        x = self.block_1(x)
        tap1 = self.block_2(x)
        tap2 = self.block_3(tap1)
        tap3 = self.block_4(tap2)
        tap4 = self.block_5(tap3)

        tap4 = self.relu(tap4)
        tap4 = self.transpose_convolution_block_512_256(tap4)

        cat = torch.add(tap4, self.one_by_one_conv_block_4(tap3))  # concatenates block_4 and block_5 output

        cat = self.bn_4_5(self.relu(cat))
        cat = self.transpose_convolution_block_256_256(cat)

        cat = torch.add(cat, self.one_by_one_conv_block_3(tap2))  # concatenates block_3, block_4 and block_5 output
        cat = self.bn_3_4_5(self.relu(cat))
        cat = self.transpose_convolution_block_256_128(cat)

        cat = torch.add(
            cat, self.one_by_one_conv_block_2(tap1))  # concatenates block_2, block_3, block_4 and block_5 output
        split_junction = self.bn_2_3_4_5(self.relu(cat))
        # return split_junction

        split_junction_detection = self.location_dependent_one_one_conv_detection_head(split_junction)
        split_junction_segmentation = self.location_dependent_one_one_conv_segmentation_head(split_junction)
        # print('det ', split_junction_detection.shape)
        # print('seg ', split_junction_segmentation.shape)
        # return torch.add(split_junction_detection, split_junction_segmentation), dim=1)
        return split_junction_detection, split_junction_segmentation


nimbRoNet = NimbRoNet2()
# input = torch.randn((2, 3, 224, 224))
# outputs = nimbRoNet(input)
# print(summary(nimbRoNet, (3, 224, 224), 8))

# print('output shape is', outputs[0].shape)  # => torch.Size([4, 2048, 7, 7])
# print('output shape is', outputs[1].shape)
# print(outputs)
