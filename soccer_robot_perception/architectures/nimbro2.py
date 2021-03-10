import torch
import torch.nn as nn
import math
import gin
from torchvision import models

from torchsummary import summary

from soccer_robot_perception.architectures.location_aware_conv2d import LocationAwareConv2d


@gin.configurable
class NimbRoNet2(nn.Module):
    def __init__(self, input_width, input_height, location_awareness):
        super().__init__()
        res18_model = models.resnet18(pretrained=True)
        for param in res18_model.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        location_bias = torch.nn.Parameter(torch.zeros(int(input_width / 4), int(input_height / 4), 3))
        location_encoder = torch.autograd.Variable(torch.ones(int(input_width / 4), int(input_height / 4), 3))


        # TODO: Find how to apply weights for the new layers
        self.block_1 = nn.Sequential(
            *list(res18_model.children())[0:4]
        )  # output: 64 channels
        self.block_2 = nn.Sequential(
            *list(res18_model.children())[4:5]
        )  # output: 64 channels
        self.block_3 = nn.Sequential(
            *list(res18_model.children())[5:6]
        )  # output: 128 channels
        self.block_4 = nn.Sequential(
            *list(res18_model.children())[6:7]
        )  # output: 256 channels
        self.block_5 = nn.Sequential(
            *list(res18_model.children())[7:8]
        )  # output: 512 channels

        self.connector_af_block_2 = nn.Conv2d(
            64, 128, 1
        )  # applies 1x1 convolution to block_2 output
        self.connector_af_block_3 = nn.Conv2d(
            128, 256, 1
        )  # applies 1x1 convolution to block_3 output
        self.connector_af_block_4 = nn.Conv2d(
            256, 256, 1
        )  # applies 1x1 convolution to block_4 output

        self.bn_af_block_2 = nn.BatchNorm2d(128)
        self.bn_af_block_3 = nn.BatchNorm2d(256)
        self.bn_af_block_4 = nn.BatchNorm2d(256)

        self.transpose_convolution_block_512_256 = nn.ConvTranspose2d(
            512, 256, 2, 2
        )  # applies transpose convolution to block_5 output
        self.transpose_convolution_block_256_256 = nn.ConvTranspose2d(
            256, 256, 2, 2
        )  # applies transpose convolution to block_4 and block_5 output
        self.transpose_convolution_block_256_128 = nn.ConvTranspose2d(
            256, 128, 2, 2
        )  # applies transpose convolution to block_3, block_4 and block_5 output

        if location_awareness:
            self.detection_head = LocationAwareConv2d(gradient=False,
                                                      w=int(input_width / 4),
                                                      h=int(input_height / 4),
                                                      location_bias=location_bias,
                                                      location_encoder=location_encoder,
                                                      in_channels=128,
                                                      out_channels=3,
                                                      kernel_size=1,
                                                      )
            self.segmentation_head = LocationAwareConv2d(gradient=False,
                                                         w=int(input_width / 4),
                                                         h=int(input_height / 4),
                                                         location_bias=location_bias,
                                                         location_encoder=location_encoder,
                                                         in_channels=128,
                                                         out_channels=3,
                                                         kernel_size=1,
                                                         )
        else:
            self.detection_head = nn.Conv2d(128, 3, 1, 1)
            self.segmentation_head = nn.Conv2d(128, 3, 1, 1)


    def forward(self, x):
        block_1_out = self.block_1(x)
        block_2_out = self.block_2(block_1_out)
        block_3_out = self.block_3(block_2_out)
        block_4_out = self.block_4(block_3_out)
        block_5_out = self.block_5(block_4_out)

        extractor_af_block_5 = self.relu(block_5_out)
        extractor_af_block_5 = self.transpose_convolution_block_512_256(
            extractor_af_block_5
        )

        concat_af_block_4_out = torch.add(
            extractor_af_block_5, self.connector_af_block_4(block_4_out)
        )
        bn_af_block_4_out = self.bn_af_block_4(self.relu(concat_af_block_4_out))
        ct_af_block_4_out = self.transpose_convolution_block_256_256(bn_af_block_4_out)

        concat_af_block_3_out = torch.add(
            ct_af_block_4_out, self.connector_af_block_3(block_3_out)
        )
        bn_af_block_3_out = self.bn_af_block_3(self.relu(concat_af_block_3_out))
        ct_af_block_3_out = self.transpose_convolution_block_256_128(bn_af_block_3_out)

        concat_af_block_2_out = torch.add(
            ct_af_block_3_out, self.connector_af_block_2(block_2_out)
        )
        bn_af_block_2_out = self.bn_af_block_2(self.relu(concat_af_block_2_out))

        detection_out = self.detection_head(bn_af_block_2_out)
        segmentation_out = self.segmentation_head(bn_af_block_2_out)

        return detection_out, segmentation_out