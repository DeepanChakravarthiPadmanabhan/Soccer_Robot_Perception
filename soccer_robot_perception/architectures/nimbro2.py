import torch
import torch.nn as nn
import math
import gin
from torchvision import models

from torchsummary import summary

from soccer_robot_perception.architectures.location_aware_conv2d import (
    LocationAwareConv2d,
)


@gin.configurable
class NimbRoNet2(nn.Module):
    def __init__(self, input_width=640, input_height=480, location_awareness=True, resnet_bb_fixed=True):
        super().__init__()
        res18_model = models.resnet18(pretrained=True)

        if resnet_bb_fixed:
            for param in res18_model.parameters():
                param.requires_grad = False

        location_bias = torch.nn.Parameter(torch.zeros(120, 160, 3))

        # TODO: Find how to apply weights for the new layers
        self.encoder_block1 = nn.Sequential(*list(res18_model.children())[0:5])
        self.encoder_block2 = nn.Sequential(*list(res18_model.children())[5:6])
        self.encoder_block3 = nn.Sequential(*list(res18_model.children())[6:7])
        self.encoder_block4 = nn.Sequential(*list(res18_model.children())[7:-2])

        self.decoder_block1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2, 0, output_padding=0),
        )

        self.decoder_block2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 2, 2, 0, output_padding=0),
        )

        self.decoder_block3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 128, 2, 2, 0, output_padding=0),
        )

        self.relu_final = nn.ReLU()
        self.bn_final = nn.BatchNorm2d(256)

        self.detection_head = LocationAwareConv2d(
                gradient=True,
                w=120,
                h=160,
                location_bias=location_bias,
                in_channels=256,
                out_channels=3,
                kernel_size=1,
            )

        self.segmentation_head = LocationAwareConv2d(
                gradient=True,
                w=120,
                h=160,
                location_bias=location_bias,
                in_channels=256,
                out_channels=3,
                kernel_size=1,
            )

        self.conv1x1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)

    def forward(self, x):
        out = self.encoder_block1(x)
        residual1 = self.conv1x1_1(out)

        out = self.encoder_block2(out)
        residual2 = self.conv1x1_2(out)

        out = self.encoder_block3(out)
        residual3 = self.conv1x1_3(out)

        out = self.encoder_block4(out)

        out = self.decoder_block1(out)

        decoder_block2_input = torch.cat((out, residual3), 1)
        out = self.decoder_block2(decoder_block2_input)

        decoder_block3_input = torch.cat((out, residual2), 1)
        out = self.decoder_block3(decoder_block3_input)

        decoder_block4_input = torch.cat((out, residual1), 1)
        head_inputs = self.relu_final(decoder_block4_input)
        head_inputs = self.bn_final(head_inputs)
        det_out = self.detection_head(head_inputs)
        seg_out = self.segmentation_head(head_inputs)

        return det_out, seg_out
