import torch
import torch.nn.functional as F
from torch import nn


class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                (3, 3),
                padding=(1, 1),
                stride=(stride, stride),
                bias=False,
            ),
            nn.InstanceNorm2d(out_ch),  # Replaced BatchNorm2d
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.extract(x)


# --- Start of Correction ---
# The original nn.Sequential structure was causing InstanceNorm2d to be called on a 1x1 tensor.
# By converting it to a standard nn.Module, we can control the forward pass and apply normalization
# after upsampling.
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        size = x.shape[-2:]
        # Apply pooling and convolution
        x = self.pool(x)
        x = self.conv(x)
        # Upsample back to the original size BEFORE applying normalization
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        # Now apply normalization and activation on the full-sized feature map
        x = self.norm(x)
        x = self.relu(x)
        return x


# --- End of Correction ---


class RCSA_ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(RCSA_ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.InstanceNorm2d(dim_out),  # Replaced BatchNorm2d
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True
            ),
            nn.InstanceNorm2d(dim_out),  # Replaced BatchNorm2d
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True
            ),
            nn.InstanceNorm2d(dim_out),  # Replaced BatchNorm2d
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(
                dim_in, dim_out, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True
            ),
            nn.InstanceNorm2d(dim_out),  # Replaced BatchNorm2d
            nn.ReLU(inplace=True),
        )
        self.branch5 = ASPPPooling(dim_in, dim_out)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.InstanceNorm2d(dim_out),  # Replaced BatchNorm2d
            nn.ReLU(inplace=True),
        )
        self.stage1_Conv2 = Conv3Relu(dim_in * 2, dim_in)
        self.stage2_Conv2 = Conv3Relu(dim_in * 2, dim_in)
        self.stage3_Conv2 = Conv3Relu(dim_in * 2, dim_in)
        self.stage4_Conv2 = Conv3Relu(dim_in * 2, dim_in)

    def forward(self, x):
        # The forward pass logic of this module remains unchanged.
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        global_feature = self.branch5(x)
        out45 = self.stage1_Conv2(torch.cat([out4, global_feature], 1))
        out345 = self.stage2_Conv2(torch.cat([out45, out3], 1))
        out2345 = self.stage3_Conv2(torch.cat([out345, out2], 1))
        out12345 = self.stage4_Conv2(torch.cat([out2345, out1], 1))
        result = out12345
        return result
