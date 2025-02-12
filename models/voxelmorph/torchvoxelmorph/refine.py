import torch
import torch.nn as nn
import torch.nn.functional as F
from models.voxelmorph.torchvoxelmorph.layers import disp_warp, SpatialTransformer


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
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


class RefinementNet(nn.Module):
    def __init__(self):   #, inshape
        super().__init__()
       
        #self.spatialTransformer = SpatialTransformer(inshape)
        
        # Target image and error
        in_channels = 4  

        self.conv1 = conv2d(in_channels, 16)
        self.conv2 = conv2d(4, 32)  # Assuming the deformation field has 2 channels (for 2D images)

        self.dilation_list = [1, 2, 4, 8, 1, 1]
        self.dilated_blocks = nn.ModuleList()

        for dilation in self.dilation_list:
            self.dilated_blocks.append(BasicBlock(32, 32, stride=1, dilation=dilation))

        self.dilated_blocks = nn.Sequential(*self.dilated_blocks)

        self.final_conv = nn.Conv2d(32, 2, 3, 1, 1)  # Output 2 channels for the refined deformation field

    def forward(self, orig_deform_field, target_img, moving_img):
        # Compute error between warped moving image and target image
        #warped_moving = warp_with_field(moving_img, orig_deform_field)  # This function needs to be defined
        #device = orig_deform_field.device
        #self.spatialTransformer.to(device)
        warped_moving = disp_warp(moving_img, orig_deform_field)[0]     #self.spatialTransformer
        error = warped_moving - target_img  # [B, C, H, W]

        concat = torch.cat((error, target_img, orig_deform_field), dim=1)  # [B, 6, H, W]
        conv_out = self.conv2(concat)  # [B, 16, H, W]
        out = self.dilated_blocks(conv_out)  # [B, 32, H, W]
        residual_field = self.final_conv(out)  # [B, 2, H, W]

        refined_deform_field = orig_deform_field + residual_field  # Add the residual to the original deformation field
        return refined_deform_field


        # concat1 = torch.cat((error, target_img), dim=1)  # [B, 6, H, W]
        # conv1 = self.conv1(concat1)  # [B, 16, H, W]
        # conv2 = self.conv2(orig_deform_field)  # [B, 16, H, W]
        # concat2 = torch.cat((conv1, conv2), dim=1)  # [B, 32, H, W]

        # out = self.dilated_blocks(concat2)  # [B, 32, H, W]
        # residual_field = self.final_conv(out)  # [B, 2, H, W]

        # refined_deform_field = orig_deform_field + residual_field  # Add the residual to the original deformation field
        # return refined_deform_field



