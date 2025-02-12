import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from . import layers
from .modelio import LoadableModel, store_config_args
from torch.nn import functional as F
import numpy as np
from torch.nn import init
def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

class Unet(nn.Module):
    def __init__(self, inshape, in_channel, out_channel, nb_features=default_unet_features):
        super(Unet, self).__init__()
        
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # 编码器特征
        enc_features = [16, 32, 32, 32]
        # 解码器特征
        dec_features = [32, 32, 32, 32, 32, 16, 16]
        
        self.enc_nf, self.dec_nf = nb_features
        # Encode
        self.conv_encode1 = self.contracting_block(in_channel, enc_features[0])
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(enc_features[0],enc_features[1])
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(enc_features[1], enc_features[2])
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(enc_features[2], enc_features[3])
        self.conv_maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_x = nn.Linear(enc_features[3], 64)    
        self.linear_y = nn.Linear(enc_features[3], 64)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=64, out_channels=128, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=64, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # Decode
        self.conv_decode4 = self.expansive_block(128,dec_features[1],dec_features[2])
        self.conv_decode3 = self.expansive_block(96, dec_features[3], dec_features[4])
        self.conv_decode2 = self.expansive_block(96, dec_features[5], dec_features[6])
        self.final_layer = self.final_block(48, 16, out_channel)

        self._init_weight()
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        return block
    
    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.uniform_(0.0, 1.0)
                m.bias.data.fill_(0)
    def forward(self, x, y):
    # Encode path for x
        x_enc1 = self.conv_encode1(x)
        x_pool1 = self.conv_maxpool1(x_enc1)
        x_enc2 = self.conv_encode2(x_pool1)
        x_pool2 = self.conv_maxpool2(x_enc2)
        x_enc3 = self.conv_encode3(x_pool2)
        x_pool3 = self.conv_maxpool3(x_enc3)
        x_enc4 = self.conv_encode4(x_pool3)
        x_pool4 = self.conv_maxpool4(x_enc4)

    # Projection layer for x
        x_avg_pool = self.avgpool(x_pool4)
        x_avg_pool_flat = torch.flatten(x_avg_pool, 1)
        f_x = self.linear_x(x_avg_pool_flat)
        f_x = f_x / torch.norm(f_x, dim=1, keepdim=True)

    # Encode path for y
        y_enc1 = self.conv_encode1(y)
        y_pool1 = self.conv_maxpool1(y_enc1)
        y_enc2 = self.conv_encode2(y_pool1)
        y_pool2 = self.conv_maxpool2(y_enc2)
        y_enc3 = self.conv_encode3(y_pool2)
        y_pool3 = self.conv_maxpool3(y_enc3)
        y_enc4 = self.conv_encode4(y_pool3)
        y_pool4 = self.conv_maxpool4(y_enc4)

    # Projection layer for y
        y_avg_pool = self.avgpool(y_pool4)
        y_avg_pool_flat = torch.flatten(y_avg_pool, 1)
        f_y = self.linear_y(y_avg_pool_flat)
        f_y = f_y / torch.norm(f_y, dim=1, keepdim=True)

    # Bottleneck
        bottleneck_output = self.bottleneck(torch.cat((x_pool4, y_pool4), 1))

    # Decode path
        dec4 = self.conv_decode4(torch.cat((bottleneck_output, self.crop_and_concat(x_enc4, y_enc4)), 1))
        dec3 = self.conv_decode3(torch.cat((dec4, self.crop_and_concat(x_enc3, y_enc3)), 1))
        dec2 = self.conv_decode2(torch.cat((dec3, self.crop_and_concat(x_enc2, y_enc2)), 1))
        final_output = self.final_layer(torch.cat((dec2, self.crop_and_concat(x_enc1, y_enc1)), 1))

        return final_output




class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
        inshape,
        nb_unet_features=None,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,in_channel=1,out_channel=2,nb_features=nb_unet_features)
        # self.unet_model = Unet(
        #     inshape,
        #     nb_features=nb_unet_features,
        #     nb_levels=nb_unet_levels,
        #     feat_mult=unet_feat_mult
        # )

        # configure unet to flow field layer
        # 最终都会变成2维度的flow图，也就是形变场
        Conv = getattr(nn, 'Conv%dd' % ndims)
        #self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
        self.flow = Conv(2, ndims, kernel_size=3, padding=1)
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        #x = torch.cat([source, target], dim=1)
        x  = self.unet_model(source, target)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, pos_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow

    def predict(self, image, flow, svf=True, **kwargs):

        if svf:
            flow = self.integrate(flow)

            if self.fullsize:
                flow = self.fullsize(flow)

        return self.transformer(image, flow, **kwargs)

    def get_flow_field(self, flow_field):
        if self.integrate:
            flow_field = self.integrate(flow_field)

            # resize to final resolution
            if self.fullsize:
                flow_field = self.fullsize(flow_field)

        return flow_field


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

