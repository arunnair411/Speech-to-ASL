from .unet_parts import *

class GeneratorUnet1_1(nn.Module): # One single encoder branch, one decoder branch with just two final layers for image generation
    def __init__(self, in_chans, out_chans, chans, normalization):
        super(GeneratorUnet1_1, self).__init__()

        self.inc = inconv(in_chans,  chans, normalization)

        self.down1 = down(chans,   2*chans, normalization)
        self.down2 = down(2*chans, 4*chans, normalization )
        self.down3 = down(4*chans, 8*chans, normalization)

        self.down4 = down(8*chans, 8*chans, normalization)

        self.upx1 = up(16*chans, 4*chans, normalization)
        self.upx2 = up(8*chans,  2*chans, normalization)
        self.upx3 = up(4*chans,    chans, normalization)

        self.upx4 = up(2*chans,    chans, normalization)

        self.outcx = outconv(chans, out_chans)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.upx1(x5, x4)
        x = self.upx2(x, x3)
        x = self.upx3(x, x2)
        x = self.upx4(x, x1)
        x = self.outcx(x)

        return x

# NOTE: One big difference between my implementation and the code below is the
# limitation of the FAIR code to bilinear interpolation. However, looking at some
# online implementations (including Deeplab v3, links are below), it seems a standard approach.
# NOTE: The code below makes a cute use of nn.ModuleList, +=, and stack operations
class GeneratorUnet1_1_FAIR(nn.Module):
    """
    PyTorch implementation of a U-Net model from FAIR for fastMRI.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans,normalization,  num_pool_layers=4):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            normalization (str): Normalization scheme used in convlayers <none|batchnorm|instancenorm>.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
        """
        # super().__init__()
        super(GeneratorUnet1_1_FAIR, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.normalization = normalization

        # Maps to self.inc above
        self.down_sample_layers = nn.ModuleList([double_conv(in_chans, chans, normalization)])

        # Maps to self.down1 ... self.down3 above
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [double_conv(ch, ch * 2, normalization)]
            ch *= 2

        # Maps to self.down4 above
        self.conv = double_conv(ch, ch, normalization)

        # Maps to self.upx1 ... self.upx3 above
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [double_conv(ch * 2, ch // 2, normalization)]
            ch //= 2

        # Maps to self.upx4 above
        self.up_sample_layers += [double_conv(ch * 2, ch, normalization)]
        
        # Maps to self.outcx above
        # # NOTE: FAIR conv uses a triple conv here versus single conv as above... Original U-net paper has a single conv...
        # # NOTE: It also does the slightly strange 1x1 convolutions with decreasing number of features...
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(ch, ch // 2, kernel_size=1),
        #     nn.Conv2d(ch // 2, out_chans, kernel_size=1),
        #     nn.Conv2d(out_chans, out_chans, kernel_size=1),
        # )
        # Instead, mapping the code to the model above for now...
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, out_chans, kernel_size=1)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input # NOTE: Here and elsewhere, the assignment appears to copy the value, not creating an issue during backprop

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            # NOTE: https://github.com/pytorch/vision/issues/1708 says setting align_corners=True improves performance, but not conclusive
            # https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py sets align_corners=False
            # https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/deeplab.py sets align_corners=True
            # Regardless, both seem to use linear interpolation for upsampling rather than learned upsampling...
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)
        return self.conv2(output)
