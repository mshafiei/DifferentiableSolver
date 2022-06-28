""" Full assembly of the parts to form the complete network """
""" https://raw.githubusercontent.com/milesial/Pytorch-UNet """
from cvgutils.nn.jaxUtils.unet_parts import *


class UNet(nn.Module):
    n_channels : int
    n_classes : int
    bilinear : bool
    test: bool
    group_norm: bool
    num_groups: int
    thickness : int
    activation: str
    main_model: str
    kernel_size: int
    unet_factor: int
    high_dim: bool
    outc_kernel_size: int
    
    
    def down(self,inp):
        out0 = self.down_out0(inp)
        out1 = self.down_out1(out0)
        out2 = self.down_out2(out1)
        out3 = self.down_out3(out2)
        out4 = self.down_out4(out3)

        return out4, [out0, out1, out2, out3]

    def up(self,inp,skips):
        out1 = self.up_out0(inp,  skips[-1])
        out2 = self.up_out1(out1, skips[-2])
        out3 = self.up_out2(out2, skips[-3])
        out4 = self.up_out3(out3, skips[-4])
        return  self.up_out4(out4)
    
    def encode(self,inp):
        in1 =  self.encoder_in0(inp)
        in2 =  self.encoder_in1(in1)
        in3 =  self.encoder_in2(in2)
        in4 =  self.encoder_in3(in3)
        return self.encoder_in4(in4)

    def decode(self,inp):
        out1 = self.decoder_out0(inp)
        out2 = self.decoder_out1(out1)
        out3 = self.decoder_out2(out2)
        out4 = self.decoder_out3(out3)
        return self.decoder_out4(out4)
    
    def setup(self):
        
        self.down_out0 = DoubleConv(self.n_channels,self.thickness, self.thickness,self.test,self.group_norm,self.num_groups,self.activation)
        self.down_out1 = Down(self.n_channels, self.thickness*2,self.test,self.group_norm,self.num_groups,self.activation)
        self.down_out2 = Down(self.thickness*2, self.thickness*4,self.test,self.group_norm,self.num_groups,self.activation)
        self.down_out3 = Down(self.thickness*4, self.thickness*8,self.test,self.group_norm,self.num_groups,self.activation)
        self.down_out4 = Down(self.thickness*8, self.thickness*16 // self.unet_factor,self.test,self.group_norm,self.num_groups,self.activation)
        self.bottleneck = DoubleConv(self.thickness*16, self.thickness*16// self.unet_factor, self.thickness*16// self.unet_factor,self.test,      self.group_norm,self.num_groups,self.activation)
        self.up_out0 = Up(self.thickness*(16// self.unet_factor+8),  self.thickness*8 // self.unet_factor, self.bilinear,self.test,self.group_norm,  self.num_groups,self.activation)
        self.up_out1 = Up(self.thickness*(8// self.unet_factor+4),   self.thickness*4 // self.unet_factor, self.bilinear,self.test,self.group_norm,  self.num_groups,self.activation)
        self.up_out2 = Up(self.thickness*(4// self.unet_factor+2),   self.thickness*2 // self.unet_factor, self.bilinear,self.test,self.group_norm,  self.num_groups,self.activation)
        self.up_out3 = Up(self.thickness*(2// self.unet_factor+1),   self.thickness,     self.bilinear,    self.test,self.group_norm,self.num_groups,self.activation)
        self.up_out4 = OutConv(self.thickness, self.n_classes,'SAME',self.outc_kernel_size)

        if('fft_highdim' in self.main_model):
            self.encoder_in0 = DoubleConv(9,self.thickness, self.thickness, self.test,     self.group_norm,self.num_groups,self.activation)
            self.encoder_in1 = DoubleConv(  self.thickness, self.thickness, self.thickness,self.test,      self.group_norm,self.num_groups,self.activation)
            self.encoder_in2 = DoubleConv(  self.thickness, self.thickness, self.thickness,self.test,      self.group_norm,self.num_groups,self.activation)
            self.encoder_in3 = DoubleConv(  self.thickness, self.thickness, self.thickness,self.test,      self.group_norm,self.num_groups,self.activation)
            self.encoder_in4 = DoubleConv( self.thickness, self.thickness, self.thickness,self.test,      self.group_norm,self.num_groups,self.activation)

            self.decoder_out0 = DoubleConv(self.thickness,self.thickness, self.thickness,self.test,self.group_norm,self.num_groups,self.activation)
            self.decoder_out1 = DoubleConv(self.thickness,self.thickness, self.thickness,self.test,self.group_norm,self.num_groups,self.activation)
            self.decoder_out2 = DoubleConv(self.thickness,self.thickness, self.thickness,self.test,self.group_norm,self.num_groups,self.activation)
            self.decoder_out3 = DoubleConv(self.thickness,self.thickness, self.thickness,self.test,self.group_norm,self.num_groups,self.activation)
            self.decoder_out4 = OutConv(   self.thickness, 3,'SAME',3)

        if('filter' in self.main_model):
            bottleneck_nchannel = self.thickness*16 // self.unet_factor
            self.rescale = lambda x: jax.image.resize(x, [x.shape[0],1,1,self.thickness * 16 // self.unet_factor // 2],method='bilinear')
            self.rescale_filters = lambda x: jax.image.resize(x, [x.shape[0],self.kernel_size,self.kernel_size,x.shape[-1]],method='bilinear')
            self.filter_enc0 = Up(bottleneck_nchannel // 2, bottleneck_nchannel // 4, self.bilinear,self.test,self.group_norm,  self.num_groups,self.activation)
            self.filter_enc1 = Up(bottleneck_nchannel // 4, bottleneck_nchannel // 8, self.bilinear,self.test,self.group_norm,  self.num_groups,self.activation)
            self.filter_enc2 = Up(bottleneck_nchannel // 8, self.n_classes, self.bilinear,self.test,self.group_norm,  self.num_groups,self.activation)
            self.filter_enc3 = Up(self.n_classes, self.n_classes, self.bilinear,self.test,self.group_norm,  self.num_groups,self.activation)
            self.filter_enc4 = DoubleConv(self.n_classes, self.n_classes,self.n_classes, self.test,self.group_norm,self.num_groups,self.activation)
    
    def unet(self,inp):
        out, skips = self.down(inp)
        bottleneck = self.bottleneck(out)
        out = self.up(bottleneck, skips)
        return out, bottleneck

    @staticmethod
    def parse_arguments(parser):
        parser.add_argument('--in_features', type=int, default=12, help='Number of input features (channels) to the UNet')
        parser.add_argument('--out_features', type=int, default=6, help='Number of output features (channels) to the UNet')
        parser.add_argument('--bilinear', type=bool, default=True, help='Should we use bilinear upsampling (True) or inverse convolution')
        parser.add_argument('--group_norm', type=bool, default=True, help='Should we use group norm or batch norm?')
        parser.add_argument('--thickness', type=int, default=128, help='Thickness of each layer')
        parser.add_argument('--num_groups', type=int, default=32, help='Num groups')
        parser.add_argument('--kernel_size', type=int, default=15, help='Num groups')
        parser.add_argument('--unet_factor', type=int, default=1, help='Num groups')
        parser.add_argument('--high_dim', action='store_true',help='Do not change the delta value')
        parser.add_argument('--outc_kernel_size', type=int, default=1,help='Do not change the delta value')

        return parser

    def __call__(self, x):
        out, bottleneck = self.unet(x)
        if('filter' in self.main_model):
            bottleneck = self.rescale(bottleneck)
            out_filters = self.filter_enc0(bottleneck,None)
            out_filters = self.filter_enc1(out_filters,None)
            out_filters = self.filter_enc2(out_filters,None)
            out_filters = self.filter_enc3(out_filters,None)
            out_filters = self.rescale_filters(out_filters)
            out_filters = self.filter_enc4(out_filters)
            if('normal' in self.main_model):
                out_filters = out_filters / (jnp.abs(out_filters).sum((-2,-3),keepdims=True) + 1e-6)

            return out, out_filters
        else:
            return out
        

class DummyConv(nn.Module):
    n_channels : int
    n_classes : int

    def setup(self):
        self.outc = OutConv(self.n_channels, self.n_classes)
        # pass

    @staticmethod
    def parse_arguments(parser):
        parser.add_argument('--in_features', type=int, default=12, help='Number of input features (channels) to the UNet')
        parser.add_argument('--out_features', type=int, default=6, help='Number of output features (channels) to the UNet')
        return parser

    def __call__(self, x):
        logits = self.outc(x)
        return logits * 1e-20
