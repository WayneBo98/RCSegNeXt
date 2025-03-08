import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class ConvNeStBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                kernel_size=3,
                stride=1,
                padding=1,
                exp_r:int=4,
                scale = 4
                ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        width = int(in_channels/scale)
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        convs = []
        for i in range(self.nums):
          convs.append(nn.Conv3d(width, width, kernel_size=kernel_size, stride = stride,padding = padding,groups = width))
        self.convs = nn.ModuleList(convs)
        self.scale = scale
        self.width  = width
        self.exp_r = exp_r

        # Normalization Layer. GroupNorm is used by default.
        self.norm = nn.InstanceNorm3d(in_channels)

        self.conv2 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            groups=scale
        )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = nn.Conv3d(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            groups=width
        )
        
        self.conv_shortcut = nn.Conv3d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = stride,
            padding = 0
        )
        self.exp_r = exp_r
 
    def forward(self, x, dummy_tensor=None):
        x1 = x
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1:
          out = torch.cat((out, spx[self.nums]),1)
        x1 = self.act(self.conv2(self.norm(out)))
        x1 = x1.reshape(x1.shape[0], self.scale, self.exp_r, -1, x1.shape[2], x1.shape[3], x1.shape[4]).reshape(x1.shape[0], self.scale*self.exp_r, -1, x1.shape[2], x1.shape[3], x1.shape[4]).permute(0,2,1,3,4,5).reshape(x1.shape[0], -1, x1.shape[2], x1.shape[3], x1.shape[4])
        x1 = self.conv3(x1)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        x1 = x + x1  
        return x1

class ConvNeXtBlock(nn.Module):

    def __init__(self, 
                in_channels:int, 
                out_channels:int, 
                kernel_size=3,
                stride=1,
                padding=1,
                exp_r:int=4
                ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            groups = in_channels
        )

        # Normalization Layer. GroupNorm is used by default.
        self.norm = nn.InstanceNorm3d(in_channels)

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = nn.Conv3d(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        self.conv_shortcut = nn.Conv3d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = stride,
            padding = 0
        )

 
    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        x1 = x + x1  
        return x1

def add_conv_stage_next(dim_in, dim_out, kernel_size=3, stride=1, padding=1,exp_r=2):
    return nn.Sequential(
      ConvNeXtBlock(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,exp_r=exp_r),
      ConvNeXtBlock(dim_out, dim_out, kernel_size=kernel_size, stride=1, padding=padding,exp_r=exp_r),
    )

def add_conv_stage_nest(dim_in, dim_out, kernel_size=3, stride=1, padding=1,exp_r=2, scale = 4):
    return nn.Sequential(
      ConvNeStBlock(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,exp_r=exp_r, scale = scale),
      ConvNeStBlock(dim_out, dim_out, kernel_size=kernel_size, stride=1, padding=padding,exp_r=exp_r, scale = scale),
    )

class UNet(nn.Module):
    def __init__(self, in_channel=1, num_class=3,channel=32,scale=4):
        super(UNet, self).__init__()
        self.stem = nn.Conv3d(in_channel, channel, 1,1)
        self.encoder1 =add_conv_stage_next(channel, channel, kernel_size=[1,3,3],stride=[1,1,1],padding=[0,1,1])  # b, 16, 10, 10
        self.encoder2 =add_conv_stage_next(channel, channel*2, kernel_size=[1,3,3],stride=[1,1,1],padding=[0,1,1]) # b, 8, 3, 3
        self.encoder3 =add_conv_stage_nest(channel*2, channel*4, scale = scale)
        self.encoder4 =add_conv_stage_nest(channel*4, channel*8, scale = scale)
        self.encoder5 = add_conv_stage_nest(channel*8, channel*16, scale = scale)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)

        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder4 = add_conv_stage_nest(channel*16, channel*8,scale=scale)  # b, 8, 15, 1
        self.decoder3 = add_conv_stage_nest(channel*8, channel*4,scale=scale) # b, 1, 28, 28
        self.decoder2 = add_conv_stage_next(channel*4, channel*2, kernel_size=[1,3,3],stride=[1,1,1],padding=[0,1,1])
        self.decoder1 = add_conv_stage_next(channel*2, channel, kernel_size=[1,3,3],stride=[1,1,1],padding=[0,1,1])
        # self.decoder5 = add_conv_stage(64, 32, 3)

        self.tran1=nn.ConvTranspose3d(channel*16,channel*8, kernel_size=[2,2,2], stride=[2,2,2])
        self.tran2=nn.ConvTranspose3d(channel*8, channel*4, kernel_size=[2,2,2], stride=[2,2,2])
        self.tran3=nn.ConvTranspose3d(channel*4, channel*2, kernel_size=[1,2,2], stride=[1,2,2])
        self.tran4=nn.ConvTranspose3d(channel*2, channel, kernel_size=[1,2,2], stride=[1,2,2])

        self.outconv =  nn.Sequential(
            nn.Conv3d(channel, num_class, 1, 1),
            # nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
        )
        print("******************RefinedUnet is initialized******************")

    def forward(self, x):
        # print("input shape",x.shape)
        x = self.stem(x)
        conv1_out = self.encoder1(x)
        conv2_out = self.encoder2(F.max_pool3d(conv1_out, kernel_size=[1,2,2], stride=[1,2,2]))
        conv3_out = self.encoder3(F.max_pool3d(conv2_out, kernel_size=[1,2,2], stride=[1,2,2]))
        conv4_out = self.encoder4(F.max_pool3d(conv3_out,2,2))
        conv5_out = self.encoder5(F.max_pool3d(conv4_out,2,2))

        conv5m_out_ = torch.cat((self.tran1(conv5_out), conv4_out), 1)
        conv4m_out = self.decoder4(conv5m_out_)

        conv4m_out_ = torch.cat((self.tran2(conv4m_out), conv3_out), 1)
        conv3m_out = self.decoder3(conv4m_out_)

        conv3m_out_ = torch.cat((self.tran3(conv3m_out), conv2_out), 1)
        conv2m_out = self.decoder2(conv3m_out_)

        conv2m_out_ = torch.cat((self.tran4(conv2m_out), conv1_out), 1)
        conv1m_out = self.decoder1(conv2m_out_)

        outconv_out = self.outconv(conv1m_out)

        return outconv_out