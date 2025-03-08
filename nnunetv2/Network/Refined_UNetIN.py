import torch
import torch.nn as nn
import torch.nn.functional as F

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
      nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.InstanceNorm3d(dim_out),
      nn.ReLU(),
      nn.Conv3d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.InstanceNorm3d(dim_out),
      nn.ReLU()
    )

class UNet(nn.Module):
    def __init__(self, in_channel=1, num_class=3,channel=32):
        super(UNet, self).__init__()
        self.stem = nn.Conv3d(in_channel, channel, 1,1)
        self.encoder1 =add_conv_stage(channel, channel, kernel_size=[1,3,3],stride=[1,1,1],padding=[0,1,1])  # b, 16, 10, 10
        self.encoder2 =add_conv_stage(channel, channel*2, kernel_size=[1,3,3],stride=[1,1,1],padding=[0,1,1]) # b, 8, 3, 3
        self.encoder3 =add_conv_stage(channel*2, channel*4)
        self.encoder4 =add_conv_stage(channel*4, channel*8)
        self.encoder5 = add_conv_stage(channel*8, channel*16)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)

        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder4 = add_conv_stage(channel*16, channel*8)  # b, 8, 15, 1
        self.decoder3 = add_conv_stage(channel*8, channel*4) # b, 1, 28, 28
        self.decoder2 = add_conv_stage(channel*4, channel*2, kernel_size=[1,3,3],stride=[1,1,1],padding=[0,1,1])
        self.decoder1 = add_conv_stage(channel*2, channel, kernel_size=[1,3,3],stride=[1,1,1],padding=[0,1,1])
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