import torch
import torch.nn as nn
import torch.nn.functional as F

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
      nn.Conv3d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm3d(dim_out),
      nn.ReLU(),
      nn.Conv3d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm3d(dim_out),
      nn.ReLU()
    )
def conv_trans_block_3d(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        nn.ReLU())

class UNet(nn.Module):
    def __init__(self, in_channel=1, num_class=3,channel=32):
        super(UNet, self).__init__()
        self.encoder1 =add_conv_stage(in_channel, channel, 3)  # b, 16, 10, 10
        self.encoder2 =add_conv_stage(channel, channel*2, 3) # b, 8, 3, 3
        self.encoder3 =add_conv_stage(channel*2, channel*4, 3)
        self.encoder4 =add_conv_stage(channel*4, channel*8, 3)
        self.encoder5 = add_conv_stage(channel*8, channel*16, 3)
        # self.encoder5=   nn.Conv3d(256, 512, 3, stride=1, padding=1)

        # self.decoder1 = nn.Conv3d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder4 = add_conv_stage(channel*16, channel*8, 3)  # b, 8, 15, 1
        self.decoder3 = add_conv_stage(channel*8, channel*4, 3) # b, 1, 28, 28
        self.decoder2 = add_conv_stage(channel*4, channel*2, 3)
        self.decoder1 = add_conv_stage(channel*2, channel, 3)
        # self.decoder5 = add_conv_stage(64, 32, 3)

        self.tran1=conv_trans_block_3d(channel*16,channel*8)
        self.tran2=conv_trans_block_3d(channel*8,channel*4)
        self.tran3=conv_trans_block_3d(channel*4,channel*2)
        self.tran4=conv_trans_block_3d(channel*2,channel)

        self.outconv =  nn.Sequential(
            nn.Conv3d(channel, num_class, 1, 1),
            # nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
        )
        print("******************MyUnet is initialized******************")

    def forward(self, x):
        # print(x.shape)
        # print(x.squeeze(0).shape)
        # x, pad_left, pad_right, pad_top, pad_bottom=pad_tensor(x.squeeze(0))
        # x=x.unsqueeze(0)
        conv1_out = self.encoder1(x)
        conv2_out = self.encoder2(F.max_pool3d(conv1_out,2,2))
        conv3_out = self.encoder3(F.max_pool3d(conv2_out,2,2))
        conv4_out = self.encoder4(F.max_pool3d(conv3_out,2,2))
        conv5_out = self.encoder5(F.max_pool3d(conv4_out,2,2))

        # print(self.tran1(conv5_out).shape)
        # print(conv4_out.shape)
        conv5m_out_ = torch.cat((self.tran1(conv5_out), conv4_out), 1)
        conv4m_out = self.decoder4(conv5m_out_)

        conv4m_out_ = torch.cat((self.tran2(conv4m_out), conv3_out), 1)
        conv3m_out = self.decoder3(conv4m_out_)

        conv3m_out_ = torch.cat((self.tran3(conv3m_out), conv2_out), 1)
        conv2m_out = self.decoder2(conv3m_out_)

        conv2m_out_ = torch.cat((self.tran4(conv2m_out), conv1_out), 1)
        conv1m_out = self.decoder1(conv2m_out_)

        outconv_out = self.outconv(conv1m_out)
        # print(outconv_out.shape)

        # outconv_out=pad_tensor_back(outconv_out.squeeze(0), pad_left, pad_right, pad_top, pad_bottom).unsqueeze(0)

        return outconv_out