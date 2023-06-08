import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DepthWiseConv(nn.Module):

    def __init__(self, in_channel, out_channel):

        super(DepthWiseConv, self).__init__()

        # 逐通道卷积 groups控制分组卷积
        self.depth_conv = nn.Conv3d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv3d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class Final_conv(nn.Module):
    def __init__(self,in_channels,out_channels,reduction=16):
        super(Final_conv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels,in_channels // reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels // reduction, out_channels),
            nn.Sigmoid()
        )
    def forward(self,x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class ResNeXtBottleneck(nn.Module):


    def __init__(self, in_planes, planes, cardinality=32, stride=1,expansion = 2,
                 downsample=None):
        super().__init__()
        self.expansion = expansion
        mid_planes = cardinality * planes // 32
        self.conv1 = conv1x1x1(in_planes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes,
                               mid_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)
        self.downsample = nn.Sequential(
                    conv1x1x1(in_planes, planes *self.expansion, stride),
                    nn.BatchNorm3d(planes * self.expansion))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):

        residual = x
        residual = self.downsample(residual)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)

        return x + residual




class _HyperDLayer0(nn.Sequential):

    # 更改为深度可分离卷积
    def __init__(self, num_input_features,output_size, drop_rate = 0.1):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            DepthWiseConv(num_input_features,output_size))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return new_features


class _HyperDLayer(nn.Module):


    def __init__(self, num_input_features,output_size, drop_rate = 0.1):
        super().__init__()
        self.features = ResNeXtBottleneck(num_input_features,output_size)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.features(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return new_features


# 对图片最开始输入时，作处理
class Initial_Conv(nn.Module):

    def __init__(self,
                 n_input_channels=2,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 num_init_features=64,
                 reduction=16,
                 ):
        super().__init__()

        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=conv1_t_size,
                                    stride=conv1_t_stride,
                                    padding=conv1_t_stride//2,
                                    bias=False)),
                         ('norm1', nn.BatchNorm3d(num_init_features)),
                         ('relu1', nn.ReLU(inplace=True))]

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_init_features,num_init_features//reduction),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_init_features//reduction,num_init_features),
            nn.Sigmoid()
        )


        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))

        self.features = nn.Sequential(OrderedDict(self.features))

    def forward(self,x):
        x = self.features(x)
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x



class HyperDNet_New(nn.Module):

    def __init__(self,
                 n_input_channels=2,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 num_init_features=64,
                 base_out=32,
                 drop_rate = 0
                 ):
        super(HyperDNet_New, self).__init__()
        # Initial Process
        self.top1 = Initial_Conv(n_input_channels,
                 conv1_t_size,
                 conv1_t_stride,
                 no_max_pool,
                 num_init_features
                 )

        self.top2 = _HyperDLayer(num_init_features*2,base_out)
        self.top3 = _HyperDLayer(num_init_features*2,base_out)
        self.top4 = _HyperDLayer(num_init_features*2,base_out)
        # self.top5 = _HyperDLayer(num_init_features*6,base_out*2)
        # self.top6 = _HyperDLayer(num_init_features*8,base_out*2)  #2
        # self.top7 = _HyperDLayer(num_init_features * 11, base_out * 3)
        # self.top8 = _HyperDLayer(num_init_features * 14, base_out * 4)



        self.bottom1 = Initial_Conv(n_input_channels,
                                    conv1_t_size,
                                    conv1_t_stride,
                                    no_max_pool,
                                    num_init_features)

        self.bottom2 = _HyperDLayer(num_init_features*2,base_out)
        self.bottom3 = _HyperDLayer(num_init_features*2,base_out)
        self.bottom4 = _HyperDLayer(num_init_features*2,base_out)

        self.final_conv = Final_conv(num_init_features*2,base_out*4)

        # self.bottom5 = _HyperDLayer(num_init_features*6, base_out*2)
        # self.bottom6 = _HyperDLayer(num_init_features*8, base_out*2)
        # self.bottom7 = _HyperDLayer(num_init_features * 11, base_out * 3)
        # self.bottom8 = _HyperDLayer(num_init_features * 14, base_out * 4)


        # 网络参数初始化设置
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight,
        #                                 mode='fan_out',
        #                                 nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)



    def forward(self,x):

        # initial process
        y_t = self.top1(x[:,(0,3),:,:,:])  # flair t2

        y_b = self.bottom1(x[:,1:3,:,:,:])  # t1ce t1

        # First cat     [2,64,64,32,32]
        y1_t_i = torch.cat([y_t,y_b],dim=1)
        y1_b_i = torch.cat([y_b,y_t],dim=1)

        # 2 32 64 32 32
        y1_t_o = self.top2(y1_t_i)
        y1_b_o = self.bottom2(y1_b_i)

        # second cat  因为首次cat一开始进行处理的特征图不能使用，到了第二次才是利用了不同path的结果
        # 2 192 64 32 32
        y2_t_i = torch.cat([y1_t_o,y1_b_o],dim=1)
        y2_b_i = torch.cat([y1_b_o,y1_t_o],dim=1)

        # 2 32 64 32 32
        y2_t_o = self.top3(y2_t_i)
        y2_b_o = self.bottom3(y2_b_i)

        # third cat   256
        y3_t_i = torch.cat([y2_t_o, y2_b_o],dim=1)
        y3_b_i = torch.cat([y2_b_o, y2_t_o],dim=1)
        # 64
        y3_t_o = self.top4(y3_t_i)
        y3_b_o = self.bottom4(y3_b_i)

        y4 = torch.cat([y3_t_o,y3_b_o],dim=1)
        y = self.final_conv(y4)

        # fourth cat 384
        # y4_t_i = torch.cat([y3_t_i, y3_t_o, y3_b_o],dim=1)
        # y4_b_i = torch.cat([y3_b_i, y3_b_o, y3_t_o],dim=1)

        # 64
        # y4_t_o = self.top5(y4_t_i)
        # y4_b_o = self.bottom5(y4_b_i)
        #
        # # fifth cat 512
        # y5_t_i = torch.cat([y4_t_i, y4_t_o, y4_b_o],dim=1)
        # y5_b_i = torch.cat([y4_b_i, y4_b_o, y4_t_o],dim=1)
        #
        # # 96
        # y5_t_o = self.top6(y5_t_i)
        # y5_b_o = self.bottom6(y5_b_i)

        # y6_t_i = torch.cat([y5_t_i, y5_t_o, y5_b_o], dim=1)
        # y6_b_i = torch.cat([y5_b_i, y5_b_o, y5_t_o], dim=1)
        #
        # # 96
        # y6_t_o = self.top7(y6_t_i)
        # y6_b_o = self.bottom7(y6_b_i)
        #
        # y7_t_i = torch.cat([y6_t_i, y6_t_o, y6_b_o], dim=1)
        # y7_b_i = torch.cat([y6_b_i, y6_b_o, y6_t_o], dim=1)
        #
        # # 96
        # y7_t_o = self.top8(y7_t_i)
        # y7_b_o = self.bottom8(y7_b_i)

        return y

if __name__ == "__main__":
    model = HyperDNet_New()
    # model = ResNeXtBottleneck(64,32)
    t = torch.randn((2, 4, 128, 128, 128))
    print(model(t).size())















