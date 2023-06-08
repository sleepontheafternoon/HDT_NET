import torch
import torch.nn as nn
#
# class SE(nn.Module):
#     def __init__(self,channel):
#         super(SE, self).__init__()
#         self.bn1 = nn.BatchNorm3d(channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.linear = nn.Linear(channel,channel)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self,x):
#         x = self.bn1(x)
#         x = self.relu(x)
#         b,c,_,_,_ = x.size()
#         y = self.avg_pool(x).view(b,c)
#         y = self.linear(y)
#         y = self.sigmoid(y).view(b,c,1,1,1)
#         return x*y
#
# t = torch.randn(4,4,128,128,128)
# fn = SE(4)
# print(fn(t).size())
checkpoint = torch.load("/home/cyx/Codes/HDT-Net/checkpoint/My_Net_focalHDT_Net2023-02-25/model_epoch_best.pth")
for k,v in checkpoint['idh_state_dict'].items():
    print(k,v.shape)