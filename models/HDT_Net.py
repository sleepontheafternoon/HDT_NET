from models.HyperDNet import HyperDNet
# from models.HyperDNet_New import HyperDNet_New
from models.Transformer import TransformerModel
from models.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
import torch
import torch.nn as nn

# from HyperDNet import HyperDNet
# from PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
# from Transformer import TransformerModel




class TransHDT(nn.Module):
    """
    args:
    patch_dim:切片的维度，大小通常是一致的
    num_channels:
    embedding_dim:q,k,v向量长度
    num_heads:多头注意力的头数
    num_layers:transformer的层数
    hidden_dim:transformer中的隐藏层的维度
    """
    def __init__(self,img_dim,
                 patch_dim,
                 # num_channels,
                 img_channels,
                 embedding_dim,
                 num_heads,
                 num_layers,
                 hidden_dim,
                 dropout_rate = 0.0,
                 attn_dropout_rate = 0.0,
                 positional_encoding_type = "learned"):

        super(TransHDT, self).__init__()

        # 先进行判定，查看是否能被切片
        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0


        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        # self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        # 把三个维度都进行切割得到相应的patch，这些patch的数量就是序列长度 128//8 16**3 = 4096
        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        # self.flatten_dim = 128 * num_channels

        #self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        # 就是transformer中没有解码器的架构
        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)


        self.conv_x = nn.Conv3d(
            img_channels,
            self.embedding_dim,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn = nn.BatchNorm3d(img_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self,x):

        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_x(x)

        # 先用contiguous 之后才能用view来变化形状
        x = x.permute(0,2,3,4,1).contiguous() # number  modalities  H W C ---> number H W C modalities
        x = x.view(x.size(0), -1, self.embedding_dim)  # numberi 4096 512
        # 上述这两个操作是用了pytorch的view函数性质，
        # pytorch的view先是按列进行连接改变的，这样改变形状的话，能够保证属于同一个部分的数据能够被放在一起

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)

        x = self.pre_head_ln(x)

        x = self._reshape_output(x)

        return x

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x




class HDT_Net(nn.Module):

    def __init__(self,img_dim = 128,
                 patch_dim = 8,
                 img_channels = 128,
                 embedding_dim = 512,
                 num_heads = 8,
                 num_layers = 6,
                 hidden_dim = 4096,
                 drop_rate = 0.1,
                 attn_dropout_rate = 0.1,
                 positional_encoding_type = "learned"):

        super(HDT_Net, self).__init__()
        self.local_fusion = HyperDNet()

        # 此处依旧是两个path 对多模态图像信息进行融合
        self.global_fusion1 = TransHDT(img_dim,patch_dim,img_channels,
                                       embedding_dim,num_heads,
                                       num_layers,hidden_dim,
                                       drop_rate,
                                       attn_dropout_rate,
                                       positional_encoding_type)

        # self.global_fusion2 = TransHDT(img_dim,patch_dim,img_channels,
        #                                embedding_dim,num_heads,
        #                                num_layers,hidden_dim,
        #                                drop_rate,
        #                                attn_dropout_rate,
        #                                positional_encoding_type)
        self.max_pool = nn.MaxPool3d(16,1)
        self.avg_pool = nn.AvgPool3d(16,1)
        self.Hidder_layer_1 = nn.Linear(1024, 512)
        self.Hidder_layer_2 = nn.Linear(512, 32)
        # self.Hidder_layer_1 = nn.Linear(512, 256)
        # self.Hidder_layer_2 = nn.Linear(256, 32)
        self.drop_layer = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(32, 2)

        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal(m.weight,mode="fan_out",
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        # top_x,bottom_x = self.local_fusion(x)
        #
        # # n 512 16 16
        # top_x_t, bottom_x_t = self.global_fusion1(top_x),self.global_fusion2(bottom_x)
        #
        # feature = torch.cat([top_x_t,bottom_x_t],dim=1)
        feature = self.local_fusion(x)
        feature = self.global_fusion1(feature)
        # n 512 16 16 16
        feature1 = self.avg_pool(feature)
        feature2 = self.max_pool(feature)
        y = torch.cat([feature1,feature2],dim=1)
        y = y.view(feature.size()[0],-1)
        y = self.drop_layer(y)
        y = self.Hidder_layer_1(y)
        y = self.Hidder_layer_2(y)
        y = self.classifier(y)

        return y


if __name__ == "__main__":
    t = torch.randn(4,4,128,128,128)
    model = HDT_Net()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))



