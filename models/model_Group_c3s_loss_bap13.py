import torchvision.models as models
import torch.nn as nn
import torch
import os
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../"))

'''
修改了分组的地方，原先仅使用了加权，改成使用残差结构
'''

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.GELU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class BAP(nn.Module):
    def __init__(self,  **kwargs):
        super(BAP, self).__init__()
    def forward(self,feature_maps,attention_maps):
        feature_shape = feature_maps.size() ## 12*768*26*26*
        attention_shape = attention_maps.size() ## 12*num_parts*26*26 # parts=32
        # print(feature_shape,attention_shape)
        phi_I = torch.einsum('imjk,injk->imn', (attention_maps, feature_maps)) ## 12*32*768
        phi_I = torch.div(phi_I, float(attention_shape[2] * attention_shape[3]))
        phi_I = torch.mul(torch.sign(phi_I), torch.sqrt(torch.abs(phi_I) + 1e-12))
        phi_I = phi_I.view(feature_shape[0],-1)
        raw_features = torch.nn.functional.normalize(phi_I, dim=-1) ##12*(32*768)
        pooling_features = raw_features*100 # 为啥乘100？
        # print(pooling_features.shape)
        return raw_features,pooling_features

# C3S Loss
class RegularLoss(nn.Module):

    def __init__(self, gamma=0, part_features=None, nparts=1):
        super(RegularLoss, self).__init__()
        self.nparts = nparts
        self.gamma = gamma

    def forward(self, x):
        assert isinstance(x, list), "parts features should be presented in a list"
        corr_matrix = torch.zeros(self.nparts, self.nparts) # np=3
        epsilon = 1e-8
        for i in range(self.nparts):
            x[i] = x[i].squeeze()
            x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True) + epsilon) # L2归一化

        # original design
        for i in range(self.nparts):
            for j in range(self.nparts):
                # torch.mm(x[i], x[j].t()) 计算相似度 利用余弦公式
                corr_matrix[i, j] = torch.mean(torch.mm(x[i], x[j].t()))
                if i == j:
                    corr_matrix[i, j] = 1.0 - corr_matrix[i, j]
                    
        regloss = torch.mul(torch.sum(torch.triu(corr_matrix)), self.gamma).cuda()
        # regloss = max(0.0, torch.mul(torch.sum(torch.triu(corr_matrix)), self.gamma).cuda())

        return regloss


class GroupInteractionFusion(nn.Module):
    def __init__(self, in_channels, num_groups=4):
        super(GroupInteractionFusion, self).__init__()
        self.num_groups = num_groups
        self.in_channels = in_channels
        
        # self.cbam = CBAM(in_channels)
        
        self.cbams = nn.ModuleList()
        
        # 使用 for 循环将模块重复添加
        for _ in range(num_groups):  # 重复4次
            self.cbams.append(CBAM(in_channels))

        # 特征嵌入，用于相似性计算
        self.embedding = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 聚类中心初始化（可学习参数） 不确定价值，但有用，，好像有个网络就是这么做的
        self.group_centers = nn.Parameter(torch.randn(num_groups, in_channels))

        self.adpavgpool = nn.AdaptiveAvgPool2d(1)
        # self.avg = nn.AvgPool2d(kernel_size=14, stride=1)
        self.map1 = nn.Linear(in_channels * num_groups, 512) # 其实512是有点小的，可以提升到1024
        self.drop = nn.Dropout(p=0.5)
        self.map2 = nn.Linear(512, in_channels)
        # self.fc = nn.Linear(2048, 13)
        self.sigmoid = nn.Sigmoid()
        
        self.flatten = nn.Flatten()

    def forward(self, x, flag='train'):
        b, c, h, w = x.shape
        if flag == 'train' or flag == 'val':
            # 特征嵌入
            x_embedded = self.embedding(x)  # (b, c, h, w)
            x_flatten = x_embedded.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)

            # 计算每个像素与分组中心的相似性
            group_centers = self.group_centers.unsqueeze(0).expand(b, -1, -1)  # (b, num_groups, c)
            similarity = torch.einsum('bkc,bnc->bkn', group_centers, x_flatten)  # (b, num_groups, h*w)
            similarity = F.softmax(similarity, dim=1)  # 对每个像素进行分组权重归一化

            # 根据相似性分配特征到组
            grouped_features = [] # len = self.num_groups
            for g in range(self.num_groups):
                weight = similarity[:, g, :].view(b, 1, h, w)  # (b, 1, h, w)
                grouped_features.append(x + weight * x)  # 按权重加权特征
                
            features = []
            for idx, cbam in enumerate(self.cbams):
                features.append(cbam(grouped_features[idx]))
            
            return features
        # elif flag == 'val':
        #     return x
    

class Net(nn.Module):
    def __init__(self, num_classes, num_groups = 2):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        resnet50 = models.resnet50(weights=None)
        load_path = os.path.join(root_dir, 'pretrained/resnet50.pth')
        resnet50.load_state_dict(torch.load(load_path, weights_only=True))
        feas = list(resnet50.children())[:-1]
        self.pre_layer = nn.Sequential(*feas[0:4])
        self.stage_1 = nn.Sequential(*feas[4])  # ResNet50 Stage1
        self.stage_2 = nn.Sequential(*feas[5])  # ResNet50 Stage2
        self.stage_3 = nn.Sequential(*feas[6])  # ResNet50 Stage3
        self.stage_4 = nn.Sequential(*feas[7])  # ResNet50 Stage4
        self.avg = feas[8]
        self.flatten = nn.Flatten()
        
        self.bap = BAP()
        
        self.convs = nn.ModuleList()
        for _ in range(num_groups):  # 重复4次
            self.convs.append(nn.Conv2d(2048, 512, 1))
        # 分组网络 还没用上
        # self.group_1 = GroupInteractionFusion(num_groups=4, in_channels=1024)
        self.group_2 = GroupInteractionFusion(num_groups=num_groups, in_channels=2048)
        self.fc = nn.Linear(2048 * 128, self.num_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x,flag='train'):
        x = self.pre_layer(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        # x, group1 = self.group_1(x)
        x = self.stage_4(x)
        if flag == 'train' or flag == 'val':
            group = self.group_2(x) # 13个组
            # for i in range(self.num_groups):
            #     group[i] = group[i] + x
            # 方案2使用512
            for idx, c in enumerate(self.convs):
                group[idx] = c(group[idx])
                
            result = []
            raw_features = []
            for i in range(self.num_groups):
                attention = group[i] # b 2048 14 14
                # attention = group[i][:,:32,:,:]
                bap_0 = self.bap(x,attention) # b 2024*2048 2048*32
                k = bap_0[0]
                raw_features.append(k)
                a = self.fc(self.flatten(bap_0[1]))
                result.append(a)
                # logit_all.append(bap_0)
            return raw_features, result # b 13
        # elif flag == 'val':
        #     x = self.group_2(x, flag='val')
        #     a = self.fc(self.flatten(self.avg(x)))
        #     return a
    
# from torchsummary import summary
from thop import profile
if __name__ == "__main__":
    net = Net(num_classes=13, num_groups=13).cuda()
    input_tensor = torch.randn(4, 3, 448, 448).cuda()
    x = net(input_tensor)
    print(len(x))
    
    # c2psa = C2PSA(c1=2048, c2=2048)
    # input_tensor = torch.randn(1, 2048, 14, 14)
    # output_tensor = c2psa(input_tensor)
    # print(output_tensor.shape)
    
    # FLOPs: 16.526798848 G
    # Params: 23.534669 M
    ### 看模型的参数两
    # summary(net, (3,448,448))
    # print(net.eval())
    flops, params = profile(net.cuda(), inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
    print(f"Params: {params / 1e6} M")