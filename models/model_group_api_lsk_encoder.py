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

# 先cat再卷积，保留了原先的实现结构
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # Mlp(2048,1024, 13)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        ## 已经包含了FFN
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous() # 拆会了特征图的格式
    
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]



class GroupInteractionFusion(nn.Module):
    def __init__(self, in_channels, num_groups=4):
        super(GroupInteractionFusion, self).__init__()
        self.num_groups = num_groups
        self.in_channels = in_channels

        # 特征嵌入，用于相似性计算
        self.embedding = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # 聚类中心初始化（可学习参数） 不确定价值，但有用，，好像有个网络就是这么做的
        self.group_centers = nn.Parameter(torch.randn(num_groups, in_channels))

        # 分组特征交互模块
        self.interaction = nn.Sequential(
            nn.Conv2d(in_channels * num_groups, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.adpavgpool = nn.AdaptiveAvgPool2d(1)
        # self.avg = nn.AvgPool2d(kernel_size=14, stride=1)
        self.map1 = nn.Conv2d(in_channels * (num_groups + 1), in_channels, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(p=0.5)
        # self.map2 = nn.Linear(512, in_channels)
        self.map2 = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)
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
            grouped_features.append(x)
            for g in range(self.num_groups):
                weight = similarity[:, g, :].view(b, 1, h, w)  # (b, 1, h, w)
                grouped_features.append(weight * x)  # 按权重加权特征
                
            features = torch.stack(grouped_features, dim=0) # groups_num b inchannel 14*14
            features = features.permute(1, 0, 2, 3,4) # b groups inchannel fw*fh

            mutual_features = features.reshape(features.size(0), -1, features.size(3), features.size(4))
            mutual_features = self.adpavgpool(mutual_features) # b c 1 1
            map2_out = self.map1(mutual_features) # 降维
            map2_out = self.map2(map2_out) # 升维

            gates = [] # 4 * b * inchannel fw fh
            for i in range(self.num_groups + 1):
                gate1 = torch.mul(map2_out, self.adpavgpool(grouped_features[i])) # 4 2048 14 14 mul相当于直接*
                gates.append(self.sigmoid(gate1))
                
            featuers_all = [] # 多组特征 分组了 group group batch inchannel fw fh
            for i in range(self.num_groups + 1):
                a = torch.mul(gates[i], x) + x
                featuers_all.append(a)
            
            
            return grouped_features, featuers_all
        
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
      
class SABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv3 = nn.Conv2d(dim, dim//2, 1)
        self.conv4 = nn.Conv2d(dim, dim//2, 1)
        self.conv5 = nn.Conv2d(dim, dim//2, 1)
        
        self.conv_squeeze = nn.Conv2d(2, 5, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, group, x):
        # x shape: group4 b inchannel fw fh   
        # attn1 = self.conv0(x) # 第一个特征图
        # attn2 = self.conv_spatial(attn1) # 在第一个特征图基础上再生成一个特征图

        attn1 = group[0] # b inchannel fw, fh
        attn2 = group[1]
        attn3 = group[2]
        attn4 = group[3]
        attn5 = group[4]
        
        attn1 = self.conv1(attn1) # 1*1卷积恒等变化
        attn2 = self.conv1(attn2)
        attn3 = self.conv1(attn3)
        attn4 = self.conv1(attn4)
        attn5 = self.conv1(attn5)
        
        attn = torch.cat([attn1, attn2, attn3, attn4, attn5], dim=1) # 联结
        ## SA start
        avg_attn = torch.mean(attn, dim=1, keepdim=True) # mean是平均函数，代表了Avg操作
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) # max最大值操作
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid() # 做卷积后 使用sigmod进行激活
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1) + attn3 * sig[:,2,:,:].unsqueeze(1) + attn4 * sig[:,3,:,:].unsqueeze(1) + attn5 * sig[:,4,:,:].unsqueeze(1)# 多张特征图融合
        # attn shape: b inchannel fw fh
        ## SA end
        attn = self.conv(attn) # 生成 S
        return x * attn # 生成 Y # default x * attn
  

class SA(nn.Module):
    def __init__(self, in_channels, num_groups=4):
        super(SA, self).__init__()
        self.num_groups = num_groups
        self.in_channels = in_channels
        
        self.sa = SABlock(in_channels)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self,group, x):
        # g b, c, h, w = x.shape
        shorcut = x.clone()
        result = self.sa(group, x)
        result = self.proj_2(result)
        result = result + shorcut ### 先试下x0看看效果，要是不好再改成原x。按理来说用x[0]
        return result
            

    

class Net(nn.Module):
    def __init__(self, num_classes, num_groups = 2, hidden_c = 256):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.num_groups = num_groups
        self.hidden_c = hidden_c
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
        
        # 分组网络 还没用上
        # self.group_1 = GroupInteractionFusion(num_groups=4, in_channels=1024)
        self.group_2 = GroupInteractionFusion(num_groups=num_groups, in_channels=2048)
        
        self.bap = BAP()
        self.sa = SA(in_channels=2048,num_groups=num_groups)
            
        self.proj =  nn.Sequential(
                    nn.Conv2d(2048, hidden_c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_c)
        )
        
        self.short = Conv(2048, hidden_c, k=1, act=False)
        
        self.aifi = AIFI(hidden_c,1024)
        # self.aifi2 = AIFI(hidden_c,1024)
        
        self.fc = nn.Linear(256, self.num_classes) # 需要修改的值
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x,flag='train'):
        x = self.pre_layer(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        # x, group1 = self.group_1(x)
        x = self.stage_4(x)
        if flag == 'train' or flag == 'val':
            b, c, h, w = x.shape
            grouped_features, group = self.group_2(x) # 在api中x弃用，但先不删除
            # res = self.flatten(self.avg(x))
            feature = self.sa(group, x) # b 2048 14 14
            
            proj = self.proj(feature)

            # sa_f = self.proj(sa_f) + self.short(x) 
            p0 = self.aifi(proj) 
            
            p0 = self.flatten(self.avg(p0))
            
            p0 = self.fc(self.drop(p0))
            
                
            return  p0 # b 13
    
# from torchsummary import summary
from thop import profile
if __name__ == "__main__":
    net = Net(num_classes=13, num_groups=4).cuda()
    input_tensor = torch.randn(1, 3, 448, 448).cuda()
    x = net(input_tensor)
    print(x)
    
    # c2psa = C2PSA(c1=2048, c2=2048)
    # input_tensor = torch.randn(1, 2048, 14, 14)
    # output_tensor = c2psa(input_tensor)
    # print(output_tensor.shape)
    
    # FLOPs: 16.526798848 G
    # Params: 23.534669 M
    ### 看模型的参数两
    # FLOPs: 131.89311744 G
    # Params: 82.285181 M
    flops, params = profile(net.cuda(), inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
    print(f"Params: {params / 1e6} M")