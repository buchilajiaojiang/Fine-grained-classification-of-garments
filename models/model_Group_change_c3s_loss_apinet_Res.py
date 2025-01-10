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

# C3S Loss
class RegularLoss(nn.Module):

    def __init__(self, gamma=0, part_features=None, nparts=1):
        """
        :param bs: batch size
        :param ncrops: number of crops used at constructing dataset
        """
        super(RegularLoss, self).__init__()
        # self.register_buffer('part_features', part_features)
        self.nparts = nparts
        self.gamma = gamma

    def forward(self, x):
        assert isinstance(x, list), "parts features should be presented in a list"
        corr_matrix = torch.zeros(self.nparts, self.nparts) # np=3
        # 特征归一化
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

        return regloss


class GroupInteractionFusion(nn.Module):
    def __init__(self, in_channels, num_groups=4):
        """
        自适应分组 + 组间交互融合 + 分组卷积
        :param in_channels: 输入通道数
        :param num_groups: 自适应分组的数量
        :param group_conv_groups: 分组卷积的组数
        """
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
        self.map1 = nn.Linear(in_channels * num_groups, 512)
        self.drop = nn.Dropout(p=0.5)
        self.map2 = nn.Linear(512, in_channels)
        self.fc = nn.Linear(2048, 13)
        self.sigmoid = nn.Sigmoid()
        
        self.flatten = nn.Flatten()

        # 分组卷积模块
        self.group_conv = nn.Sequential( # 不懂这个地方还用分组卷积的意义
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=num_groups, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

    def forward(self, x, flag='train'):
        b, c, h, w = x.shape
        if flag == 'train':
            # 特征嵌入
            x_embedded = self.embedding(x)  # (b, c, h, w)
            x_flatten = x_embedded.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)

            # 计算每个像素与分组中心的相似性
            group_centers = self.group_centers.unsqueeze(0).expand(b, -1, -1)  # (b, num_groups, c)
            similarity = torch.einsum('bkc,bnc->bkn', group_centers, x_flatten)  # (b, num_groups, h*w)
            similarity = F.softmax(similarity, dim=1)  # 对每个像素进行分组权重归一化

            # 根据相似性分配特征到组
            grouped_features = []
            for g in range(self.num_groups):
                weight = similarity[:, g, :].view(b, 1, h, w)  # (b, 1, h, w)
                grouped_features.append(x + weight * x)  # 按权重加权特征
                
            features1 = self.adpavgpool(grouped_features[0]).squeeze() # 转成1*1后，展平
            features2 = self.adpavgpool(grouped_features[1]).squeeze()

            # 先别合并
            # 合并组特征，用于交互融合
            # x_groups = torch.cat(grouped_features, dim=1)  # (b, c*num_groups, h, w)
            
            ### new
            mutual_features = torch.cat([features1, features2], dim=1)
            map2_out = self.map1(mutual_features) # 降维
            map2_out = self.drop(map2_out)
            map2_out = self.map2(map2_out) # 升维

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)
            
            features1_self = torch.mul(gate1, features1) + features1 # batch inchannel2048
            features1_other = torch.mul(gate2, features1) + features1

            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2
            
            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))
            # features1和features2已弃用，但为了保持之前的代码格式就没有修改s
            return features1, features2, [logit1_self,logit1_other,logit2_self,logit2_other]
        elif flag == 'val':
            a = self.fc(self.flatten(self.adpavgpool(x)))
            return a
        
        
        # print("x_groups", x_groups.shape, grouped_features[0].shape)
        # 组间交互融合
        # x_interacted = self.interaction(x_groups)  # (b, c, h, w)

        # 分组卷积处理
        # x_out = self.group_conv(x_interacted)
        
        # for i in range(self.num_groups):
        #     grouped_features[i] = self.adpavgpool(grouped_features[i])
    

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
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
        self.group_2 = GroupInteractionFusion(num_groups=2, in_channels=2048)
        self.linear = nn.Linear(2048, self.num_classes)

    def forward(self, x,flag='train'):
        x = self.pre_layer(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        # x, group1 = self.group_1(x)
        x = self.stage_4(x)
        if flag == 'train':
            x1, x2, group2 = self.group_2(x) # 在api中x弃用，但先不删除
            # x1, x2现在是分组特征图
            logit1_self,logit1_other,logit2_self,logit2_other = group2
            # print(x.shape)
            # x = self.avg(x)
            # x = self.flatten(x)
            # x = self.linear(x)
            return x1, x2, logit1_self,logit1_other,logit2_self,logit2_other
        elif flag == 'val':
            return self.group_2(x, flag='val')
    
# from torchsummary import summary
from thop import profile
if __name__ == "__main__":
    net = Net(num_classes=13).cuda()
    input_tensor = torch.randn(2, 3, 448, 448).cuda()
    x, _,_,_,_,_ = net(input_tensor)
    print(x.shape)
    
    # c2psa = C2PSA(c1=2048, c2=2048)
    # input_tensor = torch.randn(1, 2048, 14, 14)
    # output_tensor = c2psa(input_tensor)
    # print(output_tensor.shape)
    
    # backbone: FLOPs: 16.526798848 G
    # backbone: Params: 23.534669 M
    
    # 当前模型：FLOPs: 34.705022976 G
    # Params: 30.877261 M
    ### 看模型的参数两
    # summary(net, (3,448,448))
    # print(net.eval())
    flops, params = profile(net.cuda(), inputs=(input_tensor,))
    print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
    print(f"Params: {params / 1e6} M")