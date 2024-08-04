# 网络结构： 见visio结构图

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np
import open3d as o3d
import math
esp = 1e-8


# 视点生成网络 - 特征提取
class VGNet_Feature(nn.Module):
    def __init__(self):
        super(VGNet_Feature, self).__init__()
        self.xyz_conv11 = torch.nn.Conv1d(3, 16, 1)
        self.xyz_conv12 = torch.nn.Conv1d(16, 32, 1)
        self.xyz_conv13 = torch.nn.Conv1d(32, 64, 1)
        #self.xyz_mlp = torch.nn.Linear(227,224)
        self.xyz_mlp = torch.nn.Conv1d(115, 224,1)

        self.rgb_conv11 = torch.nn.Conv1d(3, 16, 1)
        self.rgb_conv12 = torch.nn.Conv1d(16, 32, 1)
        self.rgb_conv13 = torch.nn.Conv1d(32, 64, 1)
        #elf.rgb_mlp = torch.nn.Linear(227,224)
        self.rgb_mlp = torch.nn.Conv1d(115, 224,1)

    def forward(self,x,r,normal,view_center):                          # x = xyz, r = rgb, normal = normals
        B,N,C = x.size()
        view_center = view_center.unsqueeze(0)
        view_center = view_center.unsqueeze(2)

        x = x.permute(0, 2, 1)
        x1 = F.leaky_relu(self.xyz_conv11(x))                   # Conv1d (3,16)
        x2 = F.leaky_relu(self.xyz_conv12(x1))                  # Conv1d(16,32)
        x3 = F.leaky_relu(self.xyz_conv13(x2))                  # Conv1d(32,64)
        pf1 = torch.cat((x1,x2,x3),dim = 1)                     # PointFeature

        # gf1 = torch.max(pf1, 2, keepdim=True)[0]                # GlobalFeature
        # gf1 = gf1.repeat(1,1,N)
        # concat1 = torch.cat((pf1,gf1),dim = 1)
        concat1 = pf1


        r = r.permute(0, 2, 1)
        r1 = F.leaky_relu(self.rgb_conv11(r))                   # Conv1d(3,16)
        r2 = F.leaky_relu(self.rgb_conv12(r1))                  # Conv1d(16,32)
        r3 = F.leaky_relu(self.rgb_conv13(r2))                  # Conv1d(32,64)
        rgb_pf1 = torch.cat((r1,r2,r3), dim=1)                  # PointFeature

        # rgb_gf1 = torch.max(rgb_pf1, 2, keepdim=True)[0]        # GlobalFeature
        # rgb_gf1 = rgb_gf1.repeat(1, 1, N)
        # r_concat1 = torch.cat((rgb_pf1, rgb_gf1), dim=1)
        r_concat1= rgb_pf1

        view_center = view_center.repeat(1,1,N)
        # concat1 = torch.cat((concat1,view_center),dim = 1).permute(0,2,1)
        # r_concat1 = torch.cat((r_concat1,view_center), dim=1).permute(0,2,1)
        concat1 = torch.cat((concat1,view_center),dim = 1)
        r_concat1 = torch.cat((r_concat1,view_center), dim=1)
        concat1 = self.xyz_mlp(concat1).permute(0,2,1)
        concat1 = torch.max(concat1, 1, keepdim=True)[0]        # GlobalFeature
        r_concat1 = self.rgb_mlp(r_concat1).permute(0,2,1)
        r_concat1 = torch.max(r_concat1, 1, keepdim=True)[0]        # GlobalFeature

        return concat1,r_concat1


# 视点生成网络 - 注意力模块
class VGNet_SelfAttention(nn.Module):
    def __init__(self, hidden_size=None, num_attention_heads=1, dropout_prob=0.2):
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(VGNet_SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads  # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变

        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size)  # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [B, N, C]  假设C=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [B, N, 1, C]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [B, 1, N, C]

    def forward(self, input):

        # 线性变换
        mixed_query_layer = self.query(input)  # [B, N, C]
        mixed_key_layer = self.key(input)  # [B, N, C]
        mixed_value_layer = self.value(input)  # [B, N, C]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [B, 1, N, C]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [B, 1, N, C]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [B, 1, N, C]*[B, 8, 16, N]  ==> [B, 1, N, N]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [B, 1, N, N]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1


        # 将注意力转化为概率分布，即注意力权重
        attention_proB = nn.Softmax(dim=-1)(attention_scores)  # [B, 1, N, N]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_proB = self.dropout(attention_proB)


        context_layer = torch.matmul(attention_proB, value_layer)  # [B, 1, N, C]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [B, N, 1, C]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [B, N, C]
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer  # [B, N, C ] 得到输出


# 视点生成网络 - 视点生成
class VGNet_XYZ(nn.Module):
    def __init__(self):
        super(VGNet_XYZ, self).__init__()
        # self.mlp31 = torch.nn.Linear(448, 256)
        #self.mlp32 = torch.nn.Linear(256, 128)
        #self.mlp33 = torch.nn.Linear(128, 64)
        #self.mlp34 = torch.nn.Linear(64, 3)
        self.mlp31 = torch.nn.Linear(448, 256)
        self.mlp32 = torch.nn.Linear(256,3)

    def forward(self,gf2):
        # x = F.leaky_relu(self.mlp31(gf2))
        # x = F.leaky_relu(self.mlp32(x))
        # x = F.leaky_relu(self.mlp33(x))
        # x = self.mlp34(x)
        x = F.leaky_relu(self.mlp31(gf2))
        x = self.mlp32(x)
        return x


# 网络串联
class view_generate(nn.Module):
    def __init__(self):
        super(view_generate, self).__init__()
        self.feature    = VGNet_Feature()
        self.attention1 = VGNet_SelfAttention(224)
        self.attention2 = VGNet_SelfAttention(224)
        self.view_out = VGNet_XYZ()

    def forward(self,xyz,rgb,normal,view_cener):
        feature1,feature2 = self.feature(xyz,rgb,normal,view_cener)

        feature1 = self.attention1(feature1)
        feature2 = self.attention2(feature2)
        feature = torch.cat((feature1,feature2),dim = 2)

        out = self.view_out(feature)

        return out






if __name__ == '__main__':
    net = view_generate()
    input = o3d.io.read_point_cloud("E:\CL\Code\dataset\PCQA_dataset\model1\ColorNoise\level1.ply")
    xyz = np.asarray(input.points)
    color = np.asarray(input.colors)
    normal = np.asarray(input.normals)
    xyz = torch.from_numpy(xyz).to(torch.float32)
    rgb = torch.from_numpy(color).to(torch.float32)
    normal = torch.from_numpy(normal).to(torch.float32)

    xyz  = torch.unsqueeze(xyz,dim=0)
    color  = torch.unsqueeze(rgb,dim=0)
    with torch.no_grad():
        out = net(xyz,color,xyz,0)

    print(666)




