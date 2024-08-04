# 网络结构： 见visio结构图

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np
import open3d as o3d
esp = 1e-8


# 视点生成网络 - 特征提取
class VGNet_Feature(nn.Module):
    def __init__(self):
        super(VGNet_Feature, self).__init__()
        self.xyz_conv11 = torch.nn.Conv1d(3, 16, 1)
        self.xyz_conv12 = torch.nn.Conv1d(16, 32, 1)
        self.xyz_conv13 = torch.nn.Conv1d(32, 64, 1)
        self.xyz_conv21 = torch.nn.Conv1d(336, 64, 1)
        self.xyz_conv22 = torch.nn.Conv1d(64, 32, 1)
        self.xyz_conv23 = torch.nn.Conv1d(32, 16, 1)

        self.rgb_conv11 = torch.nn.Conv1d(3, 16, 1)
        self.rgb_conv12 = torch.nn.Conv1d(16, 32, 1)
        self.rgb_conv13 = torch.nn.Conv1d(32, 64, 1)
        self.rgb_conv21 = torch.nn.Conv1d(336, 64, 1)
        self.rgb_conv22 = torch.nn.Conv1d(64, 32, 1)
        self.rgb_conv23 = torch.nn.Conv1d(32, 16, 1)

        self.normal_conv11 = torch.nn.Conv1d(3, 16, 1)
        self.normal_conv12 = torch.nn.Conv1d(16, 32, 1)
        self.normal_conv13 = torch.nn.Conv1d(32, 64, 1)


    def forward(self,x,r,normal):                          # x = xyz, r = rgb, normal = normals
        B,N,C = x.size()

        normal = normal.permute(0, 2, 1)
        normal1 = F.leaky_relu(self.normal_conv11(normal))            # Conv1d(3,16)
        normal2 = F.leaky_relu(self.normal_conv12(normal1))           # Conv1d(16,32)
        normal3 = F.leaky_relu(self.normal_conv13(normal2))           # Conv1d(32,64)
        normal_pf1 = torch.cat((normal1,normal2,normal3), dim=1)      # PointFeature



        x = x.permute(0, 2, 1)
        x1 = F.leaky_relu(self.xyz_conv11(x))                   # Conv1d (3,16)
        x2 = F.leaky_relu(self.xyz_conv12(x1))                  # Conv1d(16,32)
        x3 = F.leaky_relu(self.xyz_conv13(x2))                  # Conv1d(32,64)
        pf1 = torch.cat((x1,x2,x3),dim = 1)                     # PointFeature

        gf1 = torch.max(pf1, 2, keepdim=True)[0]                # GlobalFeature
        gf1 = gf1.repeat(1,1,N)
        concat1 = torch.cat((pf1,gf1),dim = 1)
        concat1 = torch.cat((concat1,normal_pf1),dim = 1)       # FirstConcat Feature  1x336xN (BxCxN)

        x11 = F.leaky_relu(self.xyz_conv21(concat1))            # Conv1d(336,64)
        x12 = F.leaky_relu(self.xyz_conv22(x11))                # Conv1d(64,32)
        x13 = F.leaky_relu(self.xyz_conv23(x12))                # Conv1d(32,16)
        pf2 = torch.cat((x11,x12,x13),dim = 1)                  # PointFeature
        gf2 = torch.max(pf2, 2, keepdim=True)[0]                # GlobalFeature 1x112x1



        r = r.permute(0, 2, 1)
        r1 = F.leaky_relu(self.rgb_conv11(r))                   # Conv1d(3,16)
        r2 = F.leaky_relu(self.rgb_conv12(r1))                  # Conv1d(16,32)
        r3 = F.leaky_relu(self.rgb_conv13(r2))                  # Conv1d(32,64)
        rgb_pf1 = torch.cat((r1,r2,r3), dim=1)                  # PointFeature

        rgb_gf1 = torch.max(rgb_pf1, 2, keepdim=True)[0]        # GlobalFeature
        rgb_gf1 = rgb_gf1.repeat(1, 1, N)
        r_concat1 = torch.cat((rgb_pf1, rgb_gf1), dim=1)
        r_concat1 = torch.cat((r_concat1, normal_pf1), dim=1)   # FirstConcat Feature  1x336xN (BxCxN)

        r11 = F.leaky_relu(self.rgb_conv21(r_concat1))          # Conv1d(336,64)
        r12 = F.leaky_relu(self.rgb_conv22(r11))                # Conv1d(64,32)
        r13 = F.leaky_relu(self.rgb_conv23(r12))                # Conv1d(32,16)
        rgb_pf2 = torch.cat((r11,r12,r13), dim=1)               # PointFeature
        rgb_gf2 = torch.max(rgb_pf2, 2, keepdim=True)[0]        # GlobalFeature 1x112x1

        out = torch.concat((gf2,rgb_gf2),dim = 1)
        out = out.permute(0, 2, 1)
        return out


# 视点生成网络 - 视点生成
class VGNet_XYZ(nn.Module):
    def __init__(self):
        super(VGNet_XYZ, self).__init__()
        self.mlp31 = torch.nn.Linear(224, 256)
        self.mlp32 = torch.nn.Linear(256, 128)
        self.mlp33 = torch.nn.Linear(128, 64)
        self.mlp34 = torch.nn.Linear(64, 3)

    def forward(self,gf2):
        x = F.leaky_relu(self.mlp31(gf2))
        x = F.leaky_relu(self.mlp32(x))
        x = F.leaky_relu(self.mlp33(x))
        x = self.mlp34(x)
        return x


# 网络串联
class view_generate(nn.Module):
    def __init__(self):
        super(view_generate, self).__init__()
        self.feature    = VGNet_Feature()
        self.view_up    = VGNet_XYZ()
        self.view_down  = VGNet_XYZ()
        self.view_left  = VGNet_XYZ()
        self.view_right = VGNet_XYZ()
        self.view_front = VGNet_XYZ()
        self.view_back  = VGNet_XYZ()

    def forward(self,xyz,rgb,normal):
        feature = self.feature(xyz,rgb,normal)
        up      = self.view_up(feature)
        down    = self.view_down(feature)
        left    = self.view_left(feature)
        right   = self.view_right(feature)
        front   = self.view_front(feature)
        back    = self.view_back(feature)
        out = torch.cat((up,down,left,right,front,back),dim = 1)

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

    out = net(xyz,color,xyz)

    print(666)




