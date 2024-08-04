import os
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import Dataset


#生成 路径文档
def data_txt_generate3(input_dir_path, out_train_txt, out_test_txt, _ratio_for_train):
    model = os.listdir(input_dir_path)
    size = len(model)   # 模型总数
    type = ['OT','CN','DS','D+C','D+G','GGN','C+G']
    level = ['level1','level2','level3','level4','level5','level6']
    txt_train = open(out_train_txt, mode="w", encoding="utf-8")
    txt_test = open(out_test_txt, mode="w", encoding="utf-8")
    # 生成训练文档
    for a in range(int(size*_ratio_for_train)):
         for b in range(len(type)):
             for c in range(len(level)):
                 distored_view = input_dir_path + '/' + model[a] + '/' + type[b] + '/' + level[c]
                 distored_type = type[b]
                 txt_train.write('%s %s\n' % (distored_view,distored_type))


    # 生成测试文档
    for a in range(int(size*_ratio_for_train),len(model)):
         for b in range(len(type)):
             for c in range(len(level)):
                 distored_view = input_dir_path + '/' + model[a] + '/' + type[b] + '/' + level[c]
                 distored_type = type[b]
                 txt_test.write('%s %s\n' % (distored_view,distored_type))


# 模型获取子函数：根据txt_train文档生成2个list字符串，分别是模型路径，失真类型
def read_root_txt(root_txt_path):
    assert root_txt_path.endswith('.txt')     #判断文件某个条件是否成立，false引发异常
    path_list,type_list =[],[]
    try:
        with open(root_txt_path, "r", encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
            for line in lines:
                path,distortion= line.strip().split(' ')
                path_list.append(path)
                type_list.append(distortion)
    except UnicodeDecodeError:
        with open(root_txt_path, "r", encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
            for line in lines:
                path,distortion= line.strip().split(' ')
                path_list.append(path)
                type_list.append(distortion)
    return path_list,type_list


# 标签获取子函数：根据txt_train文档生成一个list：worstview
def selectview_txt(view_txt_path):
    assert view_txt_path.endswith('.txt')     #判断文件某个条件是否成立，false引发异常
    viewselect =[]
    try:
        with open(view_txt_path, "r", encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
            for line in lines:
                view = line.strip().split(' ')
                viewselect.append(view)

    except UnicodeDecodeError:
        with open(view_txt_path, "r", encoding='utf-8') as f_txt:
            lines = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
            for line in lines:
                view = line.strip().split(' ')
                viewselect.append(view)
    return viewselect


# 点云尺寸归一化子函数
def pc_normalize(pc_xyz):
    centroid = np.mean(pc_xyz,axis=0)
    pc_xyz = pc_xyz - centroid
    m = np.max(np.sqrt(np.sum(pc_xyz**2,axis=1)))
    pc_xyz = pc_xyz/m
    return pc_xyz


# 由 center, plane, index 得到 xyz 子函数
def get_xyz(plane,location):
    '''
    说明：输入点云中心点坐标，观察平面序号，平面视点位置序号
    平面序号：1/2：xoy上/下；  3/4:yoz左/右；  5/6:zox前/后
    视点序号：1：正交； 2：正方体顶点，不重复
    '''
    center = [0,0,0,0]
    temp = 100
    if plane == 1:
        if location == 1:
            eye = [center[1], center[2], center[3] - temp];
        elif location == 2:
            eye = [center[1] + 60, center[2] - 60, center[3] - temp];
        elif location == 3:
            eye = [center[1], center[2] - 60, center[3] - temp];
        elif location == 4:
            eye = [center[1] - 60, center[2] - 60, center[3] - temp];
        elif location == 5:
            eye = [center[1] - 60, center[2], center[3] - temp];
        elif location == 6:
            eye = [center[1] - 60, center[2] + 60, center[3] - temp];
        elif location == 7:
            eye = [center[1], center[2] + 60, center[3] - temp];
        elif location == 8:
            eye = [center[1] + 60, center[2] + 60, center[3] - temp];
        elif location == 9:
            eye = [center[1] + 60, center[2], center[3] - temp];
        pass

    elif plane == 2:
        if location == 1:
            eye = [center[1], center[2], center[3] + temp];
        elif location == 2:
            eye = [center[1]+60 , center[2]-60, center[3]+temp];
        elif location == 3:
            eye = [center[1] , center[2]-60, center[3]+temp];
        elif location == 4:
            eye = [center[1]-60 , center[2]-60, center[3]+temp];
        elif location == 5:
            eye = [center[1] - 60, center[2], center[3] + temp];
        elif location == 6:
            eye = [center[1]-60 , center[2]+60, center[3]+temp];
        elif location == 7:
            eye = [center[1], center[2] + 60, center[3] + temp];
        elif location == 8:
            eye = [center[1]+60 , center[2]+60, center[3]+temp];
        elif location == 9:
            eye = [center[1]+60 , center[2], center[3]+temp];
        pass

    elif plane == 3:
        if location == 1:
            eye = [center[1] + temp, center[2], center[3]];
        elif location == 2:
            eye = [center[1] + temp, center[2] - 60, center[3] + 60];
        elif location == 3:
            eye = [center[1]+temp , center[2], center[3]+60];
        elif location == 4:
            eye = [center[1]+temp , center[2]+60, center[3]+60];
        elif location == 5:
            eye = [center[1]+temp , center[2]+60, center[3]];
        elif location == 6:
            eye = [center[1] + temp, center[2] + 60, center[3] - 60];
        elif location == 7:
            eye = [center[1] + temp, center[2], center[3] - 60];
        elif location == 8:
            eye = [center[1] + temp, center[2] - 60, center[3] - 60];
        elif location == 9:
            eye = [center[1]+temp , center[2]-60, center[3]];
        pass

    elif plane == 4:
        if location == 1:
            eye = [center[1] - temp, center[2], center[3]];
        elif location == 2:
            eye = [center[1]-temp , center[2]-60, center[3]+60];
        elif location == 3:
            eye = [center[1]-temp , center[2], center[3]+60];
        elif location == 4:
            eye = [center[1] - temp, center[2] + 60, center[3]];
        elif location == 5:
            eye = [center[1]-temp , center[2]+60, center[3]];
        elif location == 6:
            eye = [center[1] - temp, center[2] + 60, center[3] - 60];
        elif location == 7:
            eye = [center[1] - temp, center[2], center[3] - 60];
        elif location == 8:
            eye = [center[1]-temp , center[2]-60, center[3]-60];
        elif location == 9:
            eye = [center[1]-temp , center[2]-60, center[3]];
        pass

    elif plane == 5:
        if location == 1:
            eye = [center[1], center[2] + temp, center[3]];
        elif location == 2:
            eye = [center[1] + 60, center[2] + temp, center[3] + 60];
        elif location == 3:
            eye = [center[1] , center[2]+temp, center[3]+60];
        elif location == 4:
            eye = [center[1]-60 , center[2]+temp, center[3]+60];
        elif location == 5:
            eye = [center[1] - 60, center[2] + temp, center[3]];
        elif location == 6:
            eye = [center[1]-60 , center[2]+temp, center[3]-60];
        elif location == 7:
            eye = [center[1] , center[2]+temp, center[3]-60];
        elif location == 8:
            eye = [center[1]+60 , center[2]+temp, center[3]-60];
        elif location == 9:
            eye = [center[1] + 60, center[2] + temp, center[3]];
        pass

    elif plane == 6:
        if location == 1:
            eye = [center[1] , center[2]-temp, center[3] ];
        elif location == 2:
            eye = [center[1] + 60, center[2] - temp, center[3] + 60];
        elif location == 3:
            eye = [center[1] , center[2]-temp, center[3]+60];
        elif location == 4:
            eye = [center[1]-60 , center[2]-temp, center[3]+60];
        elif location == 5:
            eye = [center[1]-60 , center[2]-temp, center[3]];
        elif location == 6:
            eye = [center[1] - 60, center[2] - temp, center[3] - 60];
        elif location == 7:
            eye = [center[1], center[2] - temp, center[3] - 60];
        elif location == 8:
            eye = [center[1] + 60, center[2] - temp, center[3] - 60];
        elif location == 9:
            eye = [center[1] + 60, center[2] - temp, center[3]];
        pass

    return np.asarray(eye)
    pass


# 视点序号list转换为视点坐标list 子函数
def viewchange(view_list):
    xyz = []
    for i in range(len(view_list)):
        view = view_list[i][0]
        plane = int(view[0])
        index = int(view[2])
        sub_xyz = get_xyz(plane,index)
        xyz.append(sub_xyz)
    return xyz


# 数据加载子函数； #root_txt_path: txt_train 文档名；
class VGDataset(Dataset):
    def __init__(self, root_txt_path):
        super(VGDataset, self).__init__()
        assert isinstance(root_txt_path, str) and root_txt_path.endswith('.txt')
        self.path_list, self.distortion_type= read_root_txt(root_txt_path)


    def __getitem__(self, index):
        '''
        :param index: 样本序号
        :return:
        '''
        path = self.path_list[index]

        model_path = path+'.ply'                                                             # 模型路径
        pc = o3d.io.read_point_cloud(model_path)
        xyz = np.asarray(pc.points)
        xyz = pc_normalize(xyz)                                                             # 归一化xyz
        pc.points = o3d.utility.Vector3dVector(xyz)
        color = np.asarray(pc.colors)
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.001, 40))   # 法线估计
        normal = np.asarray(pc.normals)
        input = []
        input.append(torch.from_numpy(xyz).to(torch.float32))
        input.append(torch.from_numpy(color).to(torch.float32))
        input.append(torch.from_numpy(normal).to(torch.float32))

        label_path = path + '/' + 'worstview.txt'                                           # 最佳视点list的路径
        view_list = selectview_txt(label_path)
        label = viewchange(view_list)                                                       # 得到6个推荐视点相对点云中心的坐标值
        label = torch.tensor(np.asarray(label))

        return input,label

    def __len__(self):
        return len(self.path_list)




