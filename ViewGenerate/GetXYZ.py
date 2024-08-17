import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch, gc
gc.collect()
torch.cuda.empty_cache()

#from model.ThreeElementCNN2 import view_generate
from model.ResCNN import view_generate
import open3d as o3d
import numpy as np
from dataloder.SJTU_PCQA_1280x1280 import pc_normalize
from tqdm import tqdm

# 子函数： 提供数据集名字，进行相应路径下的视点生成，输出generateView.txt
class experimentSet():
    def __init__(self,dataset_name=None):
        if dataset_name == 'SJTU-PCQA':
            # 数据集的路径详情
            self.dataset_name = 'SJTU-PCQA'
            self.root = '../data\SJTU-PCQA'
            self.type = ['Octree', 'ColorNoise', 'Downsample', 'Downsample+ColorNoise', 'Downsample+GaussNoise',
                    'GaussNoise', 'ColorNoise+GaussNoise']
            self.level = ['level1', 'level2', 'level3', 'level4', 'level5', 'level6']
            # 选择加载的网络参数
            self.save_model = '../log\SJTU_PCQA/best.pth'

        elif dataset_name == 'SJTU-PCQA(1280x1280)':
            # 数据集的路径详情
            self.dataset_name = 'SJTU-PCQA(1280x1280)'
            self.root = '../data\SJTU-PCQA(1280x1280)\dataset'
            self.type = ['OT', 'CN', 'DS', 'D+C', 'D+G', 'GGN', 'C+G']
            self.level = ['level1', 'level2', 'level3', 'level4', 'level5', 'level6']
            # 选择加载的网络参数
            self.save_model = '../log\SJTU_PCQA_1280x1280/best.pth'
        elif dataset_name == 'WPC2':
            # 数据集的路径详情
            self.dataset_name = 'WPC2'
            self.root = '../data\WPC2\dataset'
            self.type = ['DownSample', 'GaussNoise_g_0', 'GaussNoise_g_2', 'GaussNoise_g_4', 'G-PCC_t_4', 'G-PCC_t_6',
                    'G-PCC_t_8', 'V-PCC_g_1', 'V-PCC_g_2', 'V-PCC_g_3', 'G-PCC_p_1']
            self.level = ['level1', 'level2', 'level3']
            # 选择加载的网络参数
            self.save_model = '../log\WPC2/best.pth'
        elif dataset_name == 'WPC2(1280x1280)':
            # 数据集的路径详情
            self.dataset_name = 'WPC2(1280x1280)'
            self.root = '../data\WPC2(1280x1280)\dataset'
            self.type = ['DownSample', 'GaussNoise_g_0', 'GaussNoise_g_2', 'GaussNoise_g_4', 'G-PCC_t_4', 'G-PCC_t_6',
                    'G-PCC_t_8', 'V-PCC_g_1', 'V-PCC_g_2', 'V-PCC_g_3', 'G-PCC_p_1']
            self.level = ['level1', 'level2', 'level3']
            # 选择加载的网络参数
            self.save_model = '../log\WPC2(1280x1280)/best.pth'
        elif dataset_name == 'IRPC':
            # 数据集的路径详情
            self.dataset_name = 'IRPC'
            #self.root = '../data\IRPC\dataset'
            self.root = r'C:\Users\25808\Desktop\CL\data\IRPC\dataset'
            self.type = ['G-PCC', 'PCL', 'V-PCC']
            self.level = ['level1', 'level2', 'level3']
            # 选择加载的网络参数
            self.save_model = r'C:\Users\25808\Desktop\CL\log\IRPC/best.pth'
        elif dataset_name == 'IRPC(1280x1280)':
            # 数据集的路径详情
            self.dataset_name = 'IRPC(1280x1280)'
            self.root = '../data\IRPC(1280x1280)\dataset'
            self.type = ['G-PCC', 'PCL', 'V-PCC']
            self.level = ['level1', 'level2', 'level3']
            # 选择加载的网络参数
            self.save_model = '../log\IRPC(1280x1280)/best.pth'
        elif dataset_name == 'LSPCQA':
            # 数据集的路径详情
            self.dataset_name = 'LSPCQA'
            self.root = '../data\LSPCQA\dataset'
            self.type = ['noise_001', 'noise_002', 'noise_003', 'noise_004', 'noise_005']
            self.level = ['level1', 'level2', 'level3', 'level4', 'level5']
            # 选择加载的网络参数
            self.save_model = '../log\LSPCQA/best.pth'
        else:
            print('未找到对应数据集')
            sys.exit()


def getXYZ(path,net,device):
    f1 = open(path + '/' + 'generateView.txt', 'w', encoding='utf-8')

    model_path = path + '.ply'  # 模型路径
    pc = o3d.io.read_point_cloud(model_path)
    xyz = np.asarray(pc.points)
    xyz = pc_normalize(xyz)  # 归一化xyz
    pc.points = o3d.utility.Vector3dVector(xyz)
    rgb = np.asarray(pc.colors)
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.001, 40))  # 法线估计
    normal = np.asarray(pc.normals)

    xyz = torch.from_numpy(xyz).to(torch.float32).unsqueeze(0)
    rgb = torch.from_numpy(rgb).to(torch.float32).unsqueeze(0)
    normal = torch.from_numpy(normal).to(torch.float32).unsqueeze(0)

    for i in range(6):
        if (i == 0):
            view_center = [0, 0, -100]
        elif (i == 1):
            view_center = [0, 0, 100]
        elif (i == 2):
            view_center = [100, 0, 0]
        elif (i == 3):
            view_center = [-100, 0, 0]
        elif (i == 4):
            view_center = [0, 100, 0]
        else:
            view_center = [0, -100, 0]
        view_center = torch.tensor(np.asarray(view_center)).type(torch.float32)
        xyz = xyz.to(device)
        rgb = rgb.to(device)
        normal = normal.to(device)
        view_center = view_center.to(device)
        out = net(xyz,rgb,normal,view_center)
        generate_xyz = out[:,0,:].cpu().detach().numpy()
        while (-100 < generate_xyz[:,0]< 100) and (-100 < generate_xyz[:,1]< 100) and (-100 < generate_xyz[:,2]< 100):
            generate_xyz = generate_xyz*10
        f1.write(str(i + 1) + ':' + str(generate_xyz[0,0]) + ',' + str(generate_xyz[0,1])+ ',' + str(generate_xyz[0,2]) + '\n')
        # print(str(i + 1) + ':' + str(generate_xyz[0,0]) + ',' + str(generate_xyz[0,1])+ ',' + str(generate_xyz[0,2]) + '\n')
        del out
        torch.cuda.empty_cache()
    pass


def main(details):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = view_generate().to(device)
    net.load_state_dict(torch.load(details.save_model))
    net.eval()

    root = details.root
    model = os.listdir(root)
    type = details.type
    level = details.level

    # 获取对应的生成视点坐标
    pbar = tqdm(total=len(model)* len(type) * len(level))
    for a in range(len(model)):
        for b in range(len(type)):
            for c in range(len(level)):
                pbar.update()
                if details.dataset_name == 'IRPC' or details.dataset_name == 'IRPC(1280x1280)':
                    if (model[a] != 'model2' and model[a] != 'model3' and model[a] != 'model4') or type[b] != 'G-PCC':
                        path = root + '/' + model[a] + '/' + type[b] + '/' + level[c]
                        getXYZ(path, net,device)
                        torch.cuda.empty_cache()
                else:
                    path = root + '/' + model[a] + '/' + type[b] + '/' + level[c]
                    if os.path.exists(path):
                        getXYZ(path, net,device)
                        torch.cuda.empty_cache()
    pass


if __name__ == "__main__":
    details = experimentSet('SJTU-PCQA')
    #details = experimentSet('SJTU-PCQA(1280x1280)')
    #details = experimentSet('WPC2')
    #details = experimentSet('WPC2(1280x1280)')
    #details = experimentSet('IRPC')
    #details = experimentSet('IRPC(1280x1280)')
    #details = experimentSet('LSPCQA')
    main(details)


