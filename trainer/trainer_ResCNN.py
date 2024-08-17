import os

import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import torch, gc
gc.collect()
torch.cuda.empty_cache()
from torch.utils.data import DataLoader
from model.ResCNN import view_generate
#from model.ThreeElementCNN2 import view_generate
import torch.optim as optim
import torch.nn as nn


# 子函数： 提供数据集名字，进行实验配置
class experimentSet():
    def __init__(self,dataset_name=None):
        if dataset_name == 'SJTU-PCQA':
            # 用于生成模型路径txt
            self.dataset_name = 'SJTU-PCQA'
            self.model_path = '../data\SJTU-PCQA'
            self.out_train_txt = '../log\SJTU_PCQA/train_path.txt'
            self.out_test_txt = '../log\SJTU_PCQA/test_path.txt'
            # 用于保存训练过程中的文件路径
            self.outpath = '../log\SJTU_PCQA/'
        elif dataset_name == 'SJTU-PCQA(1280x1280)':
            # 用于生成模型路径txt
            self.dataset_name = 'SJTU-PCQA(1280x1280)'
            self.model_path = '../data\SJTU-PCQA(1280x1280)\dataset'
            self.out_train_txt = '../log\SJTU_PCQA_1280x1280/train_path.txt'
            self.out_test_txt = '../log\SJTU_PCQA_1280x1280/test_path.txt'
            # 用于保存训练过程中的文件路径
            self.outpath = '../log\SJTU_PCQA_1280x1280/'
        elif dataset_name == 'WPC2':
            # 用于生成模型路径txt
            self.dataset_name = 'WPC2'
            self.model_path = '../data\WPC2\dataset'
            self.out_train_txt = 'E:\CL\ViewPoint_Generate\log\WPC2/train_path.txt'
            self.out_test_txt = 'E:\CL\ViewPoint_Generate\log\WPC2/test_path.txt'
            # 用于保存训练过程中的文件路径
            self.outpath = 'E:\CL\ViewPoint_Generate\log\WPC2/'
        elif dataset_name == 'WPC2(1280x1280)':
            # 用于生成模型路径txt
            self.dataset_name = 'WPC2(1280x1280)'
            self.model_path = 'E:\CL\ViewPoint_Generate\data\WPC2(1280x1280)\dataset'
            self.out_train_txt = 'E:\CL\ViewPoint_Generate\log\WPC2(1280x1280)/train_path.txt'
            self.out_test_txt = 'E:\CL\ViewPoint_Generate\log\WPC2(1280x1280)/test_path.txt'
            # 用于保存训练过程中的文件路径
            self.outpath = 'E:\CL\ViewPoint_Generate\log\WPC2(1280x1280)/'
        elif dataset_name == 'IRPC':
            # 用于生成模型路径txt
            self.dataset_name = 'IRPC'
            self.model_path = 'E:\CL\ViewPoint_Generate\data\IRPC\dataset'
            self.out_train_txt = 'E:\CL\ViewPoint_Generate\log\IRPC/train_path.txt'
            self.out_test_txt = 'E:\CL\ViewPoint_Generate\log\IRPC/test_path.txt'
            # 用于保存训练过程中的文件路径
            self.outpath = 'E:\CL\ViewPoint_Generate\log\IRPC/'
        elif dataset_name == 'IRPC(1280x1280)':
            # 用于生成模型路径txt
            self.dataset_name = 'IRPC(1280x1280)'
            self.model_path = 'E:\CL\ViewPoint_Generate\data\IRPC(1280x1280)\dataset'
            self.out_train_txt = 'E:\CL\ViewPoint_Generate\log\IRPC(1280x1280)/train_path.txt'
            self.out_test_txt = 'E:\CL\ViewPoint_Generate\log\IRPC(1280x1280)/test_path.txt'
            # 用于保存训练过程中的文件路径
            self.outpath = 'E:\CL\ViewPoint_Generate\log\IRPC(1280x1280)/'
        elif dataset_name == 'LSPCQA':
            # 用于生成模型路径txt
            self.dataset_name = 'LSPCQA'
            self.model_path = r'..\data\LSPCQA\dataset'
            self.out_train_txt = r'..\log\LSPCQA/train_path.txt'
            self.out_test_txt = r'..\log\LSPCQA/test_path.txt'
            # 用于保存训练过程中的文件路径
            self.outpath = r'E:\CL\ViewPoint_Generate\log\LSPCQA/'
        else:
            print('未找到对应数据集')
            sys.exit()


# 定义向量的角度损失 (CNN1:6个坐标输出)
class VectorLoss(nn.Module):
    def __init__(self):
        super(VectorLoss,self).__init__()

    def forward(self,vector1,vector2):
        loss = 0
        for i in range(6):
            v1 = vector1[:,i,:].squeeze(0)
            v2 = vector2[:,i,:].squeeze(0)
            loss += 1 - torch.cosine_similarity(v1,v2,dim=0,eps=1e-8)
        return loss


# 定义向量的角度损失 (1个坐标输出)
class VectorLoss2(nn.Module):
    def __init__(self):
        super(VectorLoss2,self).__init__()

    def forward(self,vector1,vector2):
        loss = 0
        v1 = vector1.squeeze(0)
        v1 = v1.squeeze(0)
        v2 = vector2.squeeze(0)
        v2 = v2.squeeze(0)
        loss += 1 - torch.cosine_similarity(v1,v2,dim=0,eps=1e-8)
        return loss


class ResCNN_Trainer():
    # 测试
    def test(self,test_loader, net, criterion, test_loss=None,device=None):
        loss = 0
        sum = 0
        loss_for_caculate_standard_deviation = []
        net.eval()
        with torch.no_grad():
            for i, (data, label) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader)):
                for abc in range(6):
                    if (abc == 0):
                        view_center = [0,0,-100]
                    elif (abc==1):
                        view_center = [0,0,100]
                    elif (abc==2):
                        view_center = [100,0,0]
                    elif (abc==3):
                        view_center = [-100,0,0]
                    elif (abc==4):
                        view_center = [0,100,0]
                    else:
                        view_center = [0,-100,0]
                    xyz = data[0]
                    rgb = data[1]
                    normal = data[2]
                    xyz = xyz.to(device).type(torch.half)
                    rgb = rgb.to(device).type(torch.half)
                    normal = normal.to(device).type(torch.half)
                    view_center = torch.tensor(np.asarray(view_center)).to(device).type(torch.half)
                    sublabel = label[:, abc, :].to(device)
                    sublabel = sublabel.unsqueeze(1)
                    out = net(xyz, rgb, normal, view_center)

                    loss_net = criterion(out, sublabel)
                    loss += loss_net.cpu().detach().item()
                    loss_for_caculate_standard_deviation.append(loss)
                    sum += 1
                    del xyz, rgb, normal, out, loss_net
                    torch.cuda.empty_cache()
                del data,label
                torch.cuda.empty_cache()
        print("Average:testloss: " + str(loss / sum) + '\n')
        std = np.std(np.array(loss_for_caculate_standard_deviation))
        if test_loss is not None:
            # test_loss.write(str(loss/sum)+ '\n')
            test_loss.write('AverageLoss: ' + str(loss / sum) + '   std: ' + str(std) + '\n')
        return loss / sum

    # 训练过程中的参数设置
    def train_and_test(self,details):
        if (details.dataset_name == 'SJTU-PCQA'):
            from dataloder.SJTU_PCQA import data_txt_generate3,VGDataset
            data_txt_generate3(details.model_path,details.out_train_txt,details.out_test_txt,0.8)
            train_Dataset = VGDataset(details.out_train_txt)
            test_Dataset = VGDataset(details.out_test_txt)

        elif (details.dataset_name == 'SJTU-PCQA(1280x1280)'):
            from dataloder.SJTU_PCQA_1280x1280 import data_txt_generate3,VGDataset
            data_txt_generate3(details.model_path,details.out_train_txt,details.out_test_txt,0.8)
            train_Dataset = VGDataset(details.out_train_txt)
            test_Dataset = VGDataset(details.out_test_txt)

        elif (details.dataset_name == 'WPC2'):
            from dataloder.WPC2 import data_txt_generate3,VGDataset
            data_txt_generate3(details.model_path,details.out_train_txt,details.out_test_txt,0.8)
            train_Dataset = VGDataset(details.out_train_txt)
            test_Dataset = VGDataset(details.out_test_txt)

        elif (details.dataset_name == 'WPC2(1280x1280)'):
            from dataloder.WPC2_1280x1280 import data_txt_generate3,VGDataset
            data_txt_generate3(details.model_path,details.out_train_txt,details.out_test_txt,0.8)
            train_Dataset = VGDataset(details.out_train_txt)
            test_Dataset = VGDataset(details.out_test_txt)
        elif (details.dataset_name == 'IRPC'):
            from dataloder.IRPC import data_txt_generate3,VGDataset
            data_txt_generate3(details.model_path,details.out_train_txt,details.out_test_txt,0.9)
            train_Dataset = VGDataset(details.out_train_txt)
            test_Dataset = VGDataset(details.out_test_txt)
        elif (details.dataset_name == 'IRPC(1280x1280)'):
            from dataloder.IRPC_1280x1280 import data_txt_generate3,VGDataset
            data_txt_generate3(details.model_path,details.out_train_txt,details.out_test_txt,0.9)
            train_Dataset = VGDataset(details.out_train_txt)
            test_Dataset = VGDataset(details.out_test_txt)
        elif (details.dataset_name == 'LSPCQA'):
            from dataloder.LSPCQA import data_txt_generate3, VGDataset
            data_txt_generate3(details.model_path, details.out_train_txt, details.out_test_txt, 0.8)
            train_Dataset = VGDataset(details.out_train_txt)
            test_Dataset = VGDataset(details.out_test_txt)
        else:
            sys.exit()

        trainloader = DataLoader(train_Dataset, batch_size=1, num_workers=1, shuffle=True, drop_last=False)
        testloader = DataLoader(test_Dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)


        net = view_generate()  # 调用视点生成网络的结构
        #net.load_state_dict(torch.load("E:\CL\ViewPoint_Generate\log\SJTU_PCQA\\best.pth"))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device).half()


        criterion = VectorLoss2()
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0)
        StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

        epochs = 30
        best_loss = 10
        best_epoch = 0

        train_loss = open(details.outpath+'train_loss.txt', 'a', encoding='utf-8')
        test_loss = open(details.outpath+'test_loss.txt', 'a', encoding='utf-8')

        for epoch in range(1, epochs + 1):
            net.train()
            print("\nepoch:" + str(epoch))
            loss = 0
            for idx, (data, label) in tqdm.tqdm(enumerate(trainloader),total=len(trainloader)):

                for i in range(6):
                    if (i == 0):
                        view_center = [0,0,-100]
                    elif (i==1):
                        view_center = [0,0,100]
                    elif (i==2):
                        view_center = [100,0,0]
                    elif (i==3):
                        view_center = [-100,0,0]
                    elif (i==4):
                        view_center = [0,100,0]
                    else:
                        view_center = [0,-100,0]
                    xyz = data[0]
                    rgb = data[1]
                    normal = data[2]
                    xyz = xyz.to(device).type(torch.half)
                    rgb = rgb.to(device).type(torch.half)
                    normal = normal.to(device).type(torch.half)
                    view_center = torch.tensor(np.asarray(view_center)).to(device).type(torch.half)


                    sublabel = label[:,i,:].to(device)
                    sublabel = sublabel.unsqueeze(1)
                    out = net(xyz, rgb, normal,view_center)

                    loss_net = criterion(out, sublabel)
                    loss += loss_net.cpu().detach().item()
                    optimizer.zero_grad()
                    loss_net.backward()
                    optimizer.step()
                    del xyz, rgb, normal, out, loss_net
                    torch.cuda.empty_cache()
                if idx % 10 == 9:
                    print("loss:" + str(loss / (10 * 6)))
                    if train_loss is not None:
                        train_loss.write(str(loss / (10 * 6)) + "\n")
                    loss = 0
                del data,label
                torch.cuda.empty_cache()
            StepLR.step()
            print("Testing" + '\n')
            loss = self.test(testloader, net, criterion, test_loss, device)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(net.state_dict(), details.outpath+'best.pth')
        print("bestepoch：" + str(best_epoch) + '\n')
        print("bestLoss：" + str(best_loss) + '\n')
        print("finish training")
        train_loss.close()



if __name__ == "__main__":
    # details = experimentSet('SJTU-PCQA')
    #details = experimentSet('SJTU-PCQA(1280x1280)')
    #details = experimentSet('WPC2')
    # details = experimentSet('WPC2(1280x1280)')
    #details = experimentSet('IRPC')
    # details = experimentSet('IRPC(1280x1280)')
    details = experimentSet('LSPCQA')
    ResCNN_Trainer().train_and_test(details)

