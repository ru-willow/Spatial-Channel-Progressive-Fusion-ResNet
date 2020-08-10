import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import torch

class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResBlk,self).__init__()

        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()

        if ch_out!=ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = self.extra(x)+out
        out = F.relu(out)

        return out

class ABlock(nn.Module):
    
    def __init__(self, planes = 32):
        super(ABlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes // 2, 1, 1, 0)      
        self.conv2 = nn.Conv2d(planes // 2, planes // 2, 1, 1, 0) 
        self.conv3 = nn.Conv2d(planes // 2, planes // 2, 3, 1, 1) 
        self.conv4 = nn.Conv2d(planes // 2, planes // 2, 5, 1, 2)
        self.sigmoid = nn.Sigmoid()
        self.conv5 = nn.Conv2d(1, planes // 4, 3, 1, 1)
        self.conv6 = nn.Conv2d(planes // 4, 1, 3, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.conv7 = nn.Conv2d(planes, planes, 3, 1, 1)


    def forward(self, x):
        F0 = self.conv1(x)
        F1 = self.conv2(F0)
        F2 = self.conv3(F0)
        F3 = self.conv4(F0)

        a0 = torch.mean(F0, dim=1,keepdim = True)
        a1 = torch.mean(F1, dim=1,keepdim = True)
        a2 = torch.mean(F2, dim=1,keepdim = True)
        a3 = torch.mean(F3, dim=1,keepdim = True)

        a0 = self.sigmoid(self.conv6(F.relu(self.conv5(a0))))
        a1 = self.sigmoid(self.conv6(F.relu(self.conv5(a1))))
        a2 = self.sigmoid(self.conv6(F.relu(self.conv5(a2))))
        a3 = self.sigmoid(self.conv6(F.relu(self.conv5(a3))))

        a0_int = self.global_pool(a0) 
        a1_int = self.global_pool(a1) 
        a2_int = self.global_pool(a2) 
        a3_int = self.global_pool(a3)
      
        a_int = torch.cat([a0_int, a1_int, a2_int, a3_int], 1)  
        a_sort = torch.sort(a_int)  
        F_0 = torch.mul(F0, a0)
        F_1 = torch.mul(F1, a1)
        F_2 = torch.mul(F2, a2)
        F_3 = torch.mul(F3, a3)
        S1 = torch.zeros(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2])
        S2 = torch.zeros(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2])
        # 得到B模块的输入
        for i in range(a_sort[1].shape[0]):
            if i == 0:
                S1 = (locals()['F_' + str(a_sort[1][i][3].item())][i]).view(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2])
                S2 = (locals()['F_' + str(a_sort[1][i][2].item())][i]).view(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2])
            else:
                S1 = torch.cat([ locals()['F_' + str(a_sort[1][i][3].item())][i].view(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2]), S1 ], 0)
                S2 = torch.cat([ locals()['F_' + str(a_sort[1][i][2].item())][i].view(1, F_0[0].shape[0], F_0[0].shape[1], F_0[0].shape[2]), S2 ], 0)

        S = torch.cat([S1,S1], 1) # 通道拼接
        S = self.conv7(S)

        return F.relu(S)

class BBlock(nn.Module):

    def __init__(self, out_channel=32):
        super(BBlock, self).__init__()
        self.M = 2
        self.out_channels = out_channel
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc1 = nn.Sequential(nn.Conv2d(self.out_channels, 8, 1, bias=False),
                               nn.BatchNorm2d(8),
                               nn.ReLU(inplace=False))   
        self.fc2 = nn.Conv2d(8, self.out_channels, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        output = []
        # the part of split
        x1, x2 = torch.chunk(x, 2, 1) 
        output.append(x1) 
        output.append(x2) 
        # the part of fusion
        Vf = self.global_pool(x) 
        a = Vf.reshape(batch_size, self.M, self.out_channels // 2, -1)
        a = self.softmax(a) 
        a = list(a.chunk(self.M, dim=1))
        a = list(map(lambda x:x.reshape(batch_size, self.out_channels // 2, 1, 1), a))
        V1 = list(map(lambda x, y:x*y, output, a))
        z=self.fc1(Vf)           
        a_b = self.fc2(z)       
        a_b = a_b.reshape(batch_size, self.M, self.out_channels // 2, -1)
        a_b = self.softmax(a_b)  
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1)) 
        a_b = list(map(lambda x:x.reshape(batch_size, self.out_channels // 2, 1, 1), a_b)) 
        V2 = list(map(lambda x, y:x*y, output, a_b)) 
        V1 = V1[0] + V2[0]
        V2 = V1[1] + V2[1]
        O = torch.cat([V1, V2], 1)
        return F.relu(O)
        
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
       
        self.blk1_1 = ResBlk(16,24)
        self.blk2_1 = ResBlk(16,24)
        self.blk3_1 = ResBlk(16,24)
        self.blk1_2 = ResBlk(24,32)
        self.blk2_2 = ResBlk(24,32)
        self.blk3_2 = ResBlk(24,32)
        self.blk3 = ResBlk(64,64)
        self.blk4 = ResBlk(128,128)
        self.Ablock_32  = ABlock(32)
        self.Bblock_32  = BBlock(32)
        self.Ablock_64  = ABlock(64)
        self.Bblock_64  = BBlock(64)
        self.RP1_1 = nn.Sequential(
            nn.Conv2d(1, 4, 5, 1, 2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) 
        self.RP2_1 = nn.Sequential(
            nn.Conv2d(1, 4, 5, 1, 2, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) 
        self.RP1_2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) 
        self.RP2_2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=False)
        ) 
        self.R3 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False)
        ) 
        self.RP2_4 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16)
        )
        self.RP3_4 = nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16)
        )
        self.RP1_4= nn.Sequential(
            nn.Conv2d(8,16,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(16)
        )
        self.fc1 = nn.Linear(in_features=8192, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=12)
    
    def forward(self,ms,pan,mshpan):
        
        # PAN_fusion
        out_top = self.RP1_1(pan)
        out_top = self.RP1_2(out_top)
        out_top = self.RP1_4(out_top)
        out_top = self.blk1_1(out_top)
        out_top = self.blk1_2(out_top)
        out_top_A = self.Ablock_32(out_top)

        # PAN + MS_H
        out_center = self.RP2_1(mshpan)
        out_center = self.RP2_2(out_center)
        out_center = self.RP2_4(out_center)
        out_center = self.blk2_1(out_center)
        out_center = self.blk2_2(out_center)
        out_center_A = self.Ablock_32(out_center)
        out_center_B = self.Bblock_32(out_center)

        #　MS_fusion
        out_bottom = self.R3(ms)
        out_bottom = self.RP3_4(out_bottom)
        out_bottom = self.blk3_1(out_bottom)
        out_bottom = self.blk3_2(out_bottom)
        out_bottom_B = self.Bblock_32(out_bottom)


        # middle fusion
        input_blk3_top = torch.cat([out_top_A,out_center_A], 1)   # 深度通道拼接
        input_blk3_top = self.Ablock_64(input_blk3_top)
        out_blk3_top = self.blk3(input_blk3_top)

        input_blk3_bottom = torch.cat([out_center_B,out_bottom_B], 1)   # 深度通道拼接
        input_blk3_bottom = self.Ablock_64(input_blk3_bottom)
        out_blk3_bottom = self.blk3(input_blk3_bottom)
    
        # deep fusion
        input_blk4 = torch.cat([out_blk3_top,out_blk3_bottom], 1)   # 深度通道拼接
        out_blk4 = self.blk4(input_blk4)
  
        # 全连接分类
        input_FC = out_blk4.view(out_blk4.shape[0], -1)
        input_FC = F.relu(self.fc1(input_FC))
        input_FC = F.relu(self.fc2(input_FC))
        out_FC = self.fc3(input_FC)

        return out_FC

if __name__ == "__main__":
    pan = torch.randn(2, 1, 64, 64)
    ms = torch.randn(2, 4, 16, 16)
    mshpan = torch.randn(2,1,64,64)
    grf_net = ResNet18()
    out_result = grf_net(ms,pan,mshpan)
    print(out_result)
    print(out_result.shape)
