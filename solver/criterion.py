'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-01 22:17:27
Description : 
'''
import torch
from torch import nn
import numpy as np
from collections import OrderedDict

class LabelSmoothing(nn.Module):
    def __init__(self,win_size=11,num_classes=2,smoothing=0.1,fix_flag=True,Max_Smootning=0.2):
        # win_size: 过渡带窗口大小，统计边界及类别交界，为了实现方便，必须为整数
        # num_classes : 类别数
        # smoothing : 平滑比例
        # fix_flag : True 则固定平滑参数，False 则动态统计输入batch中过渡带像素占总输入像素比例
        super(LabelSmoothing, self).__init__()
        assert (win_size%2) == 1
        self.fix_flag = fix_flag
        self.smoothing = smoothing /(num_classes-1)
        self.win_size = win_size
        self.num_classes = num_classes
        self.Max_Smootning = Max_Smootning
        self.find_edge_Conv = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=win_size,padding=(win_size-1)//2,stride=1,bias=False)
        
        new_state_dict = OrderedDict()
        weight = torch.zeros(1,1,win_size,win_size)
        weight = weight -1
        weight[:,:,win_size//2,win_size//2] = win_size*win_size - 1
        new_state_dict['weight'] = weight
        self.find_edge_Conv.load_state_dict(new_state_dict)
        self.find_edge_Conv.weight.requires_grad=False

    def to_categorical(self,y,alpha=0.05,num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical = categorical + alpha
        categorical[np.arange(n), y] = (1-alpha) + (alpha/self.num_classes)
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        categorical = categorical.transpose(0,3,1,2)
        return categorical
        

    def forward(self, x, target):
        assert x.size(1) == self.num_classes
        log_p = nn.functional.log_softmax(x,dim=1)
        self.find_edge_Conv.cuda(device=target.device)
        edge_mask = self.find_edge_Conv(target)
        edge_mask = edge_mask.data.cpu().numpy()
        edge_mask[edge_mask!=0] = 1

        if self.fix_flag:
            pass
        else:
            self.smoothing = np.mean(edge_mask)
            
        if self.smoothing > self.Max_Smootning: 
            self.smoothing = self.Max_Smootning
        
        target = target.squeeze(dim=1)
        target = target.data.cpu().numpy()
        onehot_mask = self.to_categorical(target,0,num_classes=self.num_classes)
        onehot_mask = onehot_mask*(1-edge_mask)
        softlabel_mask = self.to_categorical(target,alpha=self.smoothing,num_classes=self.num_classes)
        softlabel_mask = softlabel_mask*edge_mask
        onehot_mask = torch.from_numpy(onehot_mask).cuda(device=log_p.device).float()
        softlabel_mask = torch.from_numpy(softlabel_mask).cuda(device=log_p.device).float()
        loss = torch.sum(onehot_mask*log_p+softlabel_mask*log_p,dim=1).mean()
        return -loss



class cross_entropy2d(nn.Module):
    def __init__(self):
        super(cross_entropy2d,self).__init__()
    def forward(self,input, target, weight=None, size_average=True):
        if weight:
            weight = torch.tensor(weight,device=target.device)
        # print(input.shape,target.shape)
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        log_p = nn.functional.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        
        loss = nn.functional.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
        if size_average:
            loss /= mask.data.sum()
        return loss

class DataParallel_Loss(nn.Module): # 待修改
    def __init__(self):
        super(DataParallel_Loss,self).__init__()
    def forward(self,input,target):
        return input

if __name__ == "__main__":
    loss = DataParallel_Loss()