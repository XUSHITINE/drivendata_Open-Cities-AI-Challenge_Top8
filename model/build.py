'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-02-29 12:12:10
'''
import model.net as Net
import pdb
import torch
import torch.nn as nn


def bulid_model(cfg_model,pretrain_path):
    cfg_model = cfg_model.copy()
    model_type = cfg_model.pop('type')
    if hasattr(Net,model_type):
        model = getattr(Net,model_type)(**cfg_model)
    else:
        print("model is not defined !")
        pdb.set_trace()

    if pretrain_path != "":
        state_dict = torch.load(pretrain_path,map_location='cpu')['model']
        model.load_state_dict(state_dict)
    return model    


class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self,inputs,targets):
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        return torch.unsqueeze(loss,0),outputs
    

def DataParallel_withLoss(model,loss,**kwargs):
    model=FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids=kwargs['device_ids']
    else:
        device_ids=None

    cudaID = device_ids[0]
    model=torch.nn.DataParallel(model, device_ids=device_ids).cuda(cudaID)
    return model